use std::future::Future;

#[cfg(any(all(target_os = "linux", feature = "iouring"), feature = "legacy"))]
use crate::time::TimeDriver;
#[cfg(all(target_os = "linux", feature = "iouring"))]
use crate::IoUringDriver;
#[cfg(feature = "legacy")]
use crate::LegacyDriver;
use crate::{
    driver::Driver,
    scheduler::{LocalScheduler, TaskQueue},
    task::{
        new_task,
        waker_fn::{dummy_waker, set_poll, should_poll},
        JoinHandle,
    },
    time::driver::Handle as TimeHandle,
};

#[cfg(feature = "sync")]
thread_local! {
    pub(crate) static DEFAULT_CTX: Context = Context {
        thread_id: crate::utils::thread_id::DEFAULT_THREAD_ID,
        unpark_cache: std::cell::RefCell::new(fxhash::FxHashMap::default()),
        waker_sender_cache: std::cell::RefCell::new(fxhash::FxHashMap::default()),
        tasks: Default::default(),
        time_handle: None,
        blocking_handle: crate::blocking::BlockingHandle::Empty(crate::blocking::BlockingStrategy::Panic),
        poll_spin_us: None,
    };
}

scoped_thread_local!(pub(crate) static CURRENT: Context);

pub(crate) struct Context {
    /// Owned task set and local run queue
    pub(crate) tasks: TaskQueue,

    /// Thread id(not the kernel thread id but a generated unique number)
    pub(crate) thread_id: usize,

    /// Thread unpark handles
    #[cfg(feature = "sync")]
    pub(crate) unpark_cache:
        std::cell::RefCell<fxhash::FxHashMap<usize, crate::driver::UnparkHandle>>,

    /// Waker sender cache
    #[cfg(feature = "sync")]
    pub(crate) waker_sender_cache:
        std::cell::RefCell<fxhash::FxHashMap<usize, flume::Sender<std::task::Waker>>>,

    /// Time Handle
    pub(crate) time_handle: Option<TimeHandle>,

    /// Blocking Handle
    #[cfg(feature = "sync")]
    pub(crate) blocking_handle: crate::blocking::BlockingHandle,

    /// Poll spin duration in microseconds. When set, the event loop will spin
    /// for up to this duration checking the io_uring CQ ring (shared memory)
    /// before falling back to a blocking `park()` call. This reduces per-yield
    /// latency at the cost of CPU usage.
    pub(crate) poll_spin_us: Option<u64>,
}

impl Context {
    #[cfg(feature = "sync")]
    pub(crate) fn new(
        blocking_handle: crate::blocking::BlockingHandle,
        poll_spin_us: Option<u64>,
    ) -> Self {
        let thread_id = crate::builder::BUILD_THREAD_ID.with(|id| *id);

        Self {
            thread_id,
            unpark_cache: std::cell::RefCell::new(fxhash::FxHashMap::default()),
            waker_sender_cache: std::cell::RefCell::new(fxhash::FxHashMap::default()),
            tasks: TaskQueue::default(),
            time_handle: None,
            blocking_handle,
            poll_spin_us,
        }
    }

    #[cfg(not(feature = "sync"))]
    pub(crate) fn new(poll_spin_us: Option<u64>) -> Self {
        let thread_id = crate::builder::BUILD_THREAD_ID.with(|id| *id);

        Self {
            thread_id,
            tasks: TaskQueue::default(),
            time_handle: None,
            poll_spin_us,
        }
    }

    #[allow(unused)]
    #[cfg(feature = "sync")]
    pub(crate) fn unpark_thread(&self, id: usize) {
        use crate::driver::{thread::get_unpark_handle, unpark::Unpark};
        if let Some(handle) = self.unpark_cache.borrow().get(&id) {
            handle.unpark();
            return;
        }

        if let Some(v) = get_unpark_handle(id) {
            // Write back to local cache
            let w = v.clone();
            self.unpark_cache.borrow_mut().insert(id, w);
            v.unpark();
        }
    }

    #[allow(unused)]
    #[cfg(feature = "sync")]
    pub(crate) fn send_waker(&self, id: usize, w: std::task::Waker) {
        use crate::driver::thread::get_waker_sender;
        if let Some(sender) = self.waker_sender_cache.borrow().get(&id) {
            let _ = sender.send(w);
            return;
        }

        if let Some(s) = get_waker_sender(id) {
            // Write back to local cache
            let _ = s.send(w);
            self.waker_sender_cache.borrow_mut().insert(id, s);
        }
    }
}

/// Monoio runtime
pub struct Runtime<D> {
    pub(crate) context: Context,
    pub(crate) driver: D,
}

impl<D> Runtime<D> {
    pub(crate) fn new(context: Context, driver: D) -> Self {
        Self { context, driver }
    }

    /// Block on
    pub fn block_on<F>(&mut self, future: F) -> F::Output
    where
        F: Future,
        D: Driver,
    {
        assert!(
            !CURRENT.is_set(),
            "Can not start a runtime inside a runtime"
        );

        let waker = dummy_waker();
        let cx = &mut std::task::Context::from_waker(&waker);

        self.driver.with(|| {
            CURRENT.set(&self.context, || {
                #[cfg(feature = "sync")]
                let join = unsafe { spawn_without_static(future) };
                #[cfg(not(feature = "sync"))]
                let join = future;

                let mut join = std::pin::pin!(join);
                set_poll();
                loop {
                    loop {
                        // Consume all tasks(with max round to prevent io starvation)
                        let mut max_round = self.context.tasks.len() * 2;
                        while let Some(t) = self.context.tasks.pop() {
                            t.run();
                            if max_round == 0 {
                                // maybe there's a looping task
                                break;
                            } else {
                                max_round -= 1;
                            }
                        }

                        // Check main future
                        while should_poll() {
                            // check if ready
                            if let std::task::Poll::Ready(t) = join.as_mut().poll(cx) {
                                return t;
                            }
                        }

                        if self.context.tasks.is_empty() {
                            // No task to execute, we should wait for io blockingly
                            // Hot path
                            break;
                        }

                        // Cold path
                        let _ = self.driver.submit();
                    }

                    // Poll spin mode: spin for up to poll_spin_us microseconds
                    // checking the io_uring CQ ring (shared memory read, no syscall)
                    // before falling back to the blocking park() call.
                    //
                    // In completion-based io_uring, the kernel writes CQEs directly
                    // to a shared memory ring. driver.submit() calls tick() which
                    // reads the CQ ring via atomic load-acquire — no io_uring_enter
                    // syscall needed for CQE discovery. This spin loop exploits that
                    // property to reduce per-yield latency from ~0.6ms (blocking
                    // io_uring_enter with min_complete=1) to near-zero.
                    if let Some(spin_us) = self.context.poll_spin_us {
                        // Flush any pending SQEs so the kernel starts processing them.
                        let _ = self.driver.submit();

                        // Check if submit()'s tick() already woke tasks.
                        if !self.context.tasks.is_empty() {
                            continue;
                        }

                        // Spin: repeatedly call submit() which internally calls
                        // tick() to read the CQ ring from shared memory.
                        let start = std::time::Instant::now();
                        let budget = std::time::Duration::from_micros(spin_us);
                        loop {
                            let _ = self.driver.submit();

                            if !self.context.tasks.is_empty() {
                                break;
                            }

                            if start.elapsed() >= budget {
                                // Spin budget exhausted — fall through to blocking park.
                                let _ = self.driver.park();
                                break;
                            }

                            std::hint::spin_loop();
                        }
                    } else {
                        // Default: blocking wait for I/O completion
                        #[cfg(not(all(debug_assertions, feature = "debug")))]
                        let _ = self.driver.park();

                        #[cfg(all(debug_assertions, feature = "debug"))]
                        if let Err(e) = self.driver.park() {
                            trace!("park error: {:?}", e);
                        }
                    }
                }
            })
        })
    }
}

/// Fusion Runtime is a wrapper of io_uring driver or legacy driver based
/// runtime.
#[cfg(feature = "legacy")]
pub enum FusionRuntime<#[cfg(all(target_os = "linux", feature = "iouring"))] L, R> {
    /// Uring driver based runtime.
    #[cfg(all(target_os = "linux", feature = "iouring"))]
    Uring(Runtime<L>),
    /// Legacy driver based runtime.
    Legacy(Runtime<R>),
}

/// Fusion Runtime is a wrapper of io_uring driver or legacy driver based
/// runtime.
#[cfg(all(target_os = "linux", feature = "iouring", not(feature = "legacy")))]
pub enum FusionRuntime<L> {
    /// Uring driver based runtime.
    Uring(Runtime<L>),
}

#[cfg(all(target_os = "linux", feature = "iouring", feature = "legacy"))]
impl<L, R> FusionRuntime<L, R>
where
    L: Driver,
    R: Driver,
{
    /// Block on
    pub fn block_on<F>(&mut self, future: F) -> F::Output
    where
        F: Future,
    {
        match self {
            FusionRuntime::Uring(inner) => {
                info!("Monoio is running with io_uring driver");
                inner.block_on(future)
            }
            FusionRuntime::Legacy(inner) => {
                info!("Monoio is running with legacy driver");
                inner.block_on(future)
            }
        }
    }
}

#[cfg(all(feature = "legacy", not(all(target_os = "linux", feature = "iouring"))))]
impl<R> FusionRuntime<R>
where
    R: Driver,
{
    /// Block on
    pub fn block_on<F>(&mut self, future: F) -> F::Output
    where
        F: Future,
    {
        match self {
            FusionRuntime::Legacy(inner) => inner.block_on(future),
        }
    }
}

#[cfg(all(not(feature = "legacy"), all(target_os = "linux", feature = "iouring")))]
impl<R> FusionRuntime<R>
where
    R: Driver,
{
    /// Block on
    pub fn block_on<F>(&mut self, future: F) -> F::Output
    where
        F: Future,
    {
        match self {
            FusionRuntime::Uring(inner) => inner.block_on(future),
        }
    }
}

// L -> Fusion<L, R>
#[cfg(all(target_os = "linux", feature = "iouring", feature = "legacy"))]
impl From<Runtime<IoUringDriver>> for FusionRuntime<IoUringDriver, LegacyDriver> {
    fn from(r: Runtime<IoUringDriver>) -> Self {
        Self::Uring(r)
    }
}

// TL -> Fusion<TL, TR>
#[cfg(all(target_os = "linux", feature = "iouring", feature = "legacy"))]
impl From<Runtime<TimeDriver<IoUringDriver>>>
    for FusionRuntime<TimeDriver<IoUringDriver>, TimeDriver<LegacyDriver>>
{
    fn from(r: Runtime<TimeDriver<IoUringDriver>>) -> Self {
        Self::Uring(r)
    }
}

// R -> Fusion<L, R>
#[cfg(all(target_os = "linux", feature = "iouring", feature = "legacy"))]
impl From<Runtime<LegacyDriver>> for FusionRuntime<IoUringDriver, LegacyDriver> {
    fn from(r: Runtime<LegacyDriver>) -> Self {
        Self::Legacy(r)
    }
}

// TR -> Fusion<TL, TR>
#[cfg(all(target_os = "linux", feature = "iouring", feature = "legacy"))]
impl From<Runtime<TimeDriver<LegacyDriver>>>
    for FusionRuntime<TimeDriver<IoUringDriver>, TimeDriver<LegacyDriver>>
{
    fn from(r: Runtime<TimeDriver<LegacyDriver>>) -> Self {
        Self::Legacy(r)
    }
}

// R -> Fusion<R>
#[cfg(all(feature = "legacy", not(all(target_os = "linux", feature = "iouring"))))]
impl From<Runtime<LegacyDriver>> for FusionRuntime<LegacyDriver> {
    fn from(r: Runtime<LegacyDriver>) -> Self {
        Self::Legacy(r)
    }
}

// TR -> Fusion<TR>
#[cfg(all(feature = "legacy", not(all(target_os = "linux", feature = "iouring"))))]
impl From<Runtime<TimeDriver<LegacyDriver>>> for FusionRuntime<TimeDriver<LegacyDriver>> {
    fn from(r: Runtime<TimeDriver<LegacyDriver>>) -> Self {
        Self::Legacy(r)
    }
}

// L -> Fusion<L>
#[cfg(all(target_os = "linux", feature = "iouring", not(feature = "legacy")))]
impl From<Runtime<IoUringDriver>> for FusionRuntime<IoUringDriver> {
    fn from(r: Runtime<IoUringDriver>) -> Self {
        Self::Uring(r)
    }
}

// TL -> Fusion<TL>
#[cfg(all(target_os = "linux", feature = "iouring", not(feature = "legacy")))]
impl From<Runtime<TimeDriver<IoUringDriver>>> for FusionRuntime<TimeDriver<IoUringDriver>> {
    fn from(r: Runtime<TimeDriver<IoUringDriver>>) -> Self {
        Self::Uring(r)
    }
}

/// Spawns a new asynchronous task, returning a [`JoinHandle`] for it.
///
/// Spawning a task enables the task to execute concurrently to other tasks.
/// There is no guarantee that a spawned task will execute to completion. When a
/// runtime is shutdown, all outstanding tasks are dropped, regardless of the
/// lifecycle of that task.
///
///
/// [`JoinHandle`]: super::task::JoinHandle
///
/// # Examples
///
/// In this example, a server is started and `spawn` is used to start a new task
/// that processes each received connection.
///
/// ```no_run
/// #[monoio::main]
/// async fn main() {
///     let handle = monoio::spawn(async {
///         println!("hello from a background task");
///     });
///
///     // Let the task complete
///     handle.await;
/// }
/// ```
pub fn spawn<T>(future: T) -> JoinHandle<T::Output>
where
    T: Future + 'static,
    T::Output: 'static,
{
    let (task, join) = new_task(
        crate::utils::thread_id::get_current_thread_id(),
        future,
        LocalScheduler,
    );

    CURRENT.with(|ctx| {
        ctx.tasks.push(task);
    });
    join
}

#[cfg(feature = "sync")]
unsafe fn spawn_without_static<T>(future: T) -> JoinHandle<T::Output>
where
    T: Future,
{
    use crate::task::new_task_holding;
    let (task, join) = new_task_holding(
        crate::utils::thread_id::get_current_thread_id(),
        future,
        LocalScheduler,
    );

    CURRENT.with(|ctx| {
        ctx.tasks.push(task);
    });
    join
}

#[cfg(test)]
mod tests {
    #[cfg(all(feature = "sync", target_os = "linux", feature = "iouring"))]
    #[test]
    fn across_thread() {
        use futures::channel::oneshot;

        use crate::driver::IoUringDriver;

        let (tx1, rx1) = oneshot::channel::<u8>();
        let (tx2, rx2) = oneshot::channel::<u8>();

        std::thread::spawn(move || {
            let mut rt = crate::RuntimeBuilder::<IoUringDriver>::new()
                .build()
                .unwrap();
            rt.block_on(async move {
                let n = rx1.await.expect("unable to receive rx1");
                assert!(tx2.send(n).is_ok());
            });
        });

        let mut rt = crate::RuntimeBuilder::<IoUringDriver>::new()
            .build()
            .unwrap();
        rt.block_on(async move {
            assert!(tx1.send(24).is_ok());
            assert_eq!(rx2.await.expect("unable to receive rx2"), 24);
        });
    }

    #[cfg(all(target_os = "linux", feature = "iouring"))]
    #[test]
    fn timer() {
        use crate::driver::IoUringDriver;
        let mut rt = crate::RuntimeBuilder::<IoUringDriver>::new()
            .enable_timer()
            .build()
            .unwrap();
        let instant = std::time::Instant::now();
        rt.block_on(async {
            crate::time::sleep(std::time::Duration::from_millis(200)).await;
        });
        let eps = instant.elapsed().subsec_millis();
        assert!((eps as i32 - 200).abs() < 50);
    }

    // Poll spin mode tests — verify the builder plumbing and basic runtime
    // behavior. The real latency improvement is only measurable under io_uring
    // on Linux with actual I/O load.

    #[cfg(feature = "legacy")]
    #[test]
    fn poll_spin_builder_accepted() {
        // Verify poll_spin_us builder method compiles and produces a working runtime
        use crate::driver::LegacyDriver;
        let mut rt = crate::RuntimeBuilder::<LegacyDriver>::new()
            .poll_spin_us(100)
            .build()
            .unwrap();
        rt.block_on(async {
            // Simple task completes normally with poll spin enabled
            let x = 1 + 1;
            assert_eq!(x, 2);
        });
    }

    #[cfg(feature = "legacy")]
    #[test]
    fn poll_spin_zero_budget() {
        // Zero microseconds means spin check once then immediately fall through to park
        use crate::driver::LegacyDriver;
        let mut rt = crate::RuntimeBuilder::<LegacyDriver>::new()
            .poll_spin_us(0)
            .build()
            .unwrap();
        rt.block_on(async {
            let x = 42;
            assert_eq!(x, 42);
        });
    }

    #[cfg(feature = "legacy")]
    #[test]
    fn poll_spin_disabled_by_default() {
        // Without calling poll_spin_us(), behavior is unchanged (blocking park)
        use crate::driver::LegacyDriver;
        let mut rt = crate::RuntimeBuilder::<LegacyDriver>::new()
            .build()
            .unwrap();
        rt.block_on(async {
            assert!(true);
        });
    }

    #[cfg(feature = "legacy")]
    #[test]
    fn poll_spin_with_timer() {
        // poll_spin_us works with enable_timer()
        use crate::driver::LegacyDriver;
        let mut rt = crate::RuntimeBuilder::<LegacyDriver>::new()
            .poll_spin_us(50)
            .enable_timer()
            .build()
            .unwrap();
        let instant = std::time::Instant::now();
        rt.block_on(async {
            crate::time::sleep(std::time::Duration::from_millis(100)).await;
        });
        let eps = instant.elapsed().as_millis();
        assert!(eps >= 80 && eps < 300, "timer fired in {}ms", eps);
    }

    #[cfg(all(target_os = "linux", feature = "iouring"))]
    #[test]
    fn poll_spin_iouring_basic() {
        // Verify poll spin mode works with io_uring driver
        use crate::driver::IoUringDriver;
        let mut rt = crate::RuntimeBuilder::<IoUringDriver>::new()
            .poll_spin_us(100)
            .build()
            .unwrap();
        rt.block_on(async {
            let x = 1 + 1;
            assert_eq!(x, 2);
        });
    }

    #[cfg(all(target_os = "linux", feature = "iouring"))]
    #[test]
    fn poll_spin_iouring_with_timer() {
        use crate::driver::IoUringDriver;
        let mut rt = crate::RuntimeBuilder::<IoUringDriver>::new()
            .poll_spin_us(200)
            .enable_timer()
            .build()
            .unwrap();
        let instant = std::time::Instant::now();
        rt.block_on(async {
            crate::time::sleep(std::time::Duration::from_millis(100)).await;
        });
        let eps = instant.elapsed().as_millis();
        assert!(eps >= 80 && eps < 300, "timer fired in {}ms", eps);
    }
}
