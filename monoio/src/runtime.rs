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

// ── Reactor stall detector (phase-as/stall-detector) ─────────────────────────
//
// When `ADAMAS_STALL_DETECTOR_MS=N` is set (N > 0), each task `poll()` is
// bracketed by a monotonic timestamp. If the poll takes longer than N ms, an
// event is published via the shard-registered callback so adamas can log it and
// bump its per-shard stall counters.
//
// **Zero overhead when off.** `STALL_HOOK` is `None` by default. The only
// hot-path cost when unset is a single `Option::is_some()` check per task
// run — one branch on a thread-local, predicted-not-taken.
//
// **No extra threads.** The detector runs inline in each shard's own reactor
// loop. There is no watchdog thread, no signal handler.

/// Per-stall event delivered to the registered callback.
#[derive(Debug, Clone)]
pub struct StallEvent {
    /// Duration of the stalling poll in milliseconds.
    pub duration_ms: u64,
}

/// Type of the per-shard stall callback.
///
/// The callback is called inline from the reactor loop with the shard holding
/// no locks. It must be cheap (bump an atomic counter, write to a lock-free
/// log). It must never block or yield.
pub type StallCallback = fn(StallEvent);

thread_local! {
    /// Per-shard stall-detector hook. `None` = off (zero overhead).
    ///
    /// Set by adamas at shard startup when `ADAMAS_STALL_DETECTOR_MS` is present.
    /// Cleared on shard shutdown.
    static STALL_HOOK: std::cell::Cell<Option<(u64, StallCallback)>> =
        std::cell::Cell::new(None);
}

/// Register a stall callback for the current shard.
///
/// `threshold_ms` — minimum task poll duration (in milliseconds) that triggers
/// a callback. Pass 0 to disable.
///
/// Must be called from the shard's own reactor thread before `block_on`.
pub fn register_stall_callback(threshold_ms: u64, cb: StallCallback) {
    if threshold_ms == 0 {
        STALL_HOOK.with(|h| h.set(None));
    } else {
        STALL_HOOK.with(|h| h.set(Some((threshold_ms, cb))));
    }
}

/// Remove the stall callback for the current shard (disables detection).
pub fn unregister_stall_callback() {
    STALL_HOOK.with(|h| h.set(None));
}

/// Inline macro: run one `Task<LocalScheduler>`, optionally measuring duration.
///
/// When the stall hook is absent (`None`), this compiles down to a plain
/// `$task.run()` with a single not-taken branch on the thread-local — zero
/// overhead in the off case.  We use a macro (not a generic fn) because
/// `Task<S>` has `run(self)` but no common trait; the macro keeps the
/// call-site type concrete.
macro_rules! run_task {
    ($task:expr) => {{
        let hook = STALL_HOOK.with(|h| h.get());
        match hook {
            None => $task.run(),
            Some((threshold_ms, cb)) => {
                let t0 = std::time::Instant::now();
                $task.run();
                let elapsed_ms = t0.elapsed().as_millis() as u64;
                if elapsed_ms >= threshold_ms {
                    cb(StallEvent { duration_ms: elapsed_ms });
                }
            }
        }
    }};
}

#[cfg(feature = "sync")]
thread_local! {
    pub(crate) static DEFAULT_CTX: Context = Context {
        thread_id: crate::utils::thread_id::DEFAULT_THREAD_ID,
        unpark_cache: std::cell::RefCell::new(rustc_hash::FxHashMap::default()),
        waker_sender_cache: std::cell::RefCell::new(rustc_hash::FxHashMap::default()),
        tasks: Default::default(),
        time_handle: None,
        blocking_handle: crate::blocking::BlockingHandle::Empty(crate::blocking::BlockingStrategy::Panic),
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
        std::cell::RefCell<rustc_hash::FxHashMap<usize, crate::driver::UnparkHandle>>,

    /// Waker sender cache
    #[cfg(feature = "sync")]
    pub(crate) waker_sender_cache:
        std::cell::RefCell<rustc_hash::FxHashMap<usize, flume::Sender<std::task::Waker>>>,

    /// Time Handle
    pub(crate) time_handle: Option<TimeHandle>,

    /// Blocking Handle
    #[cfg(feature = "sync")]
    pub(crate) blocking_handle: crate::blocking::BlockingHandle,
}

impl Context {
    #[cfg(feature = "sync")]
    pub(crate) fn new(blocking_handle: crate::blocking::BlockingHandle) -> Self {
        let thread_id = crate::builder::BUILD_THREAD_ID.with(|id| *id);

        Self {
            thread_id,
            unpark_cache: std::cell::RefCell::new(rustc_hash::FxHashMap::default()),
            waker_sender_cache: std::cell::RefCell::new(rustc_hash::FxHashMap::default()),
            tasks: TaskQueue::default(),
            time_handle: None,
            blocking_handle,
        }
    }

    #[cfg(not(feature = "sync"))]
    pub(crate) fn new() -> Self {
        let thread_id = crate::builder::BUILD_THREAD_ID.with(|id| *id);

        Self {
            thread_id,
            tasks: TaskQueue::default(),
            time_handle: None,
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

        // AR.2: preempt timer — after this many µs of continuous task execution,
        // break to io_uring to check for I/O completions before continuing.
        // Prevents long task chains from starving I/O wakeups.
        // Default 5ms; override with MONOIO_PREEMPT_US env var.
        let preempt_us: u64 = std::env::var("MONOIO_PREEMPT_US")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5_000);

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
                        // AR.2: time-bounded task drain.
                        // Drain high-priority tasks first, then background, but break
                        // to I/O whenever preempt_us elapses so completions aren't
                        // delayed by a burst of CPU-bound tasks.
                        let preempt_deadline =
                            std::time::Instant::now() + std::time::Duration::from_micros(preempt_us);

                        // Phase 1: drain high-priority tasks until preempted or empty.
                        while let Some(t) = self.context.tasks.pop_high() {
                            run_task!(t);
                            if std::time::Instant::now() >= preempt_deadline {
                                break;
                            }
                        }

                        // Phase 2: if high queue empty, run one background task then
                        // recheck high queue (avoids background starvation).
                        if self.context.tasks.high_is_empty() {
                            if let Some(t) = self.context.tasks.pop() {
                                run_task!(t);
                            }
                        }

                        // Check main future
                        while should_poll() {
                            if let std::task::Poll::Ready(t) = join.as_mut().poll(cx) {
                                return t;
                            }
                        }

                        if self.context.tasks.is_empty() {
                            // No task to execute, wait for I/O blockingly.
                            break;
                        }

                        // Cold path: submit pending SQEs before looping.
                        let _ = self.driver.submit();
                    }

                    // AR.3: adaptive park — if high-priority tasks are already
                    // queued (woken by a prior I/O completion), do a non-blocking
                    // CQ drain (park_timeout=0) instead of a blocking park.
                    // This eliminates the ~0.5ms wakeup latency for the next
                    // batch of writes when the shard is saturated.
                    // When the high queue is empty, fall back to the normal
                    // blocking park to avoid busy-waiting on idle shards.
                    #[cfg(not(all(debug_assertions, feature = "debug")))]
                    {
                        if !self.context.tasks.high_is_empty() {
                            let _ = self.driver.park_timeout(std::time::Duration::ZERO);
                        } else {
                            let _ = self.driver.park();
                        }
                    }

                    #[cfg(all(debug_assertions, feature = "debug"))]
                    if let Err(e) = if !self.context.tasks.high_is_empty() {
                        self.driver.park_timeout(std::time::Duration::ZERO)
                    } else {
                        self.driver.park()
                    } {
                        trace!("park error: {:?}", e);
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
    spawn_with_priority(future, crate::scheduler::TaskPriority::High)
}

/// Spawn with explicit priority (AR.1: two-tier task scheduler).
///
/// Use [`TaskPriority::High`] for write-path tasks (CQL handlers, coordinator
/// futures) and [`TaskPriority::Background`] for compaction, repair, flush.
pub fn spawn_with_priority<T>(future: T, priority: crate::scheduler::TaskPriority) -> JoinHandle<T::Output>
where
    T: Future + 'static,
    T::Output: 'static,
{
    let (task, join) = new_task(
        crate::utils::thread_id::get_current_thread_id(),
        future,
        LocalScheduler { priority },
    );

    CURRENT.with(|ctx| {
        ctx.tasks.push(task, priority);
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
        LocalScheduler { priority: crate::scheduler::TaskPriority::High },
    );

    CURRENT.with(|ctx| {
        ctx.tasks.push(task, crate::scheduler::TaskPriority::High);
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
}
