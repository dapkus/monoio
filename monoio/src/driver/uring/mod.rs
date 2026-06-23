//! Monoio Uring Driver.

use std::{
    cell::UnsafeCell,
    io,
    mem::ManuallyDrop,
    os::unix::prelude::{AsRawFd, RawFd},
    rc::Rc,
    task::{Context, Poll},
    time::Duration,
};

use io_uring::{cqueue, opcode, types::Timespec, IoUring};
use lifecycle::MaybeFdLifecycle;

use super::{
    op::{CompletionMeta, Op, OpAble},
    // ready::Ready,
    // scheduled_io::ScheduledIo,
    util::timespec,
    Driver,
    Inner,
    CURRENT,
};
use crate::utils::slab::Slab;

mod lifecycle;
#[cfg(feature = "sync")]
mod waker;
#[cfg(feature = "sync")]
pub(crate) use waker::UnparkHandle;

#[allow(unused)]
pub(crate) const CANCEL_USERDATA: u64 = u64::MAX;
pub(crate) const TIMEOUT_USERDATA: u64 = u64::MAX - 1;
#[allow(unused)]
pub(crate) const EVENTFD_USERDATA: u64 = u64::MAX - 2;
#[cfg(feature = "poll-io")]
pub(crate) const POLLER_USERDATA: u64 = u64::MAX - 3;

pub(crate) const MIN_REVERSED_USERDATA: u64 = u64::MAX - 3;

/// Default adaptive-spin budget (µs) for the poll-reactor's block decision when
/// the builder's `poll_spin_us` is unset. When a CQ poll reaps nothing the
/// reactor spins up to this long for the next completion before blocking — long
/// enough to bridge the inter-batch gaps that occur under load (so we don't
/// issue a context-switching `submit_and_wait(1)` on every gap), short enough
/// that at true idle the spin cost before sleeping is negligible.
pub(crate) const DEFAULT_POLL_REACTOR_SPIN_US: u64 = 50;

/// Driver with uring.
pub struct IoUringDriver {
    inner: Rc<UnsafeCell<UringInner>>,

    /// Poll spin duration in microseconds. When set, the driver will spin-poll
    /// the io_uring CQ ring (shared memory read, no syscall) before falling
    /// back to a blocking io_uring_enter. This reduces per-yield latency at
    /// the cost of CPU usage.
    poll_spin_us: Option<u64>,

    /// Adaptive non-blocking reactor mode. When true, the driver never calls
    /// `io_uring_enter(min_complete=1)` while I/O ops are in flight. Instead:
    ///   1. `submit()` — non-blocking push SQEs to kernel
    ///   2. `tick()` — read CQEs from mmap'd CQ ring (no syscall)
    ///   3. Return immediately — the runtime task drain loop IS the spin
    ///   4. Block only when truly idle (no in-flight ops)
    ///
    /// This eliminates the per-yield context switch overhead (~0.6ms) that
    /// dominates latency in the blocking reactor path.
    poll_reactor: bool,

    /// Client-gated variant of the poll-reactor. When true (and `poll_reactor`
    /// is also true), the bounded adaptive spin before blocking runs only while
    /// a *client* request is in flight on this shard (see
    /// [`crate::client_gate`]). At client idle the reactor falls straight
    /// through to the blocking path instead of busy-polling gossip/internode
    /// completions. No effect unless `poll_reactor` is on.
    poll_reactor_client_gated: bool,

    // Used for drop
    #[cfg(feature = "sync")]
    thread_id: usize,
}

pub(crate) struct UringInner {
    /// In-flight operations
    ops: Ops,

    // Used as timeout buffer
    timespec: Timespec,

    // Used as read eventfd buffer
    #[cfg(feature = "sync")]
    eventfd_read_dst: [u8; 8],

    #[cfg(feature = "poll-io")]
    poll: super::poll::Poll,
    #[cfg(feature = "poll-io")]
    poller_installed: bool,

    /// IoUring bindings
    uring: ManuallyDrop<IoUring>,

    /// Shared waker
    #[cfg(feature = "sync")]
    shared_waker: std::sync::Arc<waker::EventWaker>,

    // Mark if eventfd is in the ring
    #[cfg(feature = "sync")]
    eventfd_installed: bool,

    // Waker receiver
    #[cfg(feature = "sync")]
    waker_receiver: flume::Receiver<std::task::Waker>,

    // Uring support ext_arg
    ext_arg: bool,
}

// When dropping the driver, all in-flight operations must have completed. This
// type wraps the slab and ensures that, on drop, the slab is empty.
struct Ops {
    slab: Slab<MaybeFdLifecycle>,
}

/// Set once after the first warn about an unsupported modern-flags kernel, so the
/// fallback log is emitted at most once per process instead of once per shard.
static MODERN_FLAGS_FALLBACK_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Returns true when `ADAMAS_IOURING_MODERN_FLAGS` is set to a truthy value
/// (`1`/`true`/`yes`/`on`, case-insensitive). Default-off: any other value, an
/// empty value, or an unset var disables the modern setup flags so the A/B is
/// clean and vanilla is the safe baseline.
fn modern_flags_requested() -> bool {
    std::env::var("ADAMAS_IOURING_MODERN_FLAGS")
        .ok()
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "yes" || v == "on"
        })
        .unwrap_or(false)
}

/// Build the io_uring ring, optionally applying the modern setup flags
/// (`IORING_SETUP_SINGLE_ISSUER` + `IORING_SETUP_COOP_TASKRUN`) when
/// `ADAMAS_IOURING_MODERN_FLAGS` is truthy.
///
/// These flags cut the per-completion kernel park-tax on a busy thread-per-core
/// reactor: `COOP_TASKRUN` suppresses the inter-processor interrupt that would
/// otherwise force-run completion task-work on each CQE (completions are reaped
/// on the next ring enter instead), and `SINGLE_ISSUER` lets the kernel skip
/// submitter locking since exactly one thread owns each ring. Both require Linux
/// 5.19+/6.0+; on an older kernel `build()` fails with `EINVAL`.
///
/// Kernel-fallback (T25, no panic): when the modern flags are requested and the
/// flagged `build()` fails, we retry `build()` on the original (bare) builder so
/// an unsupported kernel degrades to vanilla io_uring rather than crashing
/// startup. The fallback is logged once per process.
fn build_uring(urb: &io_uring::Builder, entries: u32) -> io::Result<IoUring> {
    if !modern_flags_requested() {
        return urb.build(entries);
    }

    // Apply the modern flags to a clone so the shared per-runtime builder is left
    // untouched (each per-core driver clones + flags its own ring).
    let mut modern = urb.clone();
    modern.setup_single_issuer().setup_coop_taskrun();
    match modern.build(entries) {
        Ok(ring) => Ok(ring),
        Err(e) => {
            if !MODERN_FLAGS_FALLBACK_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                // Startup-only diagnostic; use eprintln so it is always visible
                // regardless of the `debug`/`tracing` feature gate (the in-crate
                // `info!`/`trace!` macros no-op without it).
                eprintln!(
                    "[monoio] WARN: ADAMAS_IOURING_MODERN_FLAGS set but io_uring build with \
                     SINGLE_ISSUER+COOP_TASKRUN failed ({e}); kernel likely <5.19. \
                     Falling back to vanilla io_uring setup."
                );
            }
            // Retry on the original bare builder — vanilla, always-supported path.
            urb.build(entries)
        }
    }
}

impl IoUringDriver {
    pub(crate) const DEFAULT_ENTRIES: u32 = 1024;

    pub(crate) fn new(b: &io_uring::Builder) -> io::Result<IoUringDriver> {
        Self::new_with_entries(b, Self::DEFAULT_ENTRIES, None, false, false)
    }

    #[cfg(not(feature = "sync"))]
    pub(crate) fn new_with_entries(
        urb: &io_uring::Builder,
        entries: u32,
        poll_spin_us: Option<u64>,
        poll_reactor: bool,
        poll_reactor_client_gated: bool,
    ) -> io::Result<IoUringDriver> {
        let uring = ManuallyDrop::new(build_uring(urb, entries)?);

        let inner = Rc::new(UnsafeCell::new(UringInner {
            #[cfg(feature = "poll-io")]
            poll: super::poll::Poll::with_capacity(entries as usize)?,
            #[cfg(feature = "poll-io")]
            poller_installed: false,
            ops: Ops::new(),
            timespec: Timespec::new(),
            ext_arg: uring.params().is_feature_ext_arg(),
            uring,
        }));

        Ok(IoUringDriver {
            inner,
            poll_spin_us,
            poll_reactor,
            poll_reactor_client_gated,
        })
    }

    #[cfg(feature = "sync")]
    pub(crate) fn new_with_entries(
        urb: &io_uring::Builder,
        entries: u32,
        poll_spin_us: Option<u64>,
        poll_reactor: bool,
        poll_reactor_client_gated: bool,
    ) -> io::Result<IoUringDriver> {
        let uring = ManuallyDrop::new(build_uring(urb, entries)?);

        // Create eventfd and register it to the ring.
        let waker = {
            let fd = crate::syscall!(eventfd@RAW(0, libc::EFD_CLOEXEC))?;
            unsafe {
                use std::os::unix::io::FromRawFd;
                std::fs::File::from_raw_fd(fd)
            }
        };

        let (waker_sender, waker_receiver) = flume::unbounded::<std::task::Waker>();

        let inner = Rc::new(UnsafeCell::new(UringInner {
            #[cfg(feature = "poll-io")]
            poller_installed: false,
            #[cfg(feature = "poll-io")]
            poll: super::poll::Poll::with_capacity(entries as usize)?,
            ops: Ops::new(),
            timespec: Timespec::new(),
            ext_arg: uring.params().is_feature_ext_arg(),
            uring,
            shared_waker: std::sync::Arc::new(waker::EventWaker::new(waker)),
            eventfd_installed: false,
            eventfd_read_dst: [0_u8; 8],
            waker_receiver,
        }));

        let thread_id = crate::builder::BUILD_THREAD_ID.with(|id| *id);
        let driver = IoUringDriver {
            inner,
            poll_spin_us,
            poll_reactor,
            poll_reactor_client_gated,
            thread_id,
        };

        // Register unpark handle
        super::thread::register_unpark_handle(thread_id, driver.unpark().into());
        super::thread::register_waker_sender(thread_id, waker_sender);
        Ok(driver)
    }

    #[allow(unused)]
    fn num_operations(&self) -> usize {
        let inner = self.inner.get();
        unsafe { (*inner).ops.slab.len() }
    }

    // Flush to make enough space
    fn flush_space(inner: &mut UringInner, need: usize) -> io::Result<()> {
        let sq = inner.uring.submission();
        debug_assert!(sq.capacity() >= need);
        if sq.len() + need > sq.capacity() {
            drop(sq);
            inner.submit()?;
        }
        Ok(())
    }

    #[cfg(feature = "sync")]
    fn install_eventfd(&self, inner: &mut UringInner, fd: RawFd) {
        let entry = opcode::Read::new(
            io_uring::types::Fd(fd),
            inner.eventfd_read_dst.as_mut_ptr(),
            8,
        )
        .build()
        .user_data(EVENTFD_USERDATA);

        let mut sq = inner.uring.submission();
        let _ = unsafe { sq.push(&entry) };
        inner.eventfd_installed = true;
    }

    #[cfg(feature = "poll-io")]
    fn install_poller(&self, inner: &mut UringInner, fd: RawFd) {
        let entry = opcode::PollAdd::new(io_uring::types::Fd(fd), libc::POLLIN as _)
            .build()
            .user_data(POLLER_USERDATA);

        let mut sq = inner.uring.submission();
        let _ = unsafe { sq.push(&entry) };
        inner.poller_installed = true;
    }

    fn install_timeout(&self, inner: &mut UringInner, duration: Duration) {
        inner.timespec = timespec(duration);
        let entry = opcode::Timeout::new(&inner.timespec as *const Timespec)
            .build()
            .user_data(TIMEOUT_USERDATA);

        let mut sq = inner.uring.submission();
        let _ = unsafe { sq.push(&entry) };
    }

    /// Adaptive non-blocking reactor park.
    ///
    /// Instead of the blocking `submit_and_wait(1)` path, this:
    /// 1. Submits pending SQEs non-blocking (`io_uring_enter(submit=N, min_complete=0)`)
    /// 2. Polls CQ ring from shared memory (no syscall)
    /// 3. Returns immediately if in-flight ops exist (the runtime task drain
    ///    loop IS the spin — no explicit busy-wait needed)
    /// 4. Blocks only when truly idle (no in-flight ops) to save CPU
    fn inner_park_poll(&self, inner: &mut UringInner, timeout: Option<Duration>) -> io::Result<()> {
        // Process cross-thread wakers before anything else.
        #[cfg(feature = "sync")]
        {
            let mut has_wakers = false;
            while let Ok(w) = inner.waker_receiver.try_recv() {
                w.wake();
                has_wakers = true;
            }
            if has_wakers {
                // Wakers found — don't block. Submit only if SQEs pending.
                if !inner.uring.submission().is_empty() {
                    inner.submit()?;
                }
                inner.tick()?;
                return Ok(());
            }
        }

        // Step 1: Submit any pending SQEs non-blocking.
        // Only issue the io_uring_enter syscall when there are actually
        // SQEs to submit. Without this check, we'd call io_uring_enter(0,0)
        // on every loop iteration even when idle — this was the root cause
        // of the 33% sys CPU observed in AD.6.
        if !inner.uring.submission().is_empty() {
            inner.submit()?;
        }

        // Step 2: Poll CQ ring from shared memory (no syscall). `reaped` is the
        // number of completions processed this iteration — the progress signal.
        let reaped = inner.tick()?;

        // Step 3: Progress-based idle detection.
        //
        // Spin only while we are making progress. If we reaped completions this
        // iteration there is active work: return immediately so the runtime
        // drains the newly-woken tasks (which generate more SQEs), and we come
        // back to reap again. This is the syscall-free hot path under load.
        //
        // Crucially, this gates on `reaped`, NOT on `slab.len()`. Under a live
        // cluster the slab is essentially never empty — long-lived ops (gossip
        // timers, internode socket reads awaiting data) keep in-flight ops
        // present indefinitely. The old `slab.len() > 0` check therefore never
        // fell through to the blocking path and the worker busy-spun at 100%
        // CPU at idle (E4 Stage 1: idle burn + −34% under load). A poll that
        // reaps nothing means no work is ready right now even if ops are in
        // flight, so we fall through and block.
        if reaped > 0 {
            #[cfg(feature = "sync")]
            inner
                .shared_waker
                .awake
                .store(true, std::sync::atomic::Ordering::Release);
            return Ok(());
        }

        // Step 3b: Bounded adaptive spin before blocking.
        //
        // A single empty poll does NOT mean idle — under load the CQ ring
        // momentarily drains between completion batches, and the next batch
        // lands microseconds later. Blocking on every such gap issues a
        // `submit_and_wait(1)` (a voluntary context switch) per gap, which
        // collapses the ctxsw reduction the reactor exists to deliver (E4
        // Stage 1 re-run: block-on-first-empty gave only ~1.5× vs the ≥5× the
        // mechanism is capable of). So spin-poll the CQ for a bounded budget
        // first: if a completion arrives within `POLL_REACTOR_SPIN_US`, keep
        // spinning (syscall-free); only if the budget expires with the ring
        // still empty do we conclude we are truly idle and block. This is the
        // classic adaptive busy-poll: under load the budget is almost always
        // satisfied (≈one op-interval), so we rarely block; at idle the budget
        // expires quickly and cheaply (tens of µs per park, negligible CPU).
        //
        // Budget source: the builder's `poll_spin_us` (env-tunable via the
        // Adamas `ADAMAS_POLL_SPIN_US` plumbing) so the value can be swept
        // without recompiling; falls back to `DEFAULT_POLL_REACTOR_SPIN_US`.
        {
            // Client-gating: when enabled, skip the speculative idle spin
            // unless a client request is in flight on this shard. This lets the
            // reactor park through client-idle gaps instead of busy-polling
            // gossip/internode completions (the E4-falsified idle-CPU tax). The
            // `reaped > 0` early-return above is NOT gated — productive client
            // completions still return immediately; only this speculative spin
            // is suppressed.
            let gated_out =
                self.poll_reactor_client_gated && !crate::client_gate::client_in_flight();
            let budget_us = self.poll_spin_us.unwrap_or(DEFAULT_POLL_REACTOR_SPIN_US);
            if budget_us > 0 && !gated_out {
                let budget = Duration::from_micros(budget_us);
                let start = std::time::Instant::now();
                loop {
                    // Submit any SQEs generated since the last poll, non-blocking.
                    if !inner.uring.submission().is_empty() {
                        inner.submit()?;
                    }
                    let r = inner.tick()?;
                    if r > 0 {
                        // A completion landed within the budget — active work.
                        #[cfg(feature = "sync")]
                        inner
                            .shared_waker
                            .awake
                            .store(true, std::sync::atomic::Ordering::Release);
                        return Ok(());
                    }
                    if start.elapsed() >= budget {
                        break; // Budget exhausted — truly idle, fall through to block.
                    }
                    std::hint::spin_loop();
                }
            }
        }

        // Step 4: Budget expired with no completions — truly idle. Block until
        // the next completion. submit_and_wait(1) returns the instant any
        // in-flight op completes and sleeps at true idle (0% CPU). The eventfd
        // installed below ensures cross-thread wakers also unblock us.
        #[cfg(feature = "sync")]
        {
            inner
                .shared_waker
                .awake
                .store(false, std::sync::atomic::Ordering::Release);

            // Double-check wakers after setting awake=false (barrier pattern).
            while let Ok(w) = inner.waker_receiver.try_recv() {
                w.wake();
                inner
                    .shared_waker
                    .awake
                    .store(true, std::sync::atomic::Ordering::Release);
                // Waker arrived — don't block.
                inner.submit()?;
                inner.tick()?;
                return Ok(());
            }
        }

        // Allocate space for eventfd/poller/timeout SQEs.
        let mut space = 0;
        #[cfg(feature = "sync")]
        if !inner.eventfd_installed {
            space += 1;
        }
        #[cfg(feature = "poll-io")]
        if !inner.poller_installed {
            space += 1;
        }
        if timeout.is_some() {
            space += 1;
        }
        if space != 0 {
            Self::flush_space(inner, space)?;
        }

        #[cfg(feature = "poll-io")]
        if !inner.poller_installed {
            self.install_poller(inner, inner.poll.as_raw_fd());
        }

        #[cfg(feature = "sync")]
        if !inner.eventfd_installed {
            self.install_eventfd(inner, inner.shared_waker.as_raw_fd());
        }

        // Block with or without timeout. The `ParkScope` guard accumulates the
        // wall-clock time we spend blocked here into this shard's per-shard
        // `REACTOR_PARKED_NANOS` (the reactor-utilization diagnostic). It is the
        // ACTUAL blocking kernel park on the poll-reactor path, so it must be
        // measured whether poll_reactor is on (here) or off (inner_park below).
        {
            let _park = crate::reactor_park::ParkScope::new();
            if let Some(duration) = timeout {
                match inner.ext_arg {
                    false => {
                        self.install_timeout(inner, duration);
                        inner.uring.submit_and_wait(1)?;
                    }
                    true => {
                        let timespec = timespec(duration);
                        let args = io_uring::types::SubmitArgs::new().timespec(&timespec);
                        if let Err(e) = inner.uring.submitter().submit_with_args(1, &args) {
                            if e.raw_os_error() != Some(libc::ETIME) {
                                return Err(e);
                            }
                        }
                    }
                }
            } else {
                inner.uring.submit_and_wait(1)?;
            }
        }

        #[cfg(feature = "sync")]
        inner
            .shared_waker
            .awake
            .store(true, std::sync::atomic::Ordering::Release);

        inner.tick()?;
        Ok(())
    }

    fn inner_park(&self, timeout: Option<Duration>) -> io::Result<()> {
        let inner = unsafe { &mut *self.inner.get() };

        // Poll reactor mode: adaptive non-blocking reactor.
        // Never block when I/O ops are in flight; only block when truly idle.
        if self.poll_reactor {
            return self.inner_park_poll(inner, timeout);
        }

        #[allow(unused_mut)]
        let mut need_wait = true;

        #[cfg(feature = "sync")]
        {
            // Process foreign wakers
            while let Ok(w) = inner.waker_receiver.try_recv() {
                w.wake();
                need_wait = false;
            }

            // Set status as not awake if we are going to sleep
            if need_wait {
                inner
                    .shared_waker
                    .awake
                    .store(false, std::sync::atomic::Ordering::Release);
            }

            // Process foreign wakers left
            while let Ok(w) = inner.waker_receiver.try_recv() {
                w.wake();
                need_wait = false;
            }
        }

        if need_wait {
            // Poll-spin mode: before blocking on io_uring_enter, submit any
            // pending SQEs and then spin-poll the CQ ring from shared memory.
            // The CQ ring is mmap'd — reading it is a memory load (no syscall).
            // This eliminates the ~0.6ms per-yield overhead from the blocking
            // io_uring_enter(min_complete=1) path.
            if let Some(spin_us) = self.poll_spin_us {
                // Submit pending SQEs without blocking (min_complete=0).
                inner.submit()?;

                // Spin-poll the CQ ring. We process completions inline
                // (same logic as tick()) to detect when work is ready.
                let budget = Duration::from_micros(spin_us);
                let start = std::time::Instant::now();
                loop {
                    // Peek the CQ ring — pure shared-memory read, no syscall.
                    let cq = inner.uring.completion();
                    let mut found = false;
                    for cqe in cq {
                        found = true;
                        let index = cqe.user_data();
                        match index {
                            #[cfg(feature = "sync")]
                            EVENTFD_USERDATA => inner.eventfd_installed = false,
                            #[cfg(feature = "poll-io")]
                            POLLER_USERDATA => {
                                inner.poller_installed = false;
                                inner.poll.tick(Some(Duration::ZERO))?;
                            }
                            _ if index >= MIN_REVERSED_USERDATA => (),
                            _ => unsafe {
                                inner.ops.complete(index as _, resultify(&cqe), cqe.flags())
                            },
                        }
                    }
                    if found {
                        // Completions processed — return immediately so the
                        // executor can run newly-woken tasks.
                        #[cfg(feature = "sync")]
                        inner
                            .shared_waker
                            .awake
                            .store(true, std::sync::atomic::Ordering::Release);
                        return Ok(());
                    }
                    if start.elapsed() >= budget {
                        break; // Spin budget exhausted, fall through to blocking.
                    }
                    std::hint::spin_loop();
                }
            }

            // Install timeout and eventfd for unpark if sync is enabled

            // 1. alloc spaces
            let mut space = 0;
            #[cfg(feature = "sync")]
            if !inner.eventfd_installed {
                space += 1;
            }
            #[cfg(feature = "poll-io")]
            if !inner.poller_installed {
                space += 1;
            }
            if timeout.is_some() {
                space += 1;
            }
            if space != 0 {
                Self::flush_space(inner, space)?;
            }

            // 2.1 install poller
            #[cfg(feature = "poll-io")]
            if !inner.poller_installed {
                self.install_poller(inner, inner.poll.as_raw_fd());
            }

            // 2.2 install eventfd and timeout
            #[cfg(feature = "sync")]
            if !inner.eventfd_installed {
                self.install_eventfd(inner, inner.shared_waker.as_raw_fd());
            }

            // 2.3 install timeout and submit_and_wait with timeout. The
            // `ParkScope` guard accumulates the wall-clock time blocked here
            // into this shard's `REACTOR_PARKED_NANOS` — this is the blocking
            // kernel park on the non-poll-reactor path, the twin of the one in
            // `inner_park_poll`, so the diagnostic is correct in both modes.
            let _park = crate::reactor_park::ParkScope::new();
            if let Some(duration) = timeout {
                match inner.ext_arg {
                    // Submit and Wait with timeout in an TimeoutOp way.
                    // Better compatibility(5.4+).
                    false => {
                        self.install_timeout(inner, duration);
                        inner.uring.submit_and_wait(1)?;
                    }
                    // Submit and Wait with enter args.
                    // Better performance(5.11+).
                    true => {
                        let timespec = timespec(duration);
                        let args = io_uring::types::SubmitArgs::new().timespec(&timespec);
                        if let Err(e) = inner.uring.submitter().submit_with_args(1, &args) {
                            if e.raw_os_error() != Some(libc::ETIME) {
                                return Err(e);
                            }
                        }
                    }
                }
            } else {
                // Submit and Wait without timeout
                inner.uring.submit_and_wait(1)?;
            }
        } else {
            // Submit only
            inner.uring.submit()?;
        }

        // Set status as awake
        #[cfg(feature = "sync")]
        inner
            .shared_waker
            .awake
            .store(true, std::sync::atomic::Ordering::Release);

        // Process CQ
        inner.tick()?;

        Ok(())
    }

    #[cfg(feature = "poll-io")]
    #[inline]
    pub(crate) fn register_poll_io(
        this: &Rc<UnsafeCell<UringInner>>,
        source: &mut impl mio::event::Source,
        interest: mio::Interest,
    ) -> io::Result<usize> {
        let inner = unsafe { &mut *this.get() };
        inner.poll.register(source, interest)
    }

    #[cfg(feature = "poll-io")]
    #[inline]
    pub(crate) fn deregister_poll_io(
        this: &Rc<UnsafeCell<UringInner>>,
        source: &mut impl mio::event::Source,
        token: usize,
    ) -> io::Result<()> {
        let inner = unsafe { &mut *this.get() };
        inner.poll.deregister(source, token)
    }
}

impl Driver for IoUringDriver {
    /// Enter the driver context. This enables using uring types.
    fn with<R>(&self, f: impl FnOnce() -> R) -> R {
        // TODO(ihciah): remove clone
        let inner = Inner::Uring(self.inner.clone());
        CURRENT.set(&inner, f)
    }

    fn submit(&self) -> io::Result<()> {
        let inner = unsafe { &mut *self.inner.get() };
        inner.submit()?;
        inner.tick()?;
        Ok(())
    }

    fn park(&self) -> io::Result<()> {
        self.inner_park(None)
    }

    fn park_timeout(&self, duration: Duration) -> io::Result<()> {
        self.inner_park(Some(duration))
    }

    #[cfg(feature = "sync")]
    type Unpark = waker::UnparkHandle;

    #[cfg(feature = "sync")]
    fn unpark(&self) -> Self::Unpark {
        UringInner::unpark(&self.inner)
    }
}

impl UringInner {
    /// Reap the completion queue. Returns the number of CQEs processed this
    /// call. The count drives the poll-reactor's adaptive idle detection
    /// (`inner_park_poll`): spin while we are reaping completions, block when a
    /// poll comes up empty. Callers that don't need the count can ignore it
    /// (`inner.tick()?;` still type-checks — the usize is discarded by `?`).
    fn tick(&mut self) -> io::Result<usize> {
        let cq = self.uring.completion();

        let mut reaped: usize = 0;
        for cqe in cq {
            reaped += 1;
            let index = cqe.user_data();
            match index {
                #[cfg(feature = "sync")]
                EVENTFD_USERDATA => self.eventfd_installed = false,
                #[cfg(feature = "poll-io")]
                POLLER_USERDATA => {
                    self.poller_installed = false;
                    self.poll.tick(Some(Duration::ZERO))?;
                }
                _ if index >= MIN_REVERSED_USERDATA => (),
                // # Safety
                // Here we can make sure the result is valid.
                _ => unsafe { self.ops.complete(index as _, resultify(&cqe), cqe.flags()) },
            }
        }
        Ok(reaped)
    }

    fn submit(&mut self) -> io::Result<()> {
        loop {
            match self.uring.submit() {
                #[cfg(feature = "unstable")]
                Err(ref e)
                    if matches!(e.kind(), io::ErrorKind::Other | io::ErrorKind::ResourceBusy) =>
                {
                    self.tick()?;
                }
                #[cfg(not(feature = "unstable"))]
                Err(ref e)
                    if matches!(e.raw_os_error(), Some(libc::EAGAIN) | Some(libc::EBUSY)) =>
                {
                    // This error is constructed with io::Error::last_os_error():
                    // https://github.com/tokio-rs/io-uring/blob/01c83bbce965d4aaf93ebfaa08c3aa8b7b0f5335/src/sys/mod.rs#L32
                    // So we can use https://doc.rust-lang.org/nightly/std/io/struct.Error.html#method.raw_os_error
                    // to get the raw error code.
                    self.tick()?;
                }
                e => return e.map(|_| ()),
            }
        }
    }

    fn new_op<T: OpAble>(data: T, inner: &mut UringInner, driver: Inner) -> Op<T> {
        Op {
            driver,
            index: inner.ops.insert(T::RET_IS_FD),
            data: Some(data),
        }
    }

    pub(crate) fn submit_with_data<T>(
        this: &Rc<UnsafeCell<UringInner>>,
        data: T,
    ) -> io::Result<Op<T>>
    where
        T: OpAble,
    {
        let inner = unsafe { &mut *this.get() };
        // If the submission queue is full, flush it to the kernel
        if inner.uring.submission().is_full() {
            inner.submit()?;
        }

        // Create the operation
        let mut op = Self::new_op(data, inner, Inner::Uring(this.clone()));

        // Configure the SQE
        let data_mut = unsafe { op.data.as_mut().unwrap_unchecked() };
        let sqe = OpAble::uring_op(data_mut).user_data(op.index as _);

        {
            let mut sq = inner.uring.submission();

            // Push the new operation
            if unsafe { sq.push(&sqe).is_err() } {
                unimplemented!("when is this hit?");
            }
        }

        // Submit the new operation. At this point, the operation has been
        // pushed onto the queue and the tail pointer has been updated, so
        // the submission entry is visible to the kernel. If there is an
        // error here (probably EAGAIN), we still return the operation. A
        // future `io_uring_enter` will fully submit the event.

        // CHIHAI: We are not going to do syscall now. If we are waiting
        // for IO, we will submit on `park`.
        // let _ = inner.submit();
        Ok(op)
    }

    pub(crate) fn poll_op(
        this: &Rc<UnsafeCell<UringInner>>,
        index: usize,
        cx: &mut Context<'_>,
    ) -> Poll<CompletionMeta> {
        let inner = unsafe { &mut *this.get() };
        let lifecycle = unsafe { inner.ops.slab.get(index).unwrap_unchecked() };
        lifecycle.poll_op(cx)
    }

    #[cfg(feature = "poll-io")]
    pub(crate) fn poll_legacy_op<T: OpAble>(
        this: &Rc<UnsafeCell<Self>>,
        data: &mut T,
        cx: &mut Context<'_>,
    ) -> Poll<CompletionMeta> {
        let inner = unsafe { &mut *this.get() };
        let (direction, index) = match data.legacy_interest() {
            Some(x) => x,
            None => {
                // if there is no index provided, it means the action does not rely on fd
                // readiness. do syscall right now.
                return Poll::Ready(CompletionMeta {
                    result: OpAble::legacy_call(data),
                    flags: 0,
                });
            }
        };

        // wait io ready and do syscall
        inner
            .poll
            .poll_syscall(cx, index, direction, || OpAble::legacy_call(data))
    }

    pub(crate) fn drop_op<T: 'static>(
        this: &Rc<UnsafeCell<UringInner>>,
        index: usize,
        data: &mut Option<T>,
        _skip_cancel: bool,
    ) {
        let inner = unsafe { &mut *this.get() };
        if index == usize::MAX {
            // already finished
            return;
        }
        if let Some(lifecycle) = inner.ops.slab.get(index) {
            let _must_finished = lifecycle.drop_op(data);
            #[cfg(feature = "async-cancel")]
            if !_must_finished && !_skip_cancel {
                unsafe {
                    let cancel = opcode::AsyncCancel::new(index as u64)
                        .build()
                        .user_data(u64::MAX);

                    // Try push cancel, if failed, will submit and re-push.
                    if inner.uring.submission().push(&cancel).is_err() {
                        let _ = inner.submit();
                        let _ = inner.uring.submission().push(&cancel);
                    }
                }
            }
        }
    }

    pub(crate) unsafe fn cancel_op(this: &Rc<UnsafeCell<UringInner>>, index: usize) {
        let inner = &mut *this.get();
        let cancel = opcode::AsyncCancel::new(index as u64)
            .build()
            .user_data(u64::MAX);
        if inner.uring.submission().push(&cancel).is_err() {
            let _ = inner.submit();
            let _ = inner.uring.submission().push(&cancel);
        }
    }

    #[cfg(feature = "sync")]
    pub(crate) fn unpark(this: &Rc<UnsafeCell<UringInner>>) -> waker::UnparkHandle {
        let inner = unsafe { &*this.get() };
        let weak = std::sync::Arc::downgrade(&inner.shared_waker);
        waker::UnparkHandle(weak)
    }
}

impl AsRawFd for IoUringDriver {
    fn as_raw_fd(&self) -> RawFd {
        unsafe { (*self.inner.get()).uring.as_raw_fd() }
    }
}

impl Drop for IoUringDriver {
    fn drop(&mut self) {
        trace!("MONOIO DEBUG[IoUringDriver]: drop");

        // Deregister thread id
        #[cfg(feature = "sync")]
        {
            use crate::driver::thread::{unregister_unpark_handle, unregister_waker_sender};
            unregister_unpark_handle(self.thread_id);
            unregister_waker_sender(self.thread_id);
        }
    }
}

impl Drop for UringInner {
    fn drop(&mut self) {
        // no need to wait for completion, as the kernel will clean up the ring asynchronically.
        let _ = self.uring.submitter().submit();
        unsafe {
            ManuallyDrop::drop(&mut self.uring);
        }
    }
}

impl Ops {
    const fn new() -> Self {
        Ops { slab: Slab::new() }
    }

    // Insert a new operation
    #[inline]
    pub(crate) fn insert(&mut self, is_fd: bool) -> usize {
        self.slab.insert(MaybeFdLifecycle::new(is_fd))
    }

    // Complete an operation
    // # Safety
    // Caller must make sure the result is valid.
    #[inline]
    unsafe fn complete(&mut self, index: usize, result: io::Result<u32>, flags: u32) {
        let lifecycle = unsafe { self.slab.get(index).unwrap_unchecked() };
        lifecycle.complete(result, flags);
    }
}

#[inline]
fn resultify(cqe: &cqueue::Entry) -> io::Result<u32> {
    let res = cqe.result();

    if res >= 0 {
        Ok(res as u32)
    } else {
        Err(io::Error::from_raw_os_error(-res))
    }
}

#[cfg(test)]
mod modern_flags_tests {
    use super::*;

    /// Serialize the env-var mutation across tests in this module: they all poke
    /// the same process-global `ADAMAS_IOURING_MODERN_FLAGS`.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn with_modern_flags<R>(value: Option<&str>, f: impl FnOnce() -> R) -> R {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("ADAMAS_IOURING_MODERN_FLAGS").ok();
        match value {
            Some(v) => std::env::set_var("ADAMAS_IOURING_MODERN_FLAGS", v),
            None => std::env::remove_var("ADAMAS_IOURING_MODERN_FLAGS"),
        }
        let out = f();
        match prev {
            Some(v) => std::env::set_var("ADAMAS_IOURING_MODERN_FLAGS", v),
            None => std::env::remove_var("ADAMAS_IOURING_MODERN_FLAGS"),
        }
        out
    }

    #[test]
    fn modern_flags_requested_parses_truthy_and_falsy() {
        for v in ["1", "true", "TRUE", "Yes", "on", " on "] {
            with_modern_flags(Some(v), || {
                assert!(modern_flags_requested(), "{v:?} should be truthy");
            });
        }
        for v in ["0", "false", "no", "off", "", "garbage"] {
            with_modern_flags(Some(v), || {
                assert!(!modern_flags_requested(), "{v:?} should be falsy");
            });
        }
        with_modern_flags(None, || {
            assert!(!modern_flags_requested(), "unset should default off");
        });
    }

    /// Vanilla path (flags off): must always build a usable ring.
    #[test]
    fn build_uring_vanilla_succeeds() {
        with_modern_flags(Some("0"), || {
            let urb = IoUring::builder();
            build_uring(&urb, 256).expect("vanilla io_uring build must succeed");
        });
    }

    /// Modern path (flags on): must build successfully on a supporting kernel
    /// (5.19+) OR transparently fall back to vanilla on an older kernel — in
    /// neither case may it error out. This exercises the kernel-fallback path on
    /// CI machines that may not support the flags.
    #[test]
    fn build_uring_modern_builds_or_falls_back() {
        with_modern_flags(Some("1"), || {
            let urb = IoUring::builder();
            // Whether or not SINGLE_ISSUER+COOP_TASKRUN are supported, build_uring
            // must yield a ring: supported -> flagged ring, unsupported -> vanilla.
            build_uring(&urb, 256)
                .expect("build_uring with modern flags must succeed or fall back, never error");
        });
    }
}
