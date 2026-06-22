//! Per-shard reactor "parked" time accounting.
//!
//! A thread-local nanosecond counter that accumulates the wall-clock time the
//! io_uring reactor spends *blocked* inside a `submit_and_wait(1)` /
//! `submit_with_args(1, ..)` enter (i.e. the kernel park where the worker is
//! asleep waiting for the next completion). The host application (Adamas) reads
//! it via [`reactor_parked_nanos`] from its /metrics endpoint to derive a direct
//! reactor utilization gauge:
//!
//! ```text
//! parked_fraction = Δ reactor_parked_nanos / Δ wall_clock_nanos   (per snapshot)
//! utilization     = 1 - parked_fraction
//! ```
//!
//! This is THE instrument that distinguishes the two write-throughput-gap
//! hypotheses: high parked_fraction => the reactor is blocked/idle between
//! batches (the park-tax theory — cores stalling on `io_uring_enter`); low
//! parked_fraction with the gap still present => the cost is heavier per-request
//! work, not parking.
//!
//! Why a thread-local `Cell<u64>` and NOT an atomic (mirrors `client_gate`):
//! the reactor is strictly single-threaded per shard (thread-per-core), so the
//! only writer and the only in-process reader live on the same thread. The
//! accumulate happens on the reactor thread inside the park; Adamas's snapshot
//! reader also runs on that shard's thread (it polls the per-shard metrics on
//! the owning shard). No cross-thread sharing => no atomic needed here. (If a
//! consumer needs the value cross-thread it copies it into a `Relaxed` atomic
//! written only by the owning shard — that lives in Adamas's `ShardMetrics`,
//! not here.)
//!
//! Cost: one `Instant::now()` pair per *park* (not per task, not per
//! completion). Parks happen at most once per reactor turn and only when the
//! reactor actually decides to block, so this is a negligible passive counter,
//! safe to leave default-ON as a diagnostic with zero behavior change.

use std::{cell::Cell, time::Instant};

thread_local! {
    /// Monotonic accumulator of nanoseconds this shard's reactor has spent
    /// blocked in a kernel park (`io_uring_enter` with `min_complete>=1`).
    static REACTOR_PARKED_NANOS: Cell<u64> = const { Cell::new(0) };
}

/// RAII guard: measures the wall-clock time from construction to drop and adds
/// it to this shard's [`REACTOR_PARKED_NANOS`] accumulator.
///
/// Construct it immediately before the blocking `submit_and_wait` / enter call
/// and let it drop right after. Using a guard (rather than an explicit
/// start/elapsed pair at each site) means the elapsed time is accumulated even
/// if the blocking call returns `Err` and the caller `?`-propagates — the Drop
/// still runs, so the counter never under-counts on the error path.
///
/// Saturating add so a pathological clock can never wrap the counter.
#[must_use = "ParkScope measures from construction to drop; bind it to a name"]
pub(crate) struct ParkScope {
    start: Instant,
}

impl ParkScope {
    #[inline]
    pub(crate) fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }
}

impl Drop for ParkScope {
    #[inline]
    fn drop(&mut self) {
        let elapsed = self.start.elapsed().as_nanos() as u64;
        REACTOR_PARKED_NANOS.with(|c| c.set(c.get().saturating_add(elapsed)));
    }
}

/// Total nanoseconds this shard's reactor has spent blocked in a kernel park
/// since process start (monotonic, per-shard thread-local).
///
/// Read by the host application's /metrics snapshot on the owning shard. A bench
/// polling at 1 Hz computes the parked fraction from the delta between two
/// snapshots divided by the wall-clock delta over the same interval.
#[inline]
pub fn reactor_parked_nanos() -> u64 {
    REACTOR_PARKED_NANOS.with(|c| c.get())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{thread, time::Duration};

    /// Reset is not exposed (the production counter is monotonic), so each test
    /// runs on its own thread to get a fresh thread-local.
    #[test]
    fn parked_nanos_starts_at_zero() {
        thread::spawn(|| {
            assert_eq!(reactor_parked_nanos(), 0);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn park_scope_accumulates_elapsed() {
        thread::spawn(|| {
            assert_eq!(reactor_parked_nanos(), 0);
            let before = reactor_parked_nanos();
            {
                let _scope = ParkScope::new();
                thread::sleep(Duration::from_millis(5));
            } // drop here accumulates
            let after = reactor_parked_nanos();
            assert!(
                after > before,
                "parked_nanos must increase after a park scope (before={before}, after={after})"
            );
            // 5ms == 5_000_000ns; allow generous slack for scheduling jitter but
            // assert we recorded at least a couple ms (not a no-op).
            assert!(
                after - before >= 2_000_000,
                "expected >= ~2ms recorded for a 5ms sleep, got {}ns",
                after - before
            );
        })
        .join()
        .unwrap();
    }

    #[test]
    fn park_scope_accumulates_monotonically_across_parks() {
        thread::spawn(|| {
            {
                let _s = ParkScope::new();
                thread::sleep(Duration::from_millis(2));
            }
            let one = reactor_parked_nanos();
            {
                let _s = ParkScope::new();
                thread::sleep(Duration::from_millis(2));
            }
            let two = reactor_parked_nanos();
            assert!(
                two > one,
                "counter is monotonic across successive parks (one={one}, two={two})"
            );
        })
        .join()
        .unwrap();
    }
}
