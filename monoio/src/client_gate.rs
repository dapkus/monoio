//! Per-shard client-request-in-flight gate.
//!
//! A thread-local counter the host application (Adamas) raises around the
//! window in which a *client* (CQL) request is being processed on this shard,
//! and the io_uring poll-reactor reads to decide whether its speculative idle
//! spin is worth running.
//!
//! Rationale: the selective poll-reactor's bounded adaptive spin keeps the
//! reactor busy-polling the CQ ring during client-idle gaps because gossip /
//! internode completions keep the spin loop satisfied (the E4-falsified
//! idle-CPU tax). Gating the spin on "is a client request actually in flight
//! on this shard?" lets the reactor park through those gaps while still
//! returning immediately for productive completions.
//!
//! The counter is a plain `Cell<u32>` — NOT an atomic — because it is strictly
//! single-threaded per shard: the writer is Adamas's connection loop and the
//! reader is the park, both running on the same thread-per-core reactor thread.
//! `begin`/`end` are exported on every platform (they are inert no-ops on the
//! reader side off-Linux, since only the io_uring spin loop consults the
//! counter); this keeps the Adamas call sites platform-agnostic.

use std::cell::Cell;

thread_local! {
    static CLIENT_IN_FLIGHT: Cell<u32> = const { Cell::new(0) };
}

/// Mark that a client request has entered processing on this shard.
///
/// Pairs with [`client_request_end`]. Increments a per-shard thread-local
/// counter; while the counter is non-zero the poll-reactor's speculative idle
/// spin is permitted (when client-gating is enabled).
#[inline]
pub fn client_request_begin() {
    CLIENT_IN_FLIGHT.with(|c| c.set(c.get() + 1));
}

/// Mark that a client request has finished processing on this shard.
///
/// Pairs with [`client_request_begin`]. Saturating-decrements the per-shard
/// counter so an unbalanced extra end can never wrap to a huge value.
#[inline]
pub fn client_request_end() {
    CLIENT_IN_FLIGHT.with(|c| c.set(c.get().saturating_sub(1)));
}

/// Whether at least one client request is currently in flight on this shard.
///
/// Read by the io_uring poll-reactor's client-gated spin decision.
#[cfg(all(target_os = "linux", feature = "iouring"))]
#[inline]
pub(crate) fn client_in_flight() -> bool {
    CLIENT_IN_FLIGHT.with(|c| c.get() > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read the raw per-shard counter (test-only; the production reader is the
    /// platform-gated `client_in_flight`).
    fn count() -> u32 {
        CLIENT_IN_FLIGHT.with(|c| c.get())
    }

    #[test]
    fn begin_end_balance_returns_to_zero() {
        // Fresh thread-local starts at zero.
        assert_eq!(count(), 0);

        client_request_begin();
        assert_eq!(count(), 1);

        client_request_begin();
        assert_eq!(count(), 2, "nested/concurrent requests stack");

        client_request_end();
        assert_eq!(count(), 1);

        client_request_end();
        assert_eq!(count(), 0, "balanced begin/end returns to idle");
    }

    #[test]
    fn end_saturates_at_zero() {
        // Each test runs on its own thread, so the counter is fresh here.
        assert_eq!(count(), 0);
        // An unbalanced extra end must NOT wrap to u32::MAX.
        client_request_end();
        assert_eq!(count(), 0, "saturating_sub floors at zero");
    }

    /// Mirrors the poll-reactor's `gated_out` predicate so the gating decision
    /// is unit-tested without a live io_uring ring:
    ///   gated_out = client_gated && !client_in_flight()
    /// When gated_out is true the speculative idle spin is skipped.
    fn gated_out(client_gated: bool) -> bool {
        client_gated && count() == 0
    }

    #[test]
    fn gating_decision_matches_spin_predicate() {
        assert_eq!(count(), 0);

        // Gate off: never gated out, spin always runs (legacy poll-reactor).
        assert!(!gated_out(false), "gate disabled => spin runs unconditionally");

        // Gate on + no client in flight: gated out, spin is skipped.
        assert!(gated_out(true), "gate on + idle => spin skipped, park instead");

        // Gate on + client in flight: not gated out, spin runs.
        client_request_begin();
        assert!(!gated_out(true), "gate on + client in flight => spin runs");
        client_request_end();
    }
}
