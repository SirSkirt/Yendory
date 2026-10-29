//! Recovery snapshots for long runs (Reading and Conversation).
//!
//! A run hands out its *working* model (weights, AdamW moments, step) with the state needed to
//! carry on from the same point: the batch it reached, the random generator's state, the judge's
//! best loss and misses, the curriculum's stage. A run given that snapshot back continues exactly
//! where the first one was (on the CPU the weights after the resumed run are bit for bit those of
//! an uninterrupted run; on the GPU the arithmetic itself is not bit-exact, so the runs agree to
//! floating-point rounding). Snapshots are taken on an interval (`RecoveryEvery`), never while a
//! check is still being measured, and when the run is stopped.
//!
//! The snapshot is the caller's to store; the accepted model is saved separately (`on_accepted`)
//! and a snapshot never replaces it.
use crate::model::Bicameral;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// How often a run hands out a recovery snapshot: after this many seconds or steps since the
/// last one (0: not on that measure). Both 0: only when the run is stopped.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RecoveryEvery {
    pub seconds: u64,
    pub steps: u64,
}

/// A snapshot handed to the caller.
pub struct Snapshot<'a, S> {
    /// The working model (not the accepted one): weights, optimizer state and step.
    pub model: &'a Bicameral,
    /// A second model the run keeps (Conversation's balanced checkpoint), if any.
    pub extra: Option<&'a Bicameral>,
    pub state: &'a S,
    /// Taken because the run was stopped (not on the interval).
    pub stopped: bool,
}

/// A snapshot given back to continue a run.
pub struct Resume<S> {
    pub state: S,
    /// The working model of the snapshot.
    pub model: Bicameral,
    pub extra: Option<Bicameral>,
}

type OnSnapshot<'a, S> = &'a mut dyn FnMut(Snapshot<'_, S>) -> Result<(), String>;

/// Recovery for one run: when to take snapshots, where they go, and the snapshot to resume from.
pub struct Recovery<'a, S> {
    pub every: RecoveryEvery,
    pub on_snapshot: Option<OnSnapshot<'a, S>>,
    pub resume: Option<Resume<S>>,
}

impl<S> Recovery<'_, S> {
    /// No snapshots and nothing to resume.
    pub fn none() -> Self {
        Recovery { every: RecoveryEvery::default(), on_snapshot: None, resume: None }
    }

    pub(crate) fn take(&mut self, snapshot: Snapshot<'_, S>) -> Result<(), String> {
        match self.on_snapshot.as_mut() {
            Some(f) => f(snapshot),
            None => Ok(()),
        }
    }

    pub(crate) fn wanted(&self) -> bool {
        self.on_snapshot.is_some()
    }
}

/// Decides when the next interval snapshot is due.
pub(crate) struct Timer {
    every: RecoveryEvery,
    last: Instant,
    last_step: u64,
}

impl Timer {
    pub(crate) fn new(every: RecoveryEvery, step: u64) -> Timer {
        Timer { every, last: Instant::now(), last_step: step }
    }
    pub(crate) fn due(&self, step: u64) -> bool {
        (self.every.steps > 0 && step >= self.last_step + self.every.steps)
            || (self.every.seconds > 0 && self.last.elapsed() >= Duration::from_secs(self.every.seconds))
    }
    pub(crate) fn reset(&mut self, step: u64) {
        self.last = Instant::now();
        self.last_step = step;
    }
}

/// A 64-bit FNV-1a fingerprint of texts (with their boundaries), to check that a run resumes on
/// the same data.
#[derive(Clone, Copy)]
pub(crate) struct Fingerprint(u64);

impl Fingerprint {
    pub(crate) fn new() -> Self {
        Fingerprint(0xcbf2_9ce4_8422_2325)
    }
    pub(crate) fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.0 ^= x as u64;
            self.0 = self.0.wrapping_mul(0x100_0000_01b3);
        }
    }
    pub(crate) fn text(&mut self, t: &str) {
        self.bytes(t.as_bytes());
        // A boundary that no text contains, so ["ab", "c"] and ["a", "bc"] differ.
        self.bytes(&[0xff]);
    }
    pub(crate) fn value(self) -> u64 {
        self.0
    }
}

/// The snapshot's working model fits the run's model (same shape, same vocabulary).
pub(crate) fn same_shape(a: &Bicameral, b: &Bicameral) -> bool {
    a.config == b.config && a.weights.len() == b.weights.len()
}
