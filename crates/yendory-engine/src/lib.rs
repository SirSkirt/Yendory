//! Yendory Engine: the shared Bicameral neural implementation used by Yendory Studio,
//! Neural Nova and third-party frontends.
//!
//! - [`model`]: configuration, parameter layout, forward/backward, AdamW with training focus,
//!   ablations, KV-cached generation and `.nnova2` checkpoints.
//! - [`tokenizer`]: ASCII characters (ids 0..127) plus learned word units (ids 128..).
//!
//! The numpy reference in `reference/` is the specification; `tests/compat` holds the
//! cross-implementation checks. Every output comes from the network; there are no canned replies.
// Style lints on the numeric code carried over from Neural Nova. Index loops mirror the
// reference's maths line by line; rewriting them buys nothing and risks the verified behaviour.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::new_without_default,
    clippy::match_overlapping_arm,
    clippy::field_reassign_with_default,
    clippy::excessive_precision
)]
pub mod fingerprint;
pub mod learn;
pub mod model;
pub mod tokenizer;

pub use model::{
    Ablation, AblationEffects, Bicameral, CacheStats, Config, Focus, KvCache, Measurements, ThinkOptions,
    ThinkingTree, Trace,
};
pub use tokenizer::Tokenizer;
