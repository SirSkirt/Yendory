//! Yendory Engine: the shared Bicameral neural implementation used by Yendory Studio,
//! Neural Nova and third-party frontends.
//!
//! - [`model`]: configuration, parameter layout, forward/backward, AdamW with training focus,
//!   ablations, KV-cached generation and `.nnova2` checkpoints.
//! - [`tokenizer`]: ASCII characters (ids 0..127) plus learned word units (ids 128..).
//! - [`nexus`]: how a model takes actions in an environment and gets results back, with
//!   permissions and limits (the environment announces itself; the model must be taught).
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
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod learn;
pub mod memory;
pub mod model;
pub mod nexus;
pub mod tokenizer;
pub use model::size_presets;

pub use model::{
    compare_outputs, Ablation, AblationEffects, Bicameral, CacheStats, Config, Dropout, Focus, Growth,
    GrowthOption, KvCache, Measurements, NotOffered, OutputComparison, Preservation, SizePreset,
    ThinkOptions, ThinkingTree, Trace,
};
pub use tokenizer::{TokenId, Tokenizer};
