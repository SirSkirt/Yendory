//! Gaming Nova's network lives in the Engine (`yendory_engine::skill`), so a Nova Neuro model
//! carries its game skills in its own file (docs/FORMAT.md on yendory-hub). This module is that
//! network, under the name the rest of Tools uses.
pub use yendory_engine::skill::*;
