//! The open model's own core (Nova Bicameral or Nova Neuro) taking part in play: a game skill
//! linked to it (its "cortex", `yendory_engine::cortex`) reads the core's state every tick
//! through its goal input. The core is the model's main weights, the same ones that hold its
//! conversation: Tools reads them from the Network (`model.checkpoint`) and, when it trains the
//! core, saves the model back (`model.checkpoint.save`).
use super::policy::Policy;
use crate::Res;
use serde_json::{json, Value};
use tauri::AppHandle;
use yendory_engine::model::{Ablation, KvCache};
use yendory_engine::Bicameral;

/// The open model's whole checkpoint: its core, optimizer, words and skills.
pub fn load(app: &AppHandle, model_id: &str) -> Res<Bicameral> {
    let bytes = crate::network_download(app, "model.checkpoint", json!({"id": model_id}))?;
    Bicameral::from_bytes(&bytes)
}

/// Saves the model (core, optimizer and skills) through the Network.
pub fn save(app: &AppHandle, model_id: &str, model: &Bicameral) -> Res<Value> {
    let bytes = model.to_bytes()?;
    let upload = crate::network_upload(app, &bytes)?;
    drop(bytes);
    crate::network_call(app, "model.checkpoint.save", json!({"id": model_id, "upload": upload}))
}

/// The core playing with a linked skill, one tick at a time. It starts again every `stretch`
/// ticks (and after a pause), as training reads it: in stretches from their first tick.
pub struct Cortex {
    pub core: Bicameral,
    cache: KvCache,
    pub stretch: usize,
}

impl Cortex {
    pub fn new(core: Bicameral, stretch: usize) -> Res<Cortex> {
        if stretch == 0 || stretch > core.config.max_tokens {
            return Err(format!(
                "The core reads at most {} ticks at once; a stretch of {stretch} doesn't fit",
                core.config.max_tokens
            ));
        }
        let cache = KvCache::new(&core, Ablation::default());
        Ok(Cortex { core, cache, stretch })
    }

    /// Starts again (a new stretch).
    pub fn reset(&mut self) {
        self.cache.reset(&self.core);
    }

    /// The goal input for this tick, from what the skill saw (its sight).
    pub fn goal(&mut self, policy: &Policy, sight: Vec<f32>) -> Res<Vec<f32>> {
        if self.cache.positions() >= self.stretch {
            self.reset();
        }
        let (left, right) = policy.link_in(&[sight]);
        let state = self.cache.sense_append(&self.core, &left, &right)?;
        Ok(policy.link_out(&state))
    }
}
