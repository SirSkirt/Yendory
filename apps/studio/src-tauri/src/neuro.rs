//! Nova Neuro (experimental) in Studio: the brain builder's exact calculator and presets.
//! The Engine computes every number (`Bicameral::tensors_for`, `activation_floats_for`);
//! nothing here is estimated by hand.
use super::*;
use yendory_engine::model::neuro::{
    self as engine_neuro, Associative, Consolidation, Experts, NeuroSpec, Relational, WorkingMemory,
};

/// What the builder sends: the language core and every module's settings.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NeuroInput {
    /// Transformer width, blocks, heads, feed-forward width, lower blocks, rounds.
    pub width: usize,
    pub blocks: usize,
    pub heads: usize,
    pub ff: usize,
    pub lower: usize,
    pub rounds: usize,
    pub bus: usize,
    pub controller: usize,
    #[serde(default)]
    pub memory: Option<WorkingMemory>,
    #[serde(default)]
    pub associative: Option<Associative>,
    #[serde(default)]
    pub relational: Option<Relational>,
    #[serde(default)]
    pub experts: Option<Experts>,
    #[serde(default)]
    pub consolidation: Option<Consolidation>,
    /// The language core's transformer blocks are used (off: the networks alone read the word
    /// embeddings, and `blocks`, `heads`, `ff` and `lower` are ignored).
    #[serde(default = "yes")]
    pub transformer: bool,
}

impl NeuroInput {
    fn from_preset(p: &engine_neuro::NeuroPreset) -> Self {
        NeuroInput {
            width: p.config.d,
            blocks: p.config.blocks,
            heads: p.config.heads,
            ff: p.config.ff,
            lower: p.config.split,
            rounds: p.config.loops,
            bus: p.design.bus,
            controller: p.design.controller,
            memory: p.design.memory.clone(),
            associative: p.design.associative.clone(),
            relational: p.design.relational.clone(),
            experts: p.design.experts.clone(),
            consolidation: p.design.consolidation.clone(),
            transformer: p.config.blocks > 0,
        }
    }
    /// The Engine configuration and design (vocabulary and context as chosen elsewhere).
    pub(crate) fn build(&self, vocab: usize, context: usize) -> Res<(Config, NeuroSpec)> {
        // Without transformer blocks the head count only has to keep the configuration valid:
        // heads of 32 values where the width allows (what growing by one head would add).
        let heads = if self.transformer {
            self.heads
        } else if self.width.is_multiple_of(64) {
            self.width / 32
        } else if self.width.is_multiple_of(4) {
            2
        } else {
            1
        };
        let config = Config {
            vocab,
            d: self.width,
            heads,
            ff: self.ff,
            blocks: if self.transformer { self.blocks } else { 0 },
            split: if self.transformer { self.lower } else { 0 },
            loops: self.rounds,
            max_tokens: context,
            window_chars: 1,
            r1: 1,
            r2: 1,
            ..Config::default()
        };
        let per_head = config.d.checked_div(config.heads).unwrap_or(0);
        if per_head == 0 || per_head * config.heads != config.d || per_head & 1 == 1 {
            return Err(if self.transformer {
                "The language core's width must divide into the heads, with an even width per head".into()
            } else {
                "Choose an even main width".into()
            });
        }
        if !(1..=4).contains(&config.loops) {
            return Err("Choose 1 to 4 computation rounds".into());
        }
        if self.transformer && (config.blocks == 0 || config.split >= config.blocks) {
            return Err(
                "Keep at least one upper transformer block (lower blocks run once, upper blocks every round)"
                    .into(),
            );
        }
        Bicameral::check_config(&config)?;
        let design = NeuroSpec {
            version: engine_neuro::VERSION,
            bus: self.bus,
            memory: self.memory.clone(),
            associative: self.associative.clone(),
            controller: self.controller,
            relational: self.relational.clone(),
            experts: self.experts.clone(),
            consolidation: self.consolidation.clone(),
        };
        if !self.transformer && design.modules().is_empty() {
            return Err("Choose at least one network: the language core or one of the others".into());
        }
        design.validate(&config)?;
        Ok((config, design))
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Part {
    name: String,
    parameters: usize,
    share: f32,
}

/// A module the builder offers, and whether it can be used yet.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Offer {
    id: &'static str,
    name: &'static str,
    role: &'static str,
    /// "available", "next stage" or "future".
    status: &'static str,
    gpu: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NeuroPlan {
    parameters: usize,
    parts: Vec<Part>,
    file_bytes: u64,
    chat_bytes: u64,
    reading_bytes: u64,
    teaching_bytes: u64,
    /// Where it can train today.
    hardware: Vec<String>,
    /// Problems with the design (empty when it can be created).
    errors: Vec<String>,
    offers: Vec<Offer>,
}

fn offers() -> Vec<Offer> {
    let o = |id, name, role, status, gpu| Offer { id, name, role, status, gpu };
    vec![
        o(
            "transformer",
            "Language core (Transformer)",
            "language, context, combining information into replies",
            "available",
            true,
        ),
        o(
            "memory",
            "Working memory (minGRU)",
            "keeps and updates what matters in the current text",
            "available",
            true,
        ),
        o(
            "associative",
            "Associative memory (modern Hopfield)",
            "learned patterns recalled by similarity",
            "available",
            true,
        ),
        o(
            "controller",
            "Executive Controller",
            "learns how strongly to use each module, per token and round",
            "available",
            true,
        ),
        o(
            "bus",
            "Communication bus",
            "carries information between the networks, both ways",
            "available",
            true,
        ),
        o(
            "relational",
            "Relational network (entity slots)",
            "entities and how their relations change, from text already read",
            "available",
            true,
        ),
        o(
            "experts",
            "Mixture of Experts",
            "specialised transformations chosen by a learned router",
            "available",
            true,
        ),
        o(
            "consolidation",
            "Consolidation (autoencoder)",
            "compresses what is on the bus into a short code; associative memory is queried with it",
            "available",
            true,
        ),
        o(
            "gru",
            "Classic GRU / LSTM / State Space Model",
            "alternative working memories for comparison",
            "future",
            false,
        ),
        o("cnn", "CNN", "images, audio, spatial input", "future", false),
        o("liquid", "Liquid network", "adaptive temporal processing", "future", false),
        o("spiking", "Spiking network", "event-driven processing", "future", false),
        o("diffusion", "Diffusion network", "generative image or audio processing", "future", false),
    ]
}

/// The exact plan of a design: parameters per part, memory, hardware, problems.
pub(crate) fn plan(input: &NeuroInput, word_limit: Option<usize>, context: usize) -> NeuroPlan {
    let vocab = yendory_engine::tokenizer::BASE + word_limit.unwrap_or(256);
    let mut errors = Vec::new();
    let built = input.build(vocab, context);
    let (parameters, parts, row) = match &built {
        Ok((config, design)) => {
            let tensors = Bicameral::tensors_for(config, Some(design));
            let total: usize = tensors.iter().map(|(_, r, c)| r * c).sum();
            let mut by: Vec<(String, usize)> = Vec::new();
            for (name, r, c) in &tensors {
                let part = engine_neuro::part_of(name).to_string();
                match by.iter_mut().find(|(n, _)| *n == part) {
                    Some(e) => e.1 += r * c,
                    None => by.push((part, r * c)),
                }
            }
            let parts = by
                .into_iter()
                .map(|(name, p)| Part { name, parameters: p, share: p as f32 / total.max(1) as f32 })
                .collect();
            let row =
                4 * Bicameral::activation_floats_for(config, Some(design)) as u64 * config.max_tokens as u64;
            (total, parts, row)
        }
        Err(e) => {
            errors.push(e.clone());
            (0, Vec::new(), 0)
        }
    };
    if parameters > 16_000_000 {
        errors.push(format!("{} parameters: above the 16M experimental limit for Nova Neuro", parameters));
    }
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let (file_bytes, chat_bytes, reading_bytes, teaching_bytes) = memory(parameters, threads, row);
    NeuroPlan {
        parameters,
        parts,
        file_bytes,
        chat_bytes,
        reading_bytes,
        teaching_bytes,
        hardware: vec![
            "GPU (DirectX 12 or Vulkan): trains every module (the device setting chooses GPU, CPU or both)"
                .into(),
            "CPU: trains, and generates replies".into(),
        ],
        errors,
        offers: offers(),
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PresetView {
    id: &'static str,
    name: &'static str,
    input: NeuroInput,
    parameters: usize,
}

#[tauri::command]
pub fn neuro_presets() -> Vec<PresetView> {
    engine_neuro::presets()
        .iter()
        .map(|p| PresetView {
            id: p.id,
            name: p.name,
            input: NeuroInput::from_preset(p),
            parameters: Bicameral::with_design(p.config.clone(), Some(p.design.clone()), 1).parameter_count(),
        })
        .collect()
}

#[tauri::command]
pub fn neuro_plan(input: NeuroInput, word_limit: Option<usize>, context: Option<usize>) -> NeuroPlan {
    plan(&input, word_limit, context.unwrap_or(256))
}

/// A new, untrained Nova Neuro model from the builder's design.
#[tauri::command]
pub fn create_neuro_model(
    app: AppHandle,
    studio: State<Studio>,
    name: String,
    input: NeuroInput,
    word_limit: Option<usize>,
    context: Option<usize>,
) -> Res<Meta> {
    not_teaching(&studio)?;
    let p = plan(&input, word_limit, context.unwrap_or(256));
    if let Some(e) = p.errors.first() {
        return Err(e.clone());
    }
    // A word limit starts with no slots (one is added per learned word), like Bicameral.
    let vocab = match word_limit {
        Some(_) => yendory_engine::tokenizer::BASE,
        None => yendory_engine::tokenizer::BASE + 256,
    };
    let (config, design) = input.build(vocab, context.unwrap_or(256))?;
    let seed = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1) | 1;
    let meta = install(&app, &studio, &name, Bicameral::with_design(config, Some(design), seed), "new")?;
    match word_limit {
        Some(_) => set_limit(&app, &studio, &meta.id, word_limit),
        None => Ok(meta),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(id: &str) -> NeuroInput {
        let p = engine_neuro::presets().into_iter().find(|p| p.id == id).unwrap();
        NeuroInput::from_preset(&p)
    }

    #[test]
    fn the_builder_makes_any_combination_of_networks() {
        // Every preset, as the builder shows it, builds the preset's own design.
        for p in engine_neuro::presets() {
            let (config, design) =
                NeuroInput::from_preset(&p).build(p.config.vocab, p.config.max_tokens).unwrap();
            assert_eq!(design, p.design, "{}", p.id);
            assert_eq!(config.blocks, p.config.blocks);
        }
        // One network alone, without the transformer.
        let mut one = input("neuro-full-1m");
        one.transformer = false;
        one.memory = None;
        one.associative = None;
        one.experts = None;
        one.consolidation = None;
        let (config, design) = one.build(384, 256).unwrap();
        assert_eq!((config.blocks, config.split), (0, 0));
        assert_eq!(design.modules().len(), 1);
        let p1 = plan(&one, None, 256);
        assert!(p1.errors.is_empty(), "{:?}", p1.errors);
        assert!(p1.parts.iter().any(|p| p.name == "relational network"));
        assert!(!p1.parts.iter().any(|p| p.name == "transformer"));
        // The transformer alone: no bus, no controller.
        let mut core = input("neuro-full-1m");
        core.memory = None;
        core.associative = None;
        core.relational = None;
        core.experts = None;
        core.consolidation = None;
        let p2 = plan(&core, None, 256);
        assert!(p2.errors.is_empty(), "{:?}", p2.errors);
        assert!(!p2.parts.iter().any(|p| p.name == "communication bus"));
        // Nothing at all is refused with a reason.
        let mut none = core.clone();
        none.transformer = false;
        assert!(!plan(&none, None, 256).errors.is_empty());
    }
}
