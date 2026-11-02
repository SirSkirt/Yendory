//! `.nnova2` checkpoint reading and writing (layout in docs/FORMAT.md).
//! The layout is unchanged from Neural Nova and the numpy reference; files are interchangeable.
use super::{spec, Bicameral, Config};
use crate::skill::{layout_len, Adam, Skill, SkillSpec, MAX_SKILL_PARAMETERS};
use std::path::Path;

const MAGIC: &[u8; 8] = b"NNOVA2\0\0";
/// Nova Neuro checkpoints: a separate, versioned format (`.nneuro`), never read as `.nnova2`.
const NEURO_MAGIC: &[u8; 8] = b"NNEURO1\0";

/// The architecture header of a `.nneuro` checkpoint. Format 1: no game skills; format 2: the
/// skills are listed here and their weights follow the learned words (docs/FORMAT.md).
#[derive(serde::Serialize, serde::Deserialize)]
struct NeuroHeader {
    architecture: String,
    format: u32,
    config: Config,
    design: super::NeuroSpec,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    skills: Vec<SkillSpec>,
}
const WORDS_MAGIC: &[u8; 8] = b"WORDUNIT";
const SKILLS_MAGIC: &[u8; 8] = b"SKILLSET";
/// The header holds every skill's description (and its trainer's notes): larger than a bare
/// configuration, still bounded.
const MAX_CONFIG_JSON: usize = 8192;
const MAX_HEADER_WITH_SKILLS: usize = 16 << 20;
const MAX_SKILLS: usize = 64;

/// Rejects configurations that would panic or allocate absurdly before any weights are read.
pub(crate) fn validate_config(c: &Config) -> Result<(), String> {
    // Zero transformer blocks is a valid Nova Neuro shape (its networks alone); Nova Bicameral
    // needs at least one upper block (checked where the architecture is known).
    let dims = [c.d, c.heads, c.ff, c.window_chars, c.r1, c.r2, c.max_tokens];
    if dims.iter().any(|&x| x == 0 || x > 1 << 16)
        || c.blocks > 1 << 16
        || c.vocab == 0
        || c.vocab > super::MAX_VOCAB
        || c.loops == 0
        || c.loops > 64
    {
        return Err("Invalid Nova-2 configuration dimensions".into());
    }
    if c.d % c.heads != 0
        || (c.d / c.heads) % 2 != 0
        || (c.blocks > 0 && c.split >= c.blocks)
        || c.split > c.blocks
    {
        return Err("Invalid Nova-2 configuration shape".into());
    }
    if !c.rope_base.is_finite() || c.rope_base <= 0. || !c.gate_bias.is_finite() {
        return Err("Invalid Nova-2 configuration constants".into());
    }
    // The vocabulary is at most 2^28 and every other dimension at most 2^16, so these products
    // cannot overflow u64.
    let (v, d, ff, b, w, r1, r2) = (
        c.vocab as u64,
        c.d as u64,
        c.ff as u64,
        c.blocks as u64,
        c.window_chars as u64,
        c.r1 as u64,
        c.r2 as u64,
    );
    let rough =
        v * d + b * (4 * d * d + 3 * ff * d) + r1 * w * 16 + r2 * r1 + r2 * d + d * (r1 + r2) + 2 * d * d;
    if rough > super::MAX_PARAMETERS as u64 {
        return Err(format!(
            "This checkpoint describes a model of about {} parameters; the Engine supports up to {}",
            rough,
            super::MAX_PARAMETERS
        ));
    }
    Ok(())
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.data.len() - self.pos < n {
            return Err("Nova-2 checkpoint is truncated".into());
        }
        let out = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32s(&mut self, out: &mut [f32]) -> Result<(), String> {
        let bytes = self.take(out.len() * 4)?;
        for (x, b) in out.iter_mut().zip(bytes.chunks_exact(4)) {
            *x = f32::from_le_bytes(b.try_into().unwrap());
            if !x.is_finite() {
                return Err("Nova-2 checkpoint has invalid parameter".into());
            }
        }
        Ok(())
    }
}

impl Bicameral {
    /// Serializes the model in `.nnova2` layout. A model loaded from a checkpoint and not
    /// reconfigured reproduces that checkpoint's config JSON byte for byte.
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        if let Some(design) = &self.neuro {
            let header = NeuroHeader {
                architecture: "nova-neuro".into(),
                format: if self.skills.is_empty() { 1 } else { 2 },
                config: self.config.clone(),
                design: design.clone(),
                skills: self.skills.iter().map(|k| k.spec.clone()).collect(),
            };
            let json = serde_json::to_vec(&header).map_err(|e| e.to_string())?;
            let mut out = self.body(NEURO_MAGIC, &json);
            if !self.skills.is_empty() {
                out.extend_from_slice(SKILLS_MAGIC);
                for k in &self.skills {
                    k.check()?;
                    out.extend_from_slice(&k.adam.t.to_le_bytes());
                    for arr in [&k.accepted, &k.latest, &k.adam.m, &k.adam.v] {
                        for &x in arr {
                            out.extend_from_slice(&x.to_le_bytes());
                        }
                    }
                }
            }
            return Ok(out);
        }
        if !self.skills.is_empty() {
            return Err("Game skills are kept only by Nova Neuro models".into());
        }
        let config = match &self.stored_config_json {
            Some(raw) if serde_json::from_slice::<Config>(raw).ok().as_ref() == Some(&self.config) => {
                raw.clone()
            }
            _ => serde_json::to_vec(&self.config).map_err(|e| e.to_string())?,
        };
        Ok(self.body(MAGIC, &config))
    }

    /// Magic, header JSON, step, weights and Adam state, learned words: the layout both
    /// formats share.
    fn body(&self, magic: &[u8; 8], config: &[u8]) -> Vec<u8> {
        let n = self.weights.len();
        let mut w =
            Vec::with_capacity(8 + 4 + config.len() + 8 + 12 * n + 12 + 28 * self.tokenizer.words().len());
        w.extend_from_slice(magic);
        w.extend_from_slice(&(config.len() as u32).to_le_bytes());
        w.extend_from_slice(config);
        w.extend_from_slice(&self.step.to_le_bytes());
        for arr in [&self.weights, &self.momentum, &self.variance] {
            for &x in arr {
                w.extend_from_slice(&x.to_le_bytes());
            }
        }
        w.extend_from_slice(WORDS_MAGIC);
        w.extend_from_slice(&(self.tokenizer.words().len() as u32).to_le_bytes());
        for s in self.tokenizer.words() {
            w.extend_from_slice(&(s.len() as u32).to_le_bytes());
            w.extend_from_slice(s.as_bytes());
        }
        w
    }

    /// Reads a `.nnova2` checkpoint (weights, Adam state, step, learned word units).
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let mut r = Reader { data, pos: 0 };
        let magic = r.take(8)?;
        let neuro = magic == NEURO_MAGIC;
        if magic != MAGIC && !neuro {
            return Err("Not a Nova-2 or Nova Neuro checkpoint".into());
        }
        let len = r.u32()? as usize;
        if len > if neuro { MAX_HEADER_WITH_SKILLS } else { MAX_CONFIG_JSON } {
            return Err("Invalid Nova-2 configuration".into());
        }
        let json = r.take(len)?;
        let mut skill_specs: Vec<SkillSpec> = Vec::new();
        let (config, design) = if neuro {
            let h: NeuroHeader = serde_json::from_slice(json).map_err(|e| e.to_string())?;
            let known = h.architecture == "nova-neuro"
                && match h.format {
                    1 => h.skills.is_empty(),
                    2 => !h.skills.is_empty(),
                    _ => false,
                };
            if !known {
                return Err(format!(
                    "Unsupported Nova Neuro checkpoint ({} format {})",
                    h.architecture, h.format
                ));
            }
            h.design.validate(&h.config)?;
            if h.skills.len() > MAX_SKILLS {
                return Err("The checkpoint lists more game skills than the Engine allows".into());
            }
            for (i, k) in h.skills.iter().enumerate() {
                k.arch.check()?;
                if layout_len(&k.arch) > MAX_SKILL_PARAMETERS {
                    return Err("A game skill in the checkpoint is larger than the Engine allows".into());
                }
                if h.skills[..i].iter().any(|o| o.game == k.game) {
                    return Err("The checkpoint has two skills for the same game".into());
                }
            }
            skill_specs = h.skills;
            (h.config, Some(h.design))
        } else {
            let c = serde_json::from_slice::<Config>(json).map_err(|e| e.to_string())?;
            if c.split >= c.blocks {
                return Err("Invalid Nova-2 configuration shape".into());
            }
            (c, None)
        };
        validate_config(&config)?;
        let n = spec(&config, design.as_ref()).last().map(|s| s.range.end).unwrap_or(0);
        if n > super::MAX_PARAMETERS {
            return Err(format!(
                "This design has {n} parameters; the Engine supports up to {}",
                super::MAX_PARAMETERS
            ));
        }
        if data.len() - r.pos < 8 + 12 * n {
            return Err("Nova-2 checkpoint is truncated".into());
        }
        let mut model = Self::with_design(config, design, 1);
        if !neuro {
            model.stored_config_json = Some(json.to_vec());
        }
        model.step = r.u64()?;
        r.f32s(&mut model.weights)?;
        r.f32s(&mut model.momentum)?;
        r.f32s(&mut model.variance)?;
        if r.take(8)? != WORDS_MAGIC {
            return Err("Nova-2 vocabulary trailer missing".into());
        }
        let count = r.u32()? as usize;
        if count > model.word_capacity() {
            return Err("The checkpoint has more learned words than its vocabulary holds".into());
        }
        for _ in 0..count {
            let n = r.u32()? as usize;
            if n > 24 {
                return Err("Invalid learned-word length".into());
            }
            let word = String::from_utf8(r.take(n)?.to_vec()).map_err(|e| e.to_string())?;
            model.tokenizer.push_word(word);
        }
        model.tokenizer.validate()?;
        if !skill_specs.is_empty() {
            if r.take(8)? != SKILLS_MAGIC {
                return Err("The checkpoint's game skills are missing".into());
            }
            for spec in skill_specs {
                let n = layout_len(&spec.arch);
                if data.len() - r.pos < 8 + 16 * n {
                    return Err("Nova-2 checkpoint is truncated".into());
                }
                let t = r.u64()?;
                let mut arrays = [vec![0f32; n], vec![0f32; n], vec![0f32; n], vec![0f32; n]];
                for a in arrays.iter_mut() {
                    r.f32s(a)?;
                }
                let [accepted, latest, m, v] = arrays;
                let skill = Skill { spec, accepted, latest, adam: Adam { m, v, t } };
                skill.check()?;
                model.skills.push(skill);
            }
        }
        if r.pos != data.len() {
            return Err("Unexpected Nova-2 checkpoint trailer".into());
        }
        Ok(model)
    }

    /// Writes atomically: a temporary file is synced, then renamed over the target.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        use std::io::Write;
        let target = path.as_ref();
        let tmp = target.with_extension("nova2.tmp");
        let bytes = self.to_bytes()?;
        let mut w = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
        w.write_all(&bytes).map_err(|e| e.to_string())?;
        w.sync_all().map_err(|e| e.to_string())?;
        drop(w);
        std::fs::rename(tmp, target).map_err(|e| e.to_string())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        Self::from_bytes(&std::fs::read(path).map_err(|e| e.to_string())?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> Config {
        Config {
            vocab: 24,
            d: 16,
            heads: 2,
            ff: 24,
            blocks: 3,
            split: 1,
            window_chars: 4,
            r1: 12,
            r2: 10,
            max_tokens: 16,
            ..Config::default()
        }
    }

    #[test]
    fn round_trip_fresh_model() {
        // Room for the 3 learned words (ids 128..130) next to the 128 characters.
        let mut m = Bicameral::with_config(Config { vocab: 131, ..small() }, 7);
        m.tokenizer.grow_from_corpus("Mira opens a box. Mira opens the box.");
        assert_eq!(m.tokenizer.words().len(), 3);
        m.step = 42;
        let bytes = m.to_bytes().unwrap();
        let back = Bicameral::from_bytes(&bytes).unwrap();
        assert_eq!(back.weights, m.weights);
        assert_eq!(back.tokenizer, m.tokenizer);
        assert_eq!(back.step, 42);
        assert_eq!(back.to_bytes().unwrap(), bytes);
    }

    #[test]
    fn rejects_malformed_files_without_panicking() {
        let bytes = Bicameral::with_config(small(), 7).to_bytes().unwrap();
        for cut in [0, 7, 12, 40, bytes.len() / 2, bytes.len() - 1] {
            assert!(Bicameral::from_bytes(&bytes[..cut]).is_err(), "cut at {cut}");
        }
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(Bicameral::from_bytes(&extra).is_err());
        let json = serde_json::to_vec(&Config { heads: 3, ..small() }).unwrap();
        let mut bad = MAGIC.to_vec();
        bad.extend_from_slice(&(json.len() as u32).to_le_bytes());
        bad.extend_from_slice(&json);
        assert!(Bicameral::from_bytes(&bad).is_err());
        let json = serde_json::to_vec(&Config { vocab: 1 << 40, ..small() }).unwrap();
        let mut huge = MAGIC.to_vec();
        huge.extend_from_slice(&(json.len() as u32).to_le_bytes());
        huge.extend_from_slice(&json);
        assert!(Bicameral::from_bytes(&huge).is_err());
    }

    fn skill(game: &str) -> Skill {
        let arch = crate::skill::check_arch();
        let spec = SkillSpec {
            game: game.into(),
            name: "Test range".into(),
            size: crate::skill::Size::Small,
            seed: 5,
            keys: vec!["w".into(), "a".into()],
            buttons: vec!["left".into()],
            arch,
            lineage: crate::skill::Lineage { generation: 0, parent: None, method: "imitation".into() },
            steps: 12,
            accepted_step: 10,
            created: 1,
            notes: serde_json::json!({"runs": [{"steps": 12}]}),
        };
        let mut k = Skill::new(spec).unwrap();
        k.latest[0] += 0.5;
        k.adam.t = 12;
        k.adam.m[1] = 0.25;
        k
    }

    fn neuro_model() -> Bicameral {
        let p = &crate::model::neuro::presets()[0];
        Bicameral::with_design(p.config.clone(), Some(p.design.clone()), 3)
    }

    #[test]
    fn game_skills_ride_in_format_2_and_leave_format_1_alone() {
        let mut m = neuro_model();
        let plain = m.to_bytes().unwrap();
        // Without skills: format 1, exactly as before.
        assert!(String::from_utf8_lossy(&plain[..400]).contains("\"format\":1"));
        assert!(!String::from_utf8_lossy(&plain[..400]).contains("skills"));
        m.skills.push(skill("yendory-range"));
        let bytes = m.to_bytes().unwrap();
        assert!(String::from_utf8_lossy(&bytes[..400]).contains("\"format\":2"));
        let back = Bicameral::from_bytes(&bytes).unwrap();
        assert_eq!(back.weights, m.weights);
        assert_eq!(back.skills.len(), 1);
        let (a, b) = (&back.skills[0], &m.skills[0]);
        assert_eq!(a.spec, b.spec);
        assert_eq!(
            (&a.accepted, &a.latest, &a.adam.m, &a.adam.v, a.adam.t),
            (&b.accepted, &b.latest, &b.adam.m, &b.adam.v, b.adam.t)
        );
        assert_eq!(back.to_bytes().unwrap(), bytes);
        // Removing the skills gives the format-1 file back.
        let mut none = back.clone();
        none.skills.clear();
        assert_eq!(none.to_bytes().unwrap(), plain);
        // Kept when the vocabulary is resized.
        let wider = back.with_word_capacity(back.word_capacity() + 4, 1).unwrap();
        assert_eq!(wider.skills.len(), 1);
    }

    #[test]
    fn damaged_or_impossible_skills_are_refused() {
        let mut m = neuro_model();
        m.skills.push(skill("g"));
        let bytes = m.to_bytes().unwrap();
        // Cut anywhere in the skills, or with a byte too many.
        let plain_len = {
            let mut p = m.clone();
            p.skills.clear();
            p.to_bytes().unwrap().len()
        };
        let extra = bytes.len() - plain_len;
        for cut in [bytes.len() - 1, bytes.len() - extra / 2, bytes.len() - extra + 4] {
            assert!(Bicameral::from_bytes(&bytes[..cut]).is_err(), "cut at {cut}");
        }
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(Bicameral::from_bytes(&longer).is_err());
        // Two skills for one game; a Bicameral model with a skill.
        m.skills.push(skill("g"));
        assert!(Bicameral::from_bytes(&m.to_bytes().unwrap()).is_err());
        let mut b = Bicameral::with_config(small(), 1);
        b.skills.push(skill("g"));
        assert!(b.to_bytes().is_err());
        // A skill whose weights don't fit its network is never written.
        let mut bad = neuro_model();
        let mut k = skill("g");
        k.latest.pop();
        bad.skills.push(k);
        assert!(bad.to_bytes().is_err());
    }
}
