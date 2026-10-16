//! `.nnova2` checkpoint reading and writing (layout in docs/FORMAT.md).
//! The layout is unchanged from Neural Nova and the numpy reference; files are interchangeable.
use super::{spec, Bicameral, Config};
use std::path::Path;

const MAGIC: &[u8; 8] = b"NNOVA2\0\0";
const WORDS_MAGIC: &[u8; 8] = b"WORDUNIT";
const MAX_CONFIG_JSON: usize = 8192;

/// Rejects configurations that would panic or allocate absurdly before any weights are read.
pub(crate) fn validate_config(c: &Config) -> Result<(), String> {
    let dims = [c.d, c.heads, c.ff, c.blocks, c.window_chars, c.r1, c.r2, c.max_tokens];
    if dims.iter().any(|&x| x == 0 || x > 1 << 16)
        || c.vocab == 0
        || c.vocab > super::MAX_VOCAB
        || c.loops == 0
        || c.loops > 64
    {
        return Err("Invalid Nova-2 configuration dimensions".into());
    }
    if c.d % c.heads != 0 || (c.d / c.heads) % 2 != 0 || c.split >= c.blocks {
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
        let config = match &self.stored_config_json {
            Some(raw) if serde_json::from_slice::<Config>(raw).ok().as_ref() == Some(&self.config) => {
                raw.clone()
            }
            _ => serde_json::to_vec(&self.config).map_err(|e| e.to_string())?,
        };
        let n = self.weights.len();
        let mut w =
            Vec::with_capacity(8 + 4 + config.len() + 8 + 12 * n + 12 + 28 * self.tokenizer.words().len());
        w.extend_from_slice(MAGIC);
        w.extend_from_slice(&(config.len() as u32).to_le_bytes());
        w.extend_from_slice(&config);
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
        Ok(w)
    }

    /// Reads a `.nnova2` checkpoint (weights, Adam state, step, learned word units).
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let mut r = Reader { data, pos: 0 };
        if r.take(8)? != MAGIC {
            return Err("Not a Nova-2 checkpoint".into());
        }
        let len = r.u32()? as usize;
        if len > MAX_CONFIG_JSON {
            return Err("Invalid Nova-2 configuration".into());
        }
        let json = r.take(len)?;
        let config: Config = serde_json::from_slice(json).map_err(|e| e.to_string())?;
        validate_config(&config)?;
        let n = spec(&config).last().map(|s| s.range.end).unwrap_or(0);
        if data.len() - r.pos < 8 + 12 * n {
            return Err("Nova-2 checkpoint is truncated".into());
        }
        let mut model = Self::with_config(config, 1);
        model.stored_config_json = Some(json.to_vec());
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
}
