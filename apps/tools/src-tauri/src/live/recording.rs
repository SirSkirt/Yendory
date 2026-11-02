//! Recordings: a game played (by a person, later by a model) as pictures and controls in sync.
//! A recording is a folder `games/<profile>/recordings/<id>/`:
//!
//! - `recording.json`: [`Info`]: the game, who played, training or testing, the picture's size,
//!   the controls in order, the length, sizes, and whether it ended cleanly;
//! - `frames.bin`: the pictures one after another, each compressed on its own (deflate), so any
//!   picture can be read directly;
//! - `ticks.bin`: one [`Tick`] (40 bytes, little-endian) per picture: its time, where its picture
//!   is, and the controls: every control that was **active at any moment** since the previous
//!   tick (so a click shorter than a tick still counts) and the mouse movement since then;
//! - `events.jsonl`: the raw presses, releases and movements with their times (microseconds), and
//!   the controls held whenever recording (re)starts, from which every tick's controls can be
//!   rebuilt ([`validate`]).
//!
//! The picture of tick k is taken at its time, after the controls of tick k; what the player did
//! after seeing it is in tick k+1 (and later ticks: people react in a few hundred milliseconds).
//! Training chooses that offset. Ticks are only recorded while the game is in front; the first
//! tick after a pause (and the first of all) has [`AFTER_GAP`].
use super::controls::Button;
use crate::Res;
use flate2::read::DeflateDecoder;
use flate2::write::DeflateEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const FORMAT: u32 = 1;
pub const TICK_BYTES: usize = 40;
/// The first tick, and the first after a pause: no history before it.
pub const AFTER_GAP: u8 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    /// Teaches.
    Train,
    /// Held out: only ever measured on.
    Test,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Info {
    pub format: u32,
    pub id: String,
    pub profile: String,
    /// The game's name when recorded.
    pub game: String,
    /// `person`, or (later) the model that played.
    pub player: String,
    pub split: Split,
    pub notes: String,
    /// Unix seconds.
    pub started: u64,
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub fps: u32,
    /// The keys in bit order (bit i of a tick's keys is `keys[i]`).
    pub keys: Vec<String>,
    /// The buttons in bit order.
    pub buttons: Vec<Button>,
    pub mouse: bool,
    pub ticks: u64,
    pub duration_ms: u64,
    /// Time the game was not in front (not recorded).
    pub paused_ms: u64,
    /// Ticks missed because a picture took too long.
    pub dropped_ticks: u64,
    pub bytes: u64,
    /// False until the recording ended cleanly (a crash leaves what was written, readable).
    pub complete: bool,
    pub ended: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Tick {
    /// Microseconds since the recording began.
    pub t_us: u64,
    pub offset: u64,
    pub keys: u64,
    pub len: u32,
    pub dx: f32,
    pub dy: f32,
    pub buttons: u8,
    pub flags: u8,
}

impl Tick {
    pub fn encode(&self) -> [u8; TICK_BYTES] {
        let mut b = [0u8; TICK_BYTES];
        b[0..8].copy_from_slice(&self.t_us.to_le_bytes());
        b[8..16].copy_from_slice(&self.offset.to_le_bytes());
        b[16..24].copy_from_slice(&self.keys.to_le_bytes());
        b[24..28].copy_from_slice(&self.len.to_le_bytes());
        b[28..32].copy_from_slice(&self.dx.to_le_bytes());
        b[32..36].copy_from_slice(&self.dy.to_le_bytes());
        b[36] = self.buttons;
        b[37] = self.flags;
        b
    }

    pub fn decode(b: &[u8; TICK_BYTES]) -> Tick {
        let u64_at = |i: usize| u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        let f32_at = |i: usize| f32::from_le_bytes(b[i..i + 4].try_into().unwrap());
        Tick {
            t_us: u64_at(0),
            offset: u64_at(8),
            keys: u64_at(16),
            len: u32::from_le_bytes(b[24..28].try_into().unwrap()),
            dx: f32_at(28),
            dy: f32_at(32),
            buttons: b[36],
            flags: b[37],
        }
    }
}

pub fn folder(data: &Path, profile: &str) -> PathBuf {
    data.join("games").join(profile).join("recordings")
}

fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 64 && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
}

fn write_info(dir: &Path, info: &Info) -> Res<()> {
    let tmp = dir.join("recording.json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(info).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, dir.join("recording.json")).map_err(|e| e.to_string())
}

fn io(e: std::io::Error) -> String {
    format!("Writing the recording failed: {e}")
}

/// Writes a recording as it is made.
pub struct Writer {
    dir: PathBuf,
    pub info: Info,
    frames: BufWriter<File>,
    ticks: BufWriter<File>,
    events: BufWriter<File>,
    offset: u64,
    written: u64,
}

impl Writer {
    pub fn create(data: &Path, mut info: Info) -> Res<Writer> {
        if !valid_id(&info.id) {
            return Err("A recording's id may only hold letters, digits and -".into());
        }
        let dir = folder(data, &info.profile).join(&info.id);
        if dir.exists() {
            return Err("A recording with this id exists".into());
        }
        std::fs::create_dir_all(&dir).map_err(io)?;
        info.format = FORMAT;
        info.complete = false;
        write_info(&dir, &info)?;
        let open = |n: &str| File::create(dir.join(n)).map(BufWriter::new).map_err(io);
        Ok(Writer {
            frames: open("frames.bin")?,
            ticks: open("ticks.bin")?,
            events: open("events.jsonl")?,
            dir,
            info,
            offset: 0,
            written: 0,
        })
    }

    /// Bytes written so far.
    pub fn bytes(&self) -> u64 {
        self.written
    }

    /// One tick and its picture (its offset and length are filled in here).
    pub fn tick(&mut self, mut tick: Tick, pixels: &[u8]) -> Res<()> {
        let mut enc = DeflateEncoder::new(Vec::with_capacity(pixels.len() / 2), Compression::fast());
        enc.write_all(pixels).map_err(io)?;
        let packed = enc.finish().map_err(io)?;
        self.frames.write_all(&packed).map_err(io)?;
        tick.offset = self.offset;
        tick.len = packed.len() as u32;
        self.offset += packed.len() as u64;
        self.ticks.write_all(&tick.encode()).map_err(io)?;
        self.written += packed.len() as u64 + TICK_BYTES as u64;
        self.info.ticks += 1;
        Ok(())
    }

    pub fn event(&mut self, line: &Value) -> Res<()> {
        let mut s = line.to_string();
        s.push('\n');
        self.written += s.len() as u64;
        self.events.write_all(s.as_bytes()).map_err(io)
    }

    /// Writes everything out and marks the recording complete.
    pub fn finish(mut self, ended: &str) -> Res<Info> {
        for f in [&mut self.frames, &mut self.ticks, &mut self.events] {
            f.flush().map_err(io)?;
        }
        self.info.complete = true;
        self.info.ended = ended.to_string();
        self.info.bytes = dir_bytes(&self.dir);
        write_info(&self.dir, &self.info)?;
        Ok(self.info)
    }
}

fn dir_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum()
}

/// Reads a recording's info; one that did not end cleanly is described from what was written.
fn read_info(dir: &Path) -> Option<Info> {
    let mut info: Info = serde_json::from_slice(&std::fs::read(dir.join("recording.json")).ok()?).ok()?;
    if !info.complete {
        let n = std::fs::metadata(dir.join("ticks.bin")).map(|m| m.len()).unwrap_or(0) / TICK_BYTES as u64;
        info.ticks = n;
        info.bytes = dir_bytes(dir);
        if info.ended.is_empty() {
            info.ended = "Tools closed while recording; what was written is kept".into();
        }
    }
    Some(info)
}

/// The game's recordings, newest first.
pub fn list(data: &Path, profile: &str) -> Vec<Info> {
    let mut out: Vec<Info> = std::fs::read_dir(folder(data, profile))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| read_info(&e.path()))
        .collect();
    out.sort_by(|a, b| b.started.cmp(&a.started).then(b.id.cmp(&a.id)));
    out
}

/// Bytes the game's recordings take on disk.
pub fn usage(data: &Path, profile: &str) -> u64 {
    std::fs::read_dir(folder(data, profile))
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| dir_bytes(&e.path()))
        .sum()
}

fn dir_of(data: &Path, profile: &str, id: &str) -> Res<PathBuf> {
    let dir = folder(data, profile).join(id);
    if !valid_id(id) || !dir.join("recording.json").exists() {
        return Err("No such recording".into());
    }
    Ok(dir)
}

pub fn update(data: &Path, profile: &str, id: &str, split: Split, notes: &str) -> Res<Info> {
    let dir = dir_of(data, profile, id)?;
    let mut info: Info =
        serde_json::from_slice(&std::fs::read(dir.join("recording.json")).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    info.split = split;
    info.notes = notes.chars().take(2000).collect();
    write_info(&dir, &info)?;
    read_info(&dir).ok_or("The recording can't be read".into())
}

pub fn delete(data: &Path, profile: &str, id: &str) -> Res<()> {
    let dir = dir_of(data, profile, id)?;
    std::fs::remove_dir_all(dir).map_err(|e| e.to_string())
}

/// An open recording: its info, its ticks, and its pictures on demand.
pub struct Reader {
    pub info: Info,
    pub ticks: Vec<Tick>,
    frames: File,
    dir: PathBuf,
}

pub fn open(data: &Path, profile: &str, id: &str) -> Res<Reader> {
    let dir = dir_of(data, profile, id)?;
    let info = read_info(&dir).ok_or("The recording can't be read")?;
    let raw = std::fs::read(dir.join("ticks.bin")).map_err(|e| e.to_string())?;
    let ticks = raw.as_chunks::<TICK_BYTES>().0.iter().map(Tick::decode).collect();
    let frames = File::open(dir.join("frames.bin")).map_err(|e| e.to_string())?;
    Ok(Reader { info, ticks, frames, dir })
}

impl Reader {
    /// Picture `i`, uncompressed: width x height x channels bytes.
    pub fn frame(&mut self, i: usize) -> Res<Vec<u8>> {
        let t = *self.ticks.get(i).ok_or("No such picture")?;
        let mut packed = vec![0u8; t.len as usize];
        self.frames.seek(SeekFrom::Start(t.offset)).map_err(|e| e.to_string())?;
        self.frames.read_exact(&mut packed).map_err(|_| "The picture is cut short".to_string())?;
        let want = (self.info.width * self.info.height * self.info.channels as u32) as usize;
        let mut out = Vec::with_capacity(want);
        DeflateDecoder::new(&packed[..])
            .read_to_end(&mut out)
            .map_err(|_| "The picture is damaged".to_string())?;
        if out.len() != want {
            return Err(format!("Picture {i} has {} bytes, not {want}", out.len()));
        }
        Ok(out)
    }

    pub fn events_path(&self) -> PathBuf {
        self.dir.join("events.jsonl")
    }

    /// The recording's pictures, read from disk as they are needed (any number of threads at
    /// once), so training holds no more of them in memory than it is using.
    pub fn frames(&self) -> Res<Frames> {
        Ok(Frames {
            file: self.frames.try_clone().map_err(|e| e.to_string())?,
            spans: self.ticks.iter().map(|t| (t.offset, t.len)).collect(),
            size: (self.info.width * self.info.height * self.info.channels as u32) as usize,
        })
    }
}

/// A recording's pictures on disk: each read at its own position, so threads can share it.
pub struct Frames {
    file: File,
    spans: Vec<(u64, u32)>,
    size: usize,
}

impl Frames {
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Picture `i`, uncompressed.
    pub fn get(&self, i: usize) -> Res<Vec<u8>> {
        let (offset, len) = *self.spans.get(i).ok_or("No such picture")?;
        let mut packed = vec![0u8; len as usize];
        read_at(&self.file, &mut packed, offset).map_err(|_| "The picture is cut short".to_string())?;
        let mut out = Vec::with_capacity(self.size);
        DeflateDecoder::new(&packed[..])
            .read_to_end(&mut out)
            .map_err(|_| "The picture is damaged".to_string())?;
        if out.len() != self.size {
            return Err(format!("Picture {i} has {} bytes, not {}", out.len(), self.size));
        }
        Ok(out)
    }
}

#[cfg(unix)]
fn read_at(f: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(f, buf, offset)
}

#[cfg(windows)]
fn read_at(f: &File, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    while !buf.is_empty() {
        match std::os::windows::fs::FileExt::seek_read(f, buf, offset) {
            Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// What checking a recording found.
#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Validation {
    pub ticks: u64,
    pub seconds: f64,
    /// Pictures a second actually recorded (pauses left out).
    pub fps: f64,
    pub longest_gap_ms: f64,
    pub pictures_bad: u64,
    /// Ticks whose controls differ from those rebuilt from the raw events.
    pub ticks_disagreeing: u64,
    pub first_disagreement: Option<String>,
    pub times_out_of_order: u64,
    /// How many ticks each control was active in (keys, then buttons).
    pub active_ticks: Vec<(String, u64)>,
    pub mouse_x: f64,
    pub mouse_y: f64,
    pub ok: bool,
}

/// Reads every picture and rebuilds every tick's controls from the raw events.
pub fn validate(data: &Path, profile: &str, id: &str) -> Res<Validation> {
    let mut r = open(data, profile, id)?;
    let info = r.info.clone();
    let mut v = Validation { ticks: r.ticks.len() as u64, ..Validation::default() };
    for i in 0..r.ticks.len() {
        if r.frame(i).is_err() {
            v.pictures_bad += 1;
        }
    }
    // The raw events, in the order written.
    let events: Vec<Value> = std::fs::File::open(r.events_path())
        .map(|f| {
            BufReader::new(f)
                .lines()
                .map_while(Result::ok)
                .filter_map(|l| serde_json::from_str(&l).ok())
                .collect()
        })
        .unwrap_or_default();
    let key_bit = |k: &str| info.keys.iter().position(|x| x == k);
    let button_bit = |b: &str| info.buttons.iter().position(|x| x.name() == b);
    let mut held: BTreeSet<String> = BTreeSet::new();
    let mut active = held.clone();
    let (mut mx, mut my) = (0f64, 0f64);
    let mut e = 0;
    let mut counts = vec![0u64; info.keys.len() + info.buttons.len()];
    let mut last_t: Option<u64> = None;
    let mut recorded_us = 0u64;
    for (i, t) in r.ticks.iter().enumerate() {
        if let Some(prev) = last_t {
            if t.t_us <= prev {
                v.times_out_of_order += 1;
            } else if t.flags & AFTER_GAP == 0 {
                let gap = (t.t_us - prev) as f64 / 1000.0;
                v.longest_gap_ms = v.longest_gap_ms.max(gap);
                recorded_us += t.t_us - prev;
            }
        }
        last_t = Some(t.t_us);
        while e < events.len() && events[e]["t"].as_u64().unwrap_or(u64::MAX) <= t.t_us {
            let ev = &events[e];
            e += 1;
            if let Some(snapshot) = ev.get("held") {
                held = snapshot
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|s| s.as_str().map(String::from))
                    .collect();
                active = held.clone();
                mx = 0.0;
                my = 0.0;
            } else if let Some(name) = ev.get("k").or_else(|| ev.get("b")).and_then(|n| n.as_str()) {
                if ev["d"].as_bool() == Some(true) {
                    held.insert(name.to_string());
                    active.insert(name.to_string());
                } else {
                    held.remove(name);
                }
            } else if let Some(m) = ev.get("m").and_then(|m| m.as_array()) {
                mx += m.first().and_then(|x| x.as_f64()).unwrap_or(0.0);
                my += m.get(1).and_then(|x| x.as_f64()).unwrap_or(0.0);
            }
        }
        let mut keys = 0u64;
        let mut buttons = 0u8;
        for name in &active {
            if let Some(b) = key_bit(name) {
                keys |= 1 << b;
            } else if let Some(b) = button_bit(name) {
                buttons |= 1 << b;
            }
        }
        let same_mouse = (t.dx - mx as f32).abs() < 1e-3 && (t.dy - my as f32).abs() < 1e-3;
        if keys != t.keys || buttons != t.buttons || !same_mouse {
            v.ticks_disagreeing += 1;
            if v.first_disagreement.is_none() {
                v.first_disagreement = Some(format!(
                    "tick {i}: recorded keys {:#x} buttons {:#x} mouse ({}, {}); rebuilt {keys:#x} {buttons:#x} ({mx}, {my})",
                    t.keys, t.buttons, t.dx, t.dy
                ));
            }
        }
        let (key_counts, button_counts) = counts.split_at_mut(info.keys.len());
        for (b, c) in key_counts.iter_mut().enumerate() {
            *c += (t.keys >> b) & 1;
        }
        for (b, c) in button_counts.iter_mut().enumerate() {
            *c += ((t.buttons >> b) & 1) as u64;
        }
        v.mouse_x += t.dx as f64;
        v.mouse_y += t.dy as f64;
        active = held.clone();
        mx = 0.0;
        my = 0.0;
    }
    v.seconds = recorded_us as f64 / 1e6;
    let intervals = r.ticks.iter().filter(|t| t.flags & AFTER_GAP == 0).count();
    v.fps = if v.seconds > 0.0 { intervals as f64 / v.seconds } else { 0.0 };
    v.active_ticks = info
        .keys
        .iter()
        .cloned()
        .chain(info.buttons.iter().map(|b| b.name().to_string()))
        .zip(counts)
        .collect();
    v.ok = v.pictures_bad == 0 && v.ticks_disagreeing == 0 && v.times_out_of_order == 0;
    Ok(v)
}

/// An event line: a key or button pressed or released.
pub fn control_line(t_us: u64, kind: &str, name: &str, down: bool) -> Value {
    json!({"t": t_us, kind: name, "d": down})
}

/// An event line: mouse movement.
pub fn motion_line(t_us: u64, dx: f64, dy: f64) -> Value {
    json!({"t": t_us, "m": [dx, dy]})
}

/// An event line: the controls held when recording (re)starts.
pub fn held_line(t_us: u64, held: &BTreeSet<String>) -> Value {
    json!({"t": t_us, "held": held.iter().collect::<Vec<_>>()})
}

#[cfg(test)]
mod tests {
    use super::*;

    pub fn info(id: &str) -> Info {
        Info {
            format: FORMAT,
            id: id.into(),
            profile: "test-game".into(),
            game: "Test".into(),
            player: "person".into(),
            split: Split::Train,
            notes: String::new(),
            started: 1,
            width: 4,
            height: 2,
            channels: 1,
            fps: 20,
            keys: vec!["w".into(), "shift".into()],
            buttons: vec![Button::Left],
            mouse: true,
            ticks: 0,
            duration_ms: 0,
            paused_ms: 0,
            dropped_ticks: 0,
            bytes: 0,
            complete: false,
            ended: String::new(),
        }
    }

    #[test]
    fn ticks_round_trip() {
        let t = Tick {
            t_us: 123_456,
            offset: 9,
            keys: 1 << 63 | 5,
            len: 77,
            dx: -1.5,
            dy: 2.25,
            buttons: 3,
            flags: 1,
        };
        assert_eq!(Tick::decode(&t.encode()), t);
    }

    #[test]
    fn a_recording_is_written_read_checked_and_removed() {
        let data = std::env::temp_dir().join(format!("tools-rec-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&data);
        let mut w = Writer::create(&data, info("r1")).unwrap();
        let none = BTreeSet::new();
        w.event(&held_line(0, &none)).unwrap();
        w.tick(Tick { t_us: 0, flags: AFTER_GAP, ..Tick::default() }, &[0; 8]).unwrap();
        // W pressed and released within one tick still counts; shift held across ticks.
        w.event(&control_line(10, "k", "w", true)).unwrap();
        w.event(&control_line(20, "k", "w", false)).unwrap();
        w.event(&control_line(30, "k", "shift", true)).unwrap();
        w.event(&motion_line(40, 3.0, -1.0)).unwrap();
        w.event(&motion_line(45, 2.0, 0.0)).unwrap();
        w.tick(Tick { t_us: 50_000, keys: 0b11, dx: 5.0, dy: -1.0, ..Tick::default() }, &[1; 8]).unwrap();
        w.event(&control_line(60_000, "b", "left", true)).unwrap();
        w.tick(Tick { t_us: 100_000, keys: 0b10, buttons: 1, ..Tick::default() }, &[2; 8]).unwrap();
        let done = w.finish("stopped").unwrap();
        assert!(done.complete);
        assert_eq!(done.ticks, 3);

        let mut r = open(&data, "test-game", "r1").unwrap();
        assert_eq!(r.ticks.len(), 3);
        assert_eq!(r.frame(2).unwrap(), vec![2; 8]);
        let v = validate(&data, "test-game", "r1").unwrap();
        assert!(v.ok, "{v:?}");
        assert_eq!(v.active_ticks, vec![("w".into(), 1), ("shift".into(), 2), ("left".into(), 1)]);
        assert!((v.fps - 20.0).abs() < 1e-6);

        // A recorded tick that disagrees with its events is caught.
        let dir = folder(&data, "test-game").join("r1");
        let mut raw = std::fs::read(dir.join("ticks.bin")).unwrap();
        raw[TICK_BYTES + 16] = 0b01;
        std::fs::write(dir.join("ticks.bin"), raw).unwrap();
        let v = validate(&data, "test-game", "r1").unwrap();
        assert_eq!(v.ticks_disagreeing, 1);
        assert!(!v.ok);

        let u = update(&data, "test-game", "r1", Split::Test, "held out").unwrap();
        assert_eq!(u.split, Split::Test);
        assert_eq!(list(&data, "test-game").len(), 1);
        assert!(usage(&data, "test-game") > 0);
        assert!(open(&data, "test-game", "../x").is_err());
        delete(&data, "test-game", "r1").unwrap();
        assert!(list(&data, "test-game").is_empty());
        let _ = std::fs::remove_dir_all(&data);
    }

    #[test]
    fn a_recording_cut_short_is_still_readable() {
        let data = std::env::temp_dir().join(format!("tools-rec-cut-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&data);
        let mut w = Writer::create(&data, info("r2")).unwrap();
        w.tick(Tick { flags: AFTER_GAP, ..Tick::default() }, &[0; 8]).unwrap();
        w.tick(Tick { t_us: 50_000, ..Tick::default() }, &[0; 8]).unwrap();
        // Flush without finishing (as if Tools closed).
        w.frames.flush().unwrap();
        w.ticks.flush().unwrap();
        drop(w);
        let l = list(&data, "test-game");
        assert_eq!((l[0].ticks, l[0].complete), (2, false));
        assert!(!l[0].ended.is_empty());
        let _ = std::fs::remove_dir_all(&data);
    }
}
