//! Test actions written as text, one per line or separated by `;`, run through the safety layer:
//!
//! ```text
//! hold w 250            hold W for 250 ms
//! hold w+shift 500      W and Shift together
//! move 15 0             the mouse 15 units right (move 15 0 200: over 200 ms)
//! press right; move 40 0 300; release right    aim while holding the right button
//! click left            press and release (tap e: the same for a key)
//! wait 300              do nothing for 300 ms
//! release all
//! ```
//!
//! A script never leaves anything held: whatever it pressed is released when it ends.
use super::controls::Control;
use super::safety::Controller;
use std::time::{Duration, Instant};

/// How long a tap or a click holds its control.
pub const TAP_MS: u64 = 60;
/// How often a movement over time sends a step.
const STEP_MS: u64 = 10;

#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    Press(Vec<Control>),
    Release(Vec<Control>),
    ReleaseAll,
    Hold(Vec<Control>, u64),
    Move { dx: f32, dy: f32, ms: u64 },
    Wait(u64),
}

fn controls(word: &str) -> Result<Vec<Control>, String> {
    word.split('+')
        .map(|w| Control::parse(w).ok_or_else(|| format!("“{w}” is not a key or mouse button")))
        .collect()
}

fn millis(word: &str) -> Result<u64, String> {
    let w = word.trim().to_ascii_lowercase();
    let (n, scale) = if let Some(n) = w.strip_suffix("ms") {
        (n, 1.0)
    } else if let Some(n) = w.strip_suffix('s') {
        (n, 1000.0)
    } else {
        (w.as_str(), 1.0)
    };
    let v: f64 = n.parse().map_err(|_| format!("“{word}” is not a time"))?;
    if !(0.0..=60_000.0).contains(&(v * scale)) {
        return Err(format!("{word} is longer than a minute"));
    }
    Ok((v * scale) as u64)
}

fn number(word: Option<&str>, what: &str) -> Result<f32, String> {
    word.ok_or(format!("{what} is missing"))?.parse().map_err(|_| format!("{what} is not a number"))
}

pub fn parse(text: &str) -> Result<Vec<Step>, String> {
    let mut steps = Vec::new();
    for (i, part) in text.split(['\n', ';']).enumerate() {
        let part = part.trim();
        if part.is_empty() || part.starts_with('#') {
            continue;
        }
        let words: Vec<&str> = part.split_whitespace().collect();
        let line = |e: String| format!("Action {} (“{part}”): {e}", i + 1);
        let step = match words[0].to_ascii_lowercase().as_str() {
            "press" if words.len() == 2 => Step::Press(controls(words[1]).map_err(line)?),
            "release" if words.len() == 2 && words[1].eq_ignore_ascii_case("all") => Step::ReleaseAll,
            "release" if words.len() == 2 => Step::Release(controls(words[1]).map_err(line)?),
            "hold" if words.len() == 3 => {
                Step::Hold(controls(words[1]).map_err(line)?, millis(words[2]).map_err(line)?)
            }
            "tap" if words.len() == 2 => Step::Hold(controls(words[1]).map_err(line)?, TAP_MS),
            "click" if words.len() <= 2 => {
                Step::Hold(controls(words.get(1).copied().unwrap_or("left")).map_err(line)?, TAP_MS)
            }
            "move" if words.len() == 3 || words.len() == 4 => Step::Move {
                dx: number(words.get(1).copied(), "the x movement").map_err(line)?,
                dy: number(words.get(2).copied(), "the y movement").map_err(line)?,
                ms: words.get(3).map(|w| millis(w)).transpose().map_err(line)?.unwrap_or(0),
            },
            "wait" if words.len() == 2 => Step::Wait(millis(words[1]).map_err(line)?),
            _ => {
                return Err(line(
                    "write press, release, hold, tap, click, move or wait (for example “hold w 250”)".into(),
                ))
            }
        };
        steps.push(step);
    }
    if steps.is_empty() {
        return Err("Write at least one action".into());
    }
    Ok(steps)
}

/// Waits until `until`, letting the controller enforce its limits meanwhile. False when the
/// emergency stop was pressed.
fn wait(c: &std::sync::Mutex<Controller>, until: Instant) -> bool {
    loop {
        {
            let mut c = c.lock().unwrap();
            c.tick();
            if c.stopped() {
                return false;
            }
        }
        let now = Instant::now();
        if now >= until {
            return true;
        }
        std::thread::sleep((until - now).min(Duration::from_millis(STEP_MS)));
    }
}

/// Runs the steps through the controller, then releases everything.
pub fn run(c: &std::sync::Mutex<Controller>, steps: &[Step]) {
    'steps: for step in steps {
        match step {
            Step::Press(cs) => {
                let mut c = c.lock().unwrap();
                for &x in cs {
                    c.press(x);
                }
            }
            Step::Release(cs) => {
                let mut c = c.lock().unwrap();
                for &x in cs {
                    c.release(x, "");
                }
            }
            Step::ReleaseAll => c.lock().unwrap().release_all(""),
            Step::Hold(cs, ms) => {
                {
                    let mut c = c.lock().unwrap();
                    for &x in cs {
                        c.press(x);
                    }
                }
                let ok = wait(c, Instant::now() + Duration::from_millis(*ms));
                let mut c = c.lock().unwrap();
                for &x in cs.iter().rev() {
                    c.release(x, "");
                }
                if !ok {
                    break 'steps;
                }
            }
            Step::Move { dx, dy, ms } => {
                let n = (*ms / STEP_MS).max(1);
                let start = Instant::now();
                for i in 0..n {
                    if i > 0 && !wait(c, start + Duration::from_millis(i * STEP_MS)) {
                        break 'steps;
                    }
                    c.lock().unwrap().motion(dx / n as f32, dy / n as f32);
                }
            }
            Step::Wait(ms) => {
                if !wait(c, Instant::now() + Duration::from_millis(*ms)) {
                    break 'steps;
                }
            }
        }
        if c.lock().unwrap().stopped() {
            break;
        }
    }
    c.lock().unwrap().release_all("the actions ended");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live::controls::Button;
    use crate::live::safety::tests::controller;
    use crate::live::safety::Limits;
    use std::sync::Mutex;

    #[test]
    fn scripts_parse() {
        let s = parse("hold w+shift 500; move 15 0\nclick; tap e\npress right; move 40 0 300; release right; wait 0.2s; release all")
            .unwrap();
        assert_eq!(s[0], Step::Hold(vec![Control::Key("w"), Control::Key("shift")], 500));
        assert_eq!(s[1], Step::Move { dx: 15.0, dy: 0.0, ms: 0 });
        assert_eq!(s[2], Step::Hold(vec![Control::Button(Button::Left)], TAP_MS));
        assert_eq!(s[7], Step::Wait(200));
        assert_eq!(s[8], Step::ReleaseAll);
        assert!(parse("hold win 100").unwrap_err().contains("win"));
        assert!(parse("jump").is_err());
        assert!(parse("wait 2h").is_err());
    }

    #[test]
    fn a_script_holds_for_its_time_and_releases_everything() {
        let (c, f) = controller(&["w", "shift"], Limits::default());
        let c = Mutex::new(c);
        let start = Instant::now();
        run(&c, &parse("hold w+shift 120; press w; move 20 0 50").unwrap());
        assert!(start.elapsed() >= Duration::from_millis(160));
        let sent = f.sent.lock().unwrap().clone();
        assert_eq!(&sent[..5], ["+w", "+shift", "-shift", "-w", "+w"]);
        let moved: i32 = sent
            .iter()
            .filter_map(|s| s.strip_prefix('m'))
            .map(|m| m.split(',').next().unwrap().parse::<i32>().unwrap())
            .sum();
        assert_eq!(moved, 20);
        assert_eq!(sent.last().unwrap(), "-w");
        assert!(c.lock().unwrap().held().is_empty());
    }
}
