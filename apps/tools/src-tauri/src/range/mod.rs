//! Yendory Range: a small first-person test game in its own window, for proving game training
//! end to end with a known answer (docs/GAME_TRAINING.md, "A test range").
//!
//! It is played like any game: Tools sees it only through the screen and drives it only through
//! virtual input. As an optional integration it also reports the truth on standard output (what
//! it received, where the player is, hits and the score) and takes a few commands on standard
//! input (`state`, `reset <seed>`, `quit`), so tests and evaluations can check against it.
//!
//! The game: an arena of blocks, three targets. W A S D move, Shift runs, the mouse turns, the
//! left button fires (once per press); a hit scores and the target moves elsewhere. An episode
//! lasts a set time, then the next one starts on a new layout.
use serde::Serialize;

pub mod app;
pub mod render;

/// Radians turned per unit of mouse movement.
pub const TURN_PER_UNIT: f32 = 0.0025;
/// Cells a second, walking.
pub const WALK: f32 = 2.5;
/// Running speed is walking speed times this.
pub const RUN: f32 = 1.8;
pub const TARGETS: usize = 3;
pub const TARGET_RADIUS: f32 = 0.25;
const PLAYER_RADIUS: f32 = 0.2;
const SIZE: usize = 16;

/// The controls the range reads, as held this moment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Held {
    pub w: bool,
    pub a: bool,
    pub s: bool,
    pub d: bool,
    pub shift: bool,
    pub fire: bool,
}

/// Something that happened, reported on standard output.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "ev", rename_all = "camelCase")]
pub enum Event {
    Shot { hit: bool },
    Episode { seed: u64, score: u32, shots: u32, hits: u32 },
}

/// The truth, for tests and evaluations.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct State {
    pub seed: u64,
    pub x: f32,
    pub y: f32,
    pub angle: f32,
    pub score: u32,
    pub shots: u32,
    pub hits: u32,
    pub time_ms: u64,
    pub episode_ms: u64,
    pub held: Held,
    /// Mouse units received since the episode began (horizontal, vertical).
    pub mouse_x: f64,
    pub mouse_y: f64,
    /// Presses received since the episode began.
    pub presses: u32,
    pub focused: bool,
    /// Where the targets are, and whether each is in line of sight from the player (the truth,
    /// for checks and tests; never shown to a network).
    pub targets: Vec<(f32, f32)>,
    pub visible: Vec<bool>,
}

pub struct Range {
    pub seed: u64,
    pub cells: Vec<bool>,
    pub x: f32,
    pub y: f32,
    pub angle: f32,
    pub targets: Vec<(f32, f32)>,
    pub score: u32,
    pub shots: u32,
    pub hits: u32,
    pub time_ms: u64,
    pub episode_ms: u64,
    /// Time left on the hit marker.
    pub flash_ms: u64,
    pub held: Held,
    pub mouse_x: f64,
    pub mouse_y: f64,
    pub presses: u32,
    pub focused: bool,
    rng: u64,
}

impl Range {
    pub fn new(seed: u64, episode_ms: u64) -> Range {
        let mut r = Range {
            seed,
            cells: vec![false; SIZE * SIZE],
            x: 0.0,
            y: 0.0,
            angle: 0.0,
            targets: Vec::new(),
            score: 0,
            shots: 0,
            hits: 0,
            time_ms: 0,
            episode_ms,
            flash_ms: 0,
            held: Held::default(),
            mouse_x: 0.0,
            mouse_y: 0.0,
            presses: 0,
            focused: false,
            rng: 0,
        };
        r.reset(seed);
        r
    }

    /// A new episode on the layout of `seed`.
    pub fn reset(&mut self, seed: u64) {
        self.seed = seed;
        self.rng = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
        self.cells = vec![false; SIZE * SIZE];
        for i in 0..SIZE {
            for j in [0, SIZE - 1] {
                self.cells[i * SIZE + j] = true;
                self.cells[j * SIZE + i] = true;
            }
        }
        for y in 1..SIZE - 1 {
            for x in 1..SIZE - 1 {
                let near_start = x.abs_diff(SIZE / 2) <= 1 && y.abs_diff(SIZE / 2) <= 1;
                if !near_start && self.next() % 100 < 13 {
                    self.cells[y * SIZE + x] = true;
                }
            }
        }
        self.close_unreachable();
        self.x = SIZE as f32 / 2.0 + 0.5;
        self.y = SIZE as f32 / 2.0 + 0.5;
        self.angle = (self.next() % 628) as f32 / 100.0;
        self.targets.clear();
        for _ in 0..TARGETS {
            let t = self.free_spot();
            self.targets.push(t);
        }
        self.score = 0;
        self.shots = 0;
        self.hits = 0;
        self.time_ms = 0;
        self.flash_ms = 0;
        self.mouse_x = 0.0;
        self.mouse_y = 0.0;
        self.presses = 0;
    }

    fn next(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Walls in every open cell the start cannot reach, so every target can be reached.
    fn close_unreachable(&mut self) {
        let mut seen = vec![false; SIZE * SIZE];
        let mut stack = vec![(SIZE / 2) * SIZE + SIZE / 2];
        while let Some(i) = stack.pop() {
            if seen[i] || self.cells[i] {
                continue;
            }
            seen[i] = true;
            stack.extend([i - 1, i + 1, i - SIZE, i + SIZE]);
        }
        for (i, wall) in self.cells.iter_mut().enumerate() {
            if !seen[i] {
                *wall = true;
            }
        }
    }

    /// An open cell at least three cells from the player and away from other targets.
    fn free_spot(&mut self) -> (f32, f32) {
        for _ in 0..1000 {
            let x = (self.next() % SIZE as u64) as usize;
            let y = (self.next() % SIZE as u64) as usize;
            let (cx, cy) = (x as f32 + 0.5, y as f32 + 0.5);
            let far = (cx - self.x).hypot(cy - self.y) >= 3.0;
            let apart =
                self.targets.iter().filter(|t| t.0.is_finite()).all(|t| (t.0 - cx).hypot(t.1 - cy) >= 1.5);
            if !self.cells[y * SIZE + x] && far && apart {
                return (cx, cy);
            }
        }
        (self.x + 1.0, self.y)
    }

    pub fn wall(&self, x: f32, y: f32) -> bool {
        if x < 0.0 || y < 0.0 || x >= SIZE as f32 || y >= SIZE as f32 {
            return true;
        }
        self.cells[y as usize * SIZE + x as usize]
    }

    fn blocked(&self, x: f32, y: f32) -> bool {
        let r = PLAYER_RADIUS;
        self.wall(x - r, y - r)
            || self.wall(x + r, y - r)
            || self.wall(x - r, y + r)
            || self.wall(x + r, y + r)
    }

    /// Mouse movement received (units).
    pub fn mouse(&mut self, dx: f64, dy: f64) {
        self.mouse_x += dx;
        self.mouse_y += dy;
        if self.over() {
            return;
        }
        self.angle = (self.angle + dx as f32 * TURN_PER_UNIT).rem_euclid(std::f32::consts::TAU);
    }

    /// A control pressed or released. A press of fire shoots.
    pub fn set(&mut self, held: Held, events: &mut Vec<Event>) {
        let fired = held.fire && !self.held.fire;
        let pressed = [
            (held.w, self.held.w),
            (held.a, self.held.a),
            (held.s, self.held.s),
            (held.d, self.held.d),
            (held.shift, self.held.shift),
            (held.fire, self.held.fire),
        ]
        .iter()
        .filter(|(now, before)| *now && !*before)
        .count();
        self.presses += pressed as u32;
        self.held = held;
        if fired && !self.over() {
            let hit = self.shoot();
            events.push(Event::Shot { hit });
        }
    }

    pub fn over(&self) -> bool {
        self.time_ms >= self.episode_ms
    }

    /// Distance to the first wall along `angle`.
    pub fn ray(&self, angle: f32) -> f32 {
        self.cast(angle).0
    }

    /// Distance to the first wall along `angle`, and whether it was hit on an east/west face.
    pub fn cast(&self, angle: f32) -> (f32, bool) {
        let (dx, dy) = (angle.cos(), angle.sin());
        let (mut cx, mut cy) = (self.x as i32, self.y as i32);
        let step_x = if dx < 0.0 { -1 } else { 1 };
        let step_y = if dy < 0.0 { -1 } else { 1 };
        let delta_x = if dx == 0.0 { f32::INFINITY } else { (1.0 / dx).abs() };
        let delta_y = if dy == 0.0 { f32::INFINITY } else { (1.0 / dy).abs() };
        let mut side_x = if dx < 0.0 { self.x - cx as f32 } else { cx as f32 + 1.0 - self.x } * delta_x;
        let mut side_y = if dy < 0.0 { self.y - cy as f32 } else { cy as f32 + 1.0 - self.y } * delta_y;
        for _ in 0..64 {
            let (d, east_west) = if side_x < side_y {
                cx += step_x;
                let d = side_x;
                side_x += delta_x;
                (d, true)
            } else {
                cy += step_y;
                let d = side_y;
                side_y += delta_y;
                (d, false)
            };
            if self.wall(cx as f32, cy as f32) {
                return (d, east_west);
            }
        }
        (64.0, false)
    }

    /// Fires along the view: the nearest target in front of the first wall is hit.
    fn shoot(&mut self) -> bool {
        self.shots += 1;
        let wall = self.ray(self.angle);
        let mut best: Option<(usize, f32)> = None;
        for (i, &(tx, ty)) in self.targets.iter().enumerate() {
            let (vx, vy) = (tx - self.x, ty - self.y);
            let dist = vx.hypot(vy);
            let off = angle_diff(vy.atan2(vx), self.angle).abs();
            if dist < wall && off < (TARGET_RADIUS / dist).atan() && best.is_none_or(|b| dist < b.1) {
                best = Some((i, dist));
            }
        }
        let Some((i, _)) = best else { return false };
        self.hits += 1;
        self.score += 1;
        self.flash_ms = 150;
        self.targets[i] = (f32::NAN, f32::NAN);
        self.targets[i] = self.free_spot();
        true
    }

    /// Advances time by `ms`. Returns the episode's end when it ends now.
    pub fn step(&mut self, ms: u64) -> Option<Event> {
        if self.over() {
            return None;
        }
        self.flash_ms = self.flash_ms.saturating_sub(ms);
        let h = self.held;
        let ahead = h.w as i32 - h.s as i32;
        let side = h.d as i32 - h.a as i32;
        if ahead != 0 || side != 0 {
            let speed = WALK * if h.shift { RUN } else { 1.0 } * ms as f32 / 1000.0;
            let (c, s) = (self.angle.cos(), self.angle.sin());
            // Right of the view is +90 degrees (y grows downwards on the map).
            let (mx, my) = (c * ahead as f32 - s * side as f32, s * ahead as f32 + c * side as f32);
            let len = mx.hypot(my);
            let (mx, my) = (mx / len * speed, my / len * speed);
            if !self.blocked(self.x + mx, self.y) {
                self.x += mx;
            }
            if !self.blocked(self.x, self.y + my) {
                self.y += my;
            }
        }
        self.time_ms = (self.time_ms + ms).min(self.episode_ms);
        self.over().then_some(Event::Episode {
            seed: self.seed,
            score: self.score,
            shots: self.shots,
            hits: self.hits,
        })
    }

    pub fn state(&self) -> State {
        State {
            seed: self.seed,
            x: self.x,
            y: self.y,
            angle: self.angle,
            score: self.score,
            shots: self.shots,
            hits: self.hits,
            time_ms: self.time_ms,
            episode_ms: self.episode_ms,
            held: self.held,
            mouse_x: self.mouse_x,
            mouse_y: self.mouse_y,
            presses: self.presses,
            focused: self.focused,
            targets: self.targets.clone(),
            visible: self
                .targets
                .iter()
                .map(|&(tx, ty)| {
                    self.ray((ty - self.y).atan2(tx - self.x)) > (tx - self.x).hypot(ty - self.y)
                })
                .collect(),
        }
    }

    pub fn size(&self) -> usize {
        SIZE
    }
}

/// `a - b`, in -pi..pi.
pub fn angle_diff(a: f32, b: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    (a - b + PI).rem_euclid(TAU) - PI
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_repeat_and_targets_are_reachable_and_open() {
        let a = Range::new(7, 60_000);
        let b = Range::new(7, 60_000);
        assert_eq!(a.cells, b.cells);
        assert_eq!(a.targets, b.targets);
        assert_ne!(Range::new(8, 60_000).cells, a.cells);
        for &(x, y) in &a.targets {
            assert!(!a.wall(x, y));
            assert!((x - a.x).hypot(y - a.y) >= 3.0);
        }
        assert!(!a.wall(a.x, a.y));
    }

    #[test]
    fn walking_forward_moves_along_the_view_and_running_is_faster() {
        let mut r = Range::new(3, 60_000);
        // Open the arena so walls don't stop the walk.
        for y in 1..SIZE - 1 {
            for x in 1..SIZE - 1 {
                r.cells[y * SIZE + x] = false;
            }
        }
        r.angle = 0.0;
        let mut ev = Vec::new();
        r.set(Held { w: true, ..Held::default() }, &mut ev);
        let x0 = r.x;
        r.step(400);
        assert!((r.x - x0 - WALK * 0.4).abs() < 1e-4, "{}", r.x - x0);
        r.set(Held { w: true, shift: true, ..Held::default() }, &mut ev);
        let x1 = r.x;
        r.step(400);
        assert!((r.x - x1 - WALK * RUN * 0.4).abs() < 1e-4);
        // D strafes to the right of the view: +y when looking along +x.
        r.set(Held { d: true, ..Held::default() }, &mut ev);
        let y0 = r.y;
        r.step(200);
        assert!(r.y > y0);
        assert_eq!(r.presses, 3);
    }

    #[test]
    fn the_mouse_turns_and_fire_hits_a_target_in_view() {
        let mut r = Range::new(5, 60_000);
        let (tx, ty) = r.targets[0];
        let want = (ty - r.y).atan2(tx - r.x);
        let units = angle_diff(want, r.angle) / TURN_PER_UNIT;
        r.mouse(units as f64, 0.0);
        assert!(angle_diff(r.angle, want).abs() < 1e-3);
        // Clear the line of fire.
        for y in 1..SIZE - 1 {
            for x in 1..SIZE - 1 {
                r.cells[y * SIZE + x] = false;
            }
        }
        let mut ev = Vec::new();
        r.set(Held { fire: true, ..Held::default() }, &mut ev);
        assert_eq!(ev, vec![Event::Shot { hit: true }]);
        assert_eq!((r.score, r.shots), (1, 1));
        // Holding fire does not shoot again; a miss counts a shot.
        r.set(Held { fire: true, ..Held::default() }, &mut ev);
        assert_eq!(ev.len(), 1);
        r.mouse(1256.0, 0.0);
        r.set(Held::default(), &mut ev);
        r.set(Held { fire: true, ..Held::default() }, &mut ev);
        assert_eq!(r.shots, 2);
    }

    #[test]
    fn an_episode_ends_once() {
        let mut r = Range::new(1, 1000);
        assert_eq!(r.step(600), None);
        assert!(matches!(r.step(600), Some(Event::Episode { .. })));
        assert_eq!(r.step(600), None);
        let x = r.x;
        r.set(Held { w: true, ..Held::default() }, &mut Vec::new());
        r.step(500);
        assert_eq!(r.x, x);
    }
}
