//! Draws the range into a pixel buffer (0x00RRGGBB, row by row): walls by ray casting, the
//! targets as discs, a crosshair, a hit marker, the score and the time left.
use super::{angle_diff, Range, TARGET_RADIUS};

pub const WIDTH: usize = 320;
pub const HEIGHT: usize = 180;
/// Horizontal field of view.
const FOV: f32 = 1.15;

const CEILING: u32 = 0xE9E4D8;
const FLOOR: u32 = 0xC9BFA6;
const WALL: u32 = 0x3B4A6B;
const WALL_SIDE: u32 = 0x2C3852;
const TARGET: u32 = 0xC0392B;
const RING: u32 = 0xF4EFE6;
const INK: u32 = 0x1E2230;
const MARK: u32 = 0xF2C230;

fn shade(c: u32, f: f32) -> u32 {
    let f = f.clamp(0.0, 1.0);
    let ch = |s: u32| (((c >> s) & 0xFF) as f32 * f) as u32;
    (ch(16) << 16) | (ch(8) << 8) | ch(0)
}

/// Draws the view into `px` (WIDTH x HEIGHT).
pub fn draw(r: &Range, px: &mut [u32]) {
    let (w, h) = (WIDTH, HEIGHT);
    let half = (FOV / 2.0).tan();
    let focal = (w as f32 / 2.0) / half;
    let mut depth = [0f32; WIDTH];
    for y in 0..h {
        let c = if y < h / 2 { CEILING } else { FLOOR };
        px[y * w..(y + 1) * w].fill(c);
    }
    for (x, z) in depth.iter_mut().enumerate() {
        let off = ((x as f32 + 0.5) - w as f32 / 2.0) / focal;
        let a = r.angle + off.atan();
        let (dist, east_west) = r.cast(a);
        let d = dist * off.atan().cos();
        *z = d;
        let tall = (focal * 0.9 / d.max(0.05)) as i32;
        let top = (h as i32 / 2 - tall / 2).max(0);
        let bottom = (h as i32 / 2 + tall / 2).min(h as i32);
        let col = shade(if east_west { WALL } else { WALL_SIDE }, 1.15 - d / 20.0);
        for y in top..bottom {
            px[y as usize * w + x] = col;
        }
    }
    // Targets, far ones first.
    let mut order: Vec<(f32, f32, f32)> =
        r.targets.iter().map(|&(tx, ty)| (tx - r.x, ty - r.y, (tx - r.x).hypot(ty - r.y))).collect();
    order.sort_by(|a, b| b.2.total_cmp(&a.2));
    for (vx, vy, dist) in order {
        let off = angle_diff(vy.atan2(vx), r.angle);
        if off.abs() > FOV {
            continue;
        }
        let d = dist * off.cos();
        if d < 0.1 {
            continue;
        }
        let cx = w as f32 / 2.0 + off.tan() * focal;
        let rad = TARGET_RADIUS / d * focal;
        let cy = h as f32 / 2.0;
        let (x0, x1) = ((cx - rad).floor().max(0.0) as usize, ((cx + rad).ceil() as usize).min(w));
        let (y0, y1) = ((cy - rad).floor().max(0.0) as usize, ((cy + rad).ceil() as usize).min(h));
        for x in x0..x1 {
            if depth[x] < d {
                continue;
            }
            for y in y0..y1 {
                let e = ((x as f32 + 0.5 - cx).hypot(y as f32 + 0.5 - cy)) / rad;
                if e <= 1.0 {
                    let c = if (0.45..0.7).contains(&e) { RING } else { TARGET };
                    px[y * w + x] = shade(c, 1.1 - d / 25.0);
                }
            }
        }
    }
    let (mx, my) = (w / 2, h / 2);
    for i in 3..7 {
        for (x, y) in [(mx + i, my), (mx - i, my), (mx, my + i), (mx, my - i)] {
            px[y * w + x] = INK;
        }
    }
    if r.flash_ms > 0 {
        for i in 5..10 {
            for (x, y) in [(mx + i, my + i), (mx - i, my - i), (mx + i, my - i), (mx - i, my + i)] {
                px[y * w + x] = MARK;
            }
        }
    }
    // Time left along the top, the score below it.
    let left = 1.0 - r.time_ms as f32 / r.episode_ms.max(1) as f32;
    for x in 0..(left * w as f32) as usize {
        px[x] = INK;
        px[w + x] = INK;
    }
    number(px, r.score, 4, 5, 2);
    if r.over() {
        for p in px.iter_mut() {
            *p = shade(*p, 0.45);
        }
        number(px, r.score, w / 2 - 12, h / 2 - 10, 4);
    }
}

/// Digits 0-9 in a 3x5 font, one bit a pixel, rows from the top.
const DIGITS: [[u8; 5]; 10] = [
    [7, 5, 5, 5, 7],
    [2, 6, 2, 2, 7],
    [7, 1, 7, 4, 7],
    [7, 1, 3, 1, 7],
    [5, 5, 7, 1, 1],
    [7, 4, 7, 1, 7],
    [7, 4, 7, 5, 7],
    [7, 1, 1, 2, 2],
    [7, 5, 7, 5, 7],
    [7, 5, 7, 1, 7],
];

fn number(px: &mut [u32], n: u32, x: usize, y: usize, scale: usize) {
    for (i, ch) in n.to_string().bytes().enumerate() {
        let glyph = DIGITS[(ch - b'0') as usize];
        for (row, bits) in glyph.iter().enumerate() {
            for col in 0..3 {
                if bits & (4 >> col) == 0 {
                    continue;
                }
                for dy in 0..scale {
                    for dx in 0..scale {
                        let (px_x, px_y) = (x + (i * 4 + col) * scale + dx, y + row * scale + dy);
                        if px_x < WIDTH && px_y < HEIGHT {
                            px[px_y * WIDTH + px_x] = INK;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turning_changes_the_picture_and_a_target_in_front_is_drawn() {
        let mut r = Range::new(5, 60_000);
        let mut a = vec![0; WIDTH * HEIGHT];
        draw(&r, &mut a);
        r.mouse(300.0, 0.0);
        let mut b = vec![0; WIDTH * HEIGHT];
        draw(&r, &mut b);
        assert_ne!(a, b);
        // Face the first target with nothing in the way: its colour is at the crosshair's side.
        for c in r.cells.iter_mut().enumerate().filter(|(i, _)| {
            let (x, y) = (i % 16, i / 16);
            x > 0 && y > 0 && x < 15 && y < 15
        }) {
            *c.1 = false;
        }
        let (tx, ty) = r.targets[0];
        r.angle = (ty - r.y).atan2(tx - r.x);
        draw(&r, &mut b);
        let row = &b[(HEIGHT / 2) * WIDTH..(HEIGHT / 2 + 1) * WIDTH];
        assert!(row.iter().any(|&p| (p >> 16) > 0x80 && (p & 0xFF) < 0x40), "no target colour");
    }
}
