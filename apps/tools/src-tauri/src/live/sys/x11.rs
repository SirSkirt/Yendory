//! Linux on X11, through x11rb (plain Rust; errors come back as values, so a closed window never
//! ends the program): windows from the window manager's list (or the root's children without
//! one), pictures with GetImage, virtual input with the XTEST extension.
use super::{shrink, Desktop, Rect, WindowInfo};
use crate::live::controls::{Button, KeyInfo};
use crate::Res;
use std::collections::HashMap;
use x11rb::connection::Connection;
use x11rb::protocol::xproto::{
    Atom, AtomEnum, ClientMessageEvent, ConfigureWindowAux, ConnectionExt as _, EventMask, GrabMode,
    ImageFormat, InputFocus, MapState, ModMask, StackMode, Window,
};
use x11rb::protocol::xtest::ConnectionExt as _;
use x11rb::protocol::Event;
use x11rb::rust_connection::RustConnection;

fn err(e: impl std::fmt::Display) -> String {
    format!("X11: {e}")
}

pub struct X {
    conn: RustConnection,
    root: Window,
    screen_w: u16,
    screen_h: u16,
    atoms: HashMap<&'static str, Atom>,
    keycodes: HashMap<u32, u8>,
}

/// A connection to the X server named by `DISPLAY`.
pub fn open() -> Res<Box<dyn Desktop>> {
    Ok(Box::new(X::connect()?))
}

impl X {
    fn connect() -> Res<X> {
        if std::env::var_os("DISPLAY").is_none() {
            return Err(if std::env::var_os("WAYLAND_DISPLAY").is_some() {
                "This is a Wayland session without X11: real games need an X11 session (or the game under XWayland with DISPLAY set)".into()
            } else {
                "No X11 display (DISPLAY is not set)".into()
            });
        }
        let (conn, n) = x11rb::connect(None).map_err(err)?;
        let screen = &conn.setup().roots[n];
        let (root, screen_w, screen_h) = (screen.root, screen.width_in_pixels, screen.height_in_pixels);
        let mut x = X { conn, root, screen_w, screen_h, atoms: HashMap::new(), keycodes: HashMap::new() };
        x.conn
            .xtest_get_version(2, 2)
            .map_err(err)?
            .reply()
            .map_err(|_| "The X server has no XTEST extension".to_string())?;
        x.keycodes = keycodes(&x.conn)?;
        Ok(x)
    }

    fn atom(&mut self, name: &'static str) -> Res<Atom> {
        if let Some(a) = self.atoms.get(name) {
            return Ok(*a);
        }
        let a = self.conn.intern_atom(false, name.as_bytes()).map_err(err)?.reply().map_err(err)?.atom;
        self.atoms.insert(name, a);
        Ok(a)
    }

    fn property(&mut self, window: Window, name: &'static str) -> Option<(Atom, Vec<u8>, u8)> {
        let atom = self.atom(name).ok()?;
        let r = self.conn.get_property(false, window, atom, AtomEnum::ANY, 0, 1 << 20).ok()?.reply().ok()?;
        (r.type_ != 0).then_some((r.type_, r.value, r.format))
    }

    fn windows_property(&mut self, window: Window, name: &'static str) -> Option<Vec<u32>> {
        let (_, value, format) = self.property(window, name)?;
        (format == 32).then(|| value.as_chunks::<4>().0.iter().map(|c| u32::from_ne_bytes(*c)).collect())
    }

    fn title(&mut self, window: Window) -> String {
        for name in ["_NET_WM_NAME", "WM_NAME"] {
            if let Some((_, v, _)) = self.property(window, name) {
                let s = String::from_utf8_lossy(&v).trim().to_string();
                if !s.is_empty() {
                    return s;
                }
            }
        }
        String::new()
    }

    fn pid(&mut self, window: Window) -> u32 {
        self.windows_property(window, "_NET_WM_PID").and_then(|v| v.first().copied()).unwrap_or(0)
    }

    fn parent(&self, window: Window) -> Option<Window> {
        let r = self.conn.query_tree(window).ok()?.reply().ok()?;
        (r.parent != 0 && r.parent != window).then_some(r.parent)
    }

    fn keycode(&self, keysym: u32) -> Res<u8> {
        self.keycodes
            .get(&keysym)
            .copied()
            .ok_or_else(|| format!("No key on this keyboard makes keysym {keysym:#x}"))
    }

    fn fake(&self, kind: u8, detail: u8, x: i16, y: i16) -> Res<()> {
        let root = if kind == x11rb::protocol::xproto::MOTION_NOTIFY_EVENT { self.root } else { x11rb::NONE };
        self.conn.xtest_fake_input(kind, detail, x11rb::CURRENT_TIME, root, x, y, 0).map_err(err)?;
        self.conn.flush().map_err(err)
    }
}

/// keysym -> the first keycode that makes it.
fn keycodes(conn: &RustConnection) -> Res<HashMap<u32, u8>> {
    let setup = conn.setup();
    let (min, max) = (setup.min_keycode, setup.max_keycode);
    let r = conn.get_keyboard_mapping(min, max - min + 1).map_err(err)?.reply().map_err(err)?;
    let per = r.keysyms_per_keycode.max(1) as usize;
    let mut map = HashMap::new();
    // Prefer unshifted keysyms: columns in order, lower codes first.
    for col in 0..per {
        for (i, chunk) in r.keysyms.chunks(per).enumerate() {
            if let Some(&sym) = chunk.get(col) {
                if sym != 0 {
                    map.entry(sym).or_insert(min + i as u8);
                }
            }
        }
    }
    Ok(map)
}

impl Desktop for X {
    fn windows(&mut self) -> Res<Vec<WindowInfo>> {
        let candidates = match self.windows_property(self.root, "_NET_CLIENT_LIST") {
            Some(list) => list,
            None => self.conn.query_tree(self.root).map_err(err)?.reply().map_err(err)?.children,
        };
        let me = std::process::id();
        let mut out = Vec::new();
        for w in candidates {
            let Ok(Ok(attrs)) = self.conn.get_window_attributes(w).map(|c| c.reply()) else { continue };
            if attrs.map_state != MapState::VIEWABLE || attrs.override_redirect {
                continue;
            }
            let title = self.title(w);
            let pid = self.pid(w);
            if title.is_empty() || pid == me {
                continue;
            }
            let Ok(Ok(g)) = self.conn.get_geometry(w).map(|c| c.reply()) else { continue };
            let process = std::fs::read_to_string(format!("/proc/{pid}/comm"))
                .map(|s| s.trim().to_string())
                .unwrap_or_default();
            out.push(WindowInfo {
                id: w as u64,
                title,
                process,
                pid,
                width: g.width as u32,
                height: g.height as u32,
            });
        }
        Ok(out)
    }

    fn in_front(&mut self, window: u64) -> bool {
        let window = window as Window;
        if let Some(active) =
            self.windows_property(self.root, "_NET_ACTIVE_WINDOW").and_then(|v| v.first().copied())
        {
            if active != 0 {
                return active == window;
            }
        }
        let Ok(Ok(focus)) = self.conn.get_input_focus().map(|c| c.reply()) else { return false };
        let mut w = focus.focus;
        for _ in 0..32 {
            if w == window {
                return true;
            }
            match self.parent(w) {
                Some(p) if p != self.root => w = p,
                _ => return false,
            }
        }
        false
    }

    fn bring_to_front(&mut self, window: u64) -> Res<()> {
        let window = window as Window;
        let active = self.atom("_NET_ACTIVE_WINDOW")?;
        // Through the window manager when there is one (source 2: a pager, i.e. the user's choice)...
        let msg = ClientMessageEvent::new(32, window, active, [2, x11rb::CURRENT_TIME, 0, 0, 0]);
        self.conn
            .send_event(
                false,
                self.root,
                EventMask::SUBSTRUCTURE_REDIRECT | EventMask::SUBSTRUCTURE_NOTIFY,
                msg,
            )
            .map_err(err)?;
        // ...and directly, for X servers without one.
        self.conn
            .configure_window(window, &ConfigureWindowAux::new().stack_mode(StackMode::ABOVE))
            .map_err(err)?;
        self.conn.set_input_focus(InputFocus::PARENT, window, x11rb::CURRENT_TIME).map_err(err)?;
        // A round trip, so the window has moved before this returns.
        self.conn.get_input_focus().map_err(err)?.reply().map_err(err)?;
        Ok(())
    }

    fn area(&mut self, window: u64) -> Res<Rect> {
        let w = window as Window;
        let g =
            self.conn.get_geometry(w).map_err(err)?.reply().map_err(|_| "The window is gone".to_string())?;
        let t = self.conn.translate_coordinates(w, self.root, 0, 0).map_err(err)?.reply().map_err(err)?;
        Ok(Rect { x: t.dst_x as i32, y: t.dst_y as i32, width: g.width as u32, height: g.height as u32 })
    }

    fn key(&mut self, key: &KeyInfo, down: bool) -> Res<()> {
        let code = self.keycode(key.keysym)?;
        let kind = if down {
            x11rb::protocol::xproto::KEY_PRESS_EVENT
        } else {
            x11rb::protocol::xproto::KEY_RELEASE_EVENT
        };
        self.fake(kind, code, 0, 0)
    }

    fn button(&mut self, button: Button, down: bool) -> Res<()> {
        let detail = match button {
            Button::Left => 1,
            Button::Middle => 2,
            Button::Right => 3,
        };
        let kind = if down {
            x11rb::protocol::xproto::BUTTON_PRESS_EVENT
        } else {
            x11rb::protocol::xproto::BUTTON_RELEASE_EVENT
        };
        self.fake(kind, detail, 0, 0)
    }

    fn motion(&mut self, dx: i32, dy: i32) -> Res<()> {
        let c = |v: i32| v.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        // detail 1: relative movement.
        self.fake(x11rb::protocol::xproto::MOTION_NOTIFY_EVENT, 1, c(dx), c(dy))
    }

    fn grab(&mut self, window: u64, width: u32, height: u32) -> Res<Vec<u8>> {
        let area = self.area(window)?;
        let (w, h) = (area.width.min(u16::MAX as u32) as u16, area.height.min(u16::MAX as u32) as u16);
        // The window's own picture; where that fails (partly off screen), what the screen shows.
        let image = match self
            .conn
            .get_image(ImageFormat::Z_PIXMAP, window as Window, 0, 0, w, h, !0)
            .map(|c| c.reply())
        {
            Ok(Ok(i)) => (i, w as usize, h as usize),
            _ => {
                let x0 = area.x.clamp(0, self.screen_w as i32);
                let y0 = area.y.clamp(0, self.screen_h as i32);
                let x1 = (area.x + area.width as i32).clamp(0, self.screen_w as i32);
                let y1 = (area.y + area.height as i32).clamp(0, self.screen_h as i32);
                let (cw, ch) = ((x1 - x0).max(1) as u16, (y1 - y0).max(1) as u16);
                let i = self
                    .conn
                    .get_image(ImageFormat::Z_PIXMAP, self.root, x0 as i16, y0 as i16, cw, ch, !0)
                    .map_err(err)?
                    .reply()
                    .map_err(err)?;
                (i, cw as usize, ch as usize)
            }
        };
        let (img, sw, sh) = image;
        let bpp = self
            .conn
            .setup()
            .pixmap_formats
            .iter()
            .find(|f| f.depth == img.depth)
            .map(|f| f.bits_per_pixel)
            .unwrap_or(32);
        if bpp != 32 || img.data.len() < sw * sh * 4 {
            return Err(format!("Unsupported picture format ({} bits a pixel)", bpp));
        }
        let stride = img.data.len() / sh;
        let data = &img.data;
        // 32 bits a pixel, little-endian: blue, green, red, unused.
        Ok(shrink(sw, sh, width as usize, height as usize, |x, y| {
            let at = y * stride + x * 4;
            [data[at + 2], data[at + 1], data[at]]
        }))
    }
}

/// Watches for the emergency stop keys (Ctrl+Alt+End) everywhere on the X display, on its own
/// connection and thread. The keys are grabbed, so the game never sees them.
pub fn stop_key_watch(on_stop: Box<dyn Fn() + Send>) -> Res<()> {
    let x = X::connect()?;
    let end = x.keycode(0xFF57)?;
    let base = ModMask::CONTROL | ModMask::M1;
    for extra in [ModMask::from(0u16), ModMask::LOCK, ModMask::M2, ModMask::LOCK | ModMask::M2] {
        x.conn
            .grab_key(false, x.root, base | extra, end, GrabMode::ASYNC, GrabMode::ASYNC)
            .map_err(err)?
            .check()
            .map_err(|_| "Another program already uses Ctrl+Alt+End".to_string())?;
    }
    x.conn.flush().map_err(err)?;
    std::thread::Builder::new()
        .name("emergency stop keys".into())
        .spawn(move || loop {
            match x.conn.wait_for_event() {
                Ok(Event::KeyPress(_)) => on_stop(),
                Ok(_) => {}
                Err(_) => break,
            }
        })
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Listens to the keyboard and mouse through XInput2 raw events (what the devices report,
/// before any window or grab): keys from the controls table, the three buttons, and relative
/// movement. Input from XTEST (programs, Tools included) is skipped unless `accept_virtual`,
/// which only tests use (they play the person's part through XTEST).
pub fn listen(accept_virtual: bool, tx: std::sync::mpsc::Sender<super::Heard>) -> Res<super::Listening> {
    use super::Input;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use x11rb::protocol::xinput::{self, ConnectionExt as _};

    let x = X::connect()?;
    x.conn
        .xinput_xi_query_version(2, 2)
        .map_err(err)?
        .reply()
        .map_err(|_| "The X server has no XInput 2 (needed to record input)".to_string())?;
    // Which devices are XTEST's (virtual input from programs).
    let devices = x.conn.xinput_xi_query_device(xinput::Device::ALL).map_err(err)?.reply().map_err(err)?;
    let virtual_ids: Vec<u16> = devices
        .infos
        .iter()
        .filter(|d| String::from_utf8_lossy(&d.name).contains("XTEST"))
        .map(|d| d.deviceid)
        .collect();
    // keycode -> key name, for the keys in the table (right-hand modifiers count as their key).
    let mut names: HashMap<u8, &'static str> = HashMap::new();
    for k in crate::live::controls::KEYS {
        if let Ok(code) = x.keycode(k.keysym) {
            names.entry(code).or_insert(k.name);
        }
    }
    for (sym, name) in [(0xFFE2, "shift"), (0xFFE4, "ctrl"), (0xFFEA, "alt"), (0xFE03, "alt")] {
        if let Ok(code) = x.keycode(sym) {
            names.entry(code).or_insert(name);
        }
    }
    let mask = xinput::XIEventMask::RAW_KEY_PRESS
        | xinput::XIEventMask::RAW_KEY_RELEASE
        | xinput::XIEventMask::RAW_BUTTON_PRESS
        | xinput::XIEventMask::RAW_BUTTON_RELEASE
        | xinput::XIEventMask::RAW_MOTION;
    x.conn
        .xinput_xi_select_events(
            x.root,
            &[xinput::EventMask { deviceid: xinput::Device::ALL_MASTER.into(), mask: vec![mask] }],
        )
        .map_err(err)?
        .check()
        .map_err(err)?;
    x.conn.flush().map_err(err)?;
    let stop = Arc::new(AtomicBool::new(false));
    let stopping = stop.clone();
    let thread = std::thread::Builder::new()
        .name("input listener".into())
        .spawn(move || {
            let fp = |v: &xinput::Fp3232| v.integral as f64 + v.frac as f64 / 4_294_967_296.0;
            while !stopping.load(Ordering::Acquire) {
                let event = match x.conn.poll_for_event() {
                    Ok(Some(e)) => e,
                    Ok(None) => {
                        std::thread::sleep(Duration::from_millis(1));
                        continue;
                    }
                    Err(_) => break,
                };
                let now = Instant::now();
                let heard = match event {
                    Event::XinputRawKeyPress(e) | Event::XinputRawKeyRelease(e) => {
                        if !accept_virtual && virtual_ids.contains(&e.sourceid) {
                            continue;
                        }
                        let Some(&key) = names.get(&(e.detail as u8)) else { continue };
                        Input::Key { key, down: e.event_type == xinput::RAW_KEY_PRESS_EVENT }
                    }
                    Event::XinputRawButtonPress(e) | Event::XinputRawButtonRelease(e) => {
                        if !accept_virtual && virtual_ids.contains(&e.sourceid) {
                            continue;
                        }
                        let button = match e.detail {
                            1 => Button::Left,
                            2 => Button::Middle,
                            3 => Button::Right,
                            _ => continue,
                        };
                        Input::Button { button, down: e.event_type == xinput::RAW_BUTTON_PRESS_EVENT }
                    }
                    Event::XinputRawMotion(e) => {
                        if !accept_virtual && virtual_ids.contains(&e.sourceid) {
                            continue;
                        }
                        // The values present, in axis order; axes 0 and 1 are x and y.
                        let bits = e.valuator_mask.first().copied().unwrap_or(0);
                        let mut values = e.axisvalues_raw.iter();
                        let (mut dx, mut dy) = (0.0, 0.0);
                        for axis in 0..32 {
                            if bits & (1 << axis) == 0 {
                                continue;
                            }
                            let Some(v) = values.next() else { break };
                            match axis {
                                0 => dx = fp(v),
                                1 => dy = fp(v),
                                _ => {}
                            }
                        }
                        if dx == 0.0 && dy == 0.0 {
                            continue;
                        }
                        Input::Motion { dx, dy }
                    }
                    _ => continue,
                };
                if tx.send((now, heard)).is_err() {
                    break;
                }
            }
        })
        .map_err(|e| e.to_string())?;
    Ok(super::Listening::new(stop, None, thread))
}
