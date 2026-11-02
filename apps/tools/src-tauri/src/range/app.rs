//! The range's window (`yendory-tools --range [--seed N] [--seconds S]`): reads the keyboard and
//! mouse like a game, draws 60 times a second, and reports the truth as JSON lines on standard
//! output (`ready`, `shot`, `episode`, and `state` when asked on standard input).
//!
//! Keys: raw key presses. Mouse: on Windows the raw movement (as games read it); on Linux the
//! cursor's movement from the window's centre, put back in the centre after every move.
use super::render::{self, HEIGHT, WIDTH};
use super::{Event as RangeEvent, Held, Range};
use std::cell::RefCell;
use std::io::{BufRead, Write};
use std::rc::Rc;
use std::time::{Duration, Instant};
use tao::dpi::LogicalSize;
use tao::event::{ElementState, Event, MouseButton, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoopBuilder};
use tao::keyboard::KeyCode;
use tao::window::{Window, WindowBuilder};

pub const TITLE: &str = "Yendory Range";

enum Command {
    State,
    Reset(u64),
    Quit,
}

fn say(value: &impl serde::Serialize) {
    let mut out = std::io::stdout().lock();
    let _ = serde_json::to_writer(&mut out, value);
    let _ = out.write_all(b"\n");
    let _ = out.flush();
}

fn argument(args: &[String], name: &str) -> Option<u64> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok())
}

/// Runs the range until its window closes or `quit` arrives.
pub fn run(args: &[String]) {
    let seed = argument(args, "--seed").unwrap_or(1);
    let seconds = argument(args, "--seconds").unwrap_or(60).clamp(5, 3600);
    let event_loop = EventLoopBuilder::<Command>::with_user_event().build();
    let proxy = event_loop.create_proxy();
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines() {
            let Ok(line) = line else { break };
            let mut words = line.split_whitespace();
            let cmd = match words.next() {
                Some("state") => Command::State,
                Some("reset") => Command::Reset(words.next().and_then(|s| s.parse().ok()).unwrap_or(1)),
                Some("quit") => Command::Quit,
                _ => continue,
            };
            if proxy.send_event(cmd).is_err() {
                break;
            }
        }
        // Tools closed (its end of the pipe is gone): the range closes too.
        let _ = proxy.send_event(Command::Quit);
    });
    let window = Rc::new(
        WindowBuilder::new()
            .with_title(TITLE)
            .with_inner_size(LogicalSize::new(WIDTH as f64 * 2.0, HEIGHT as f64 * 2.0))
            .build(&event_loop)
            .expect("the range's window"),
    );
    let pixels = Rc::new(RefCell::new(vec![0u32; WIDTH * HEIGHT]));
    let mut screen = Screen::new(&window, pixels.clone());
    let mut range = Range::new(seed, seconds * 1000);
    let mut held = Held::default();
    let mut last = Instant::now();
    let mut over_since: Option<Instant> = None;
    say(&serde_json::json!({"ev": "ready", "title": TITLE, "pid": std::process::id(), "seed": seed}));

    event_loop.run(move |event, _, flow| {
        let mut events = Vec::new();
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *flow = ControlFlow::Exit,
                WindowEvent::Focused(f) => {
                    range.focused = f;
                    // Like a game: losing focus lets go of everything.
                    if !f {
                        held = Held::default();
                        range.set(held, &mut events);
                    }
                    let _ = window.set_cursor_grab(f);
                    window.set_cursor_visible(!f);
                    if f {
                        centre(&window);
                    }
                }
                WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                    held.fire = state == ElementState::Pressed;
                    range.set(held, &mut events);
                }
                #[cfg(not(windows))]
                WindowEvent::CursorMoved { position, .. } if range.focused => {
                    let size = window.inner_size();
                    let (cx, cy) = ((size.width / 2) as f64, (size.height / 2) as f64);
                    let (dx, dy) = (position.x - cx, position.y - cy);
                    if dx != 0.0 || dy != 0.0 {
                        range.mouse(dx, dy);
                        centre(&window);
                    }
                }
                _ => {}
            },
            #[cfg(windows)]
            Event::DeviceEvent { event: tao::event::DeviceEvent::MouseMotion { delta, .. }, .. } => {
                if range.focused {
                    range.mouse(delta.0, delta.1);
                }
            }
            // Keys as games read them: raw key presses (Windows raw input, X11 XInput2), only
            // while the range has the focus. (On Linux, tao's ordinary key events drop keys
            // pressed while Shift is held.)
            Event::DeviceEvent { event: tao::event::DeviceEvent::Key(key), .. } if range.focused => {
                let down = key.state == ElementState::Pressed;
                match key.physical_key {
                    KeyCode::KeyW => held.w = down,
                    KeyCode::KeyA => held.a = down,
                    KeyCode::KeyS => held.s = down,
                    KeyCode::KeyD => held.d = down,
                    KeyCode::ShiftLeft | KeyCode::ShiftRight => held.shift = down,
                    _ => {}
                }
                range.set(held, &mut events);
            }
            Event::UserEvent(cmd) => match cmd {
                Command::State => say(&serde_json::json!({"ev": "state", "state": range.state()})),
                Command::Reset(s) => {
                    range.reset(s);
                    over_since = None;
                    say(&serde_json::json!({"ev": "reset", "seed": s}));
                }
                Command::Quit => *flow = ControlFlow::Exit,
            },
            Event::NewEvents(_) => {
                let now = Instant::now();
                let ms = now.duration_since(last).as_millis() as u64;
                if ms >= 16 {
                    last = now;
                    if let Some(end) = range.step(ms) {
                        events.push(end);
                        over_since = Some(now);
                    }
                    // A person playing gets the next layout after a short pause.
                    if over_since.is_some_and(|t| now.duration_since(t) > Duration::from_secs(3)) {
                        let next = range.seed + 1;
                        range.reset(next);
                        over_since = None;
                        say(&serde_json::json!({"ev": "reset", "seed": next}));
                    }
                    render::draw(&range, &mut pixels.borrow_mut());
                    screen.show();
                }
                *flow = ControlFlow::WaitUntil(last + Duration::from_millis(16));
            }
            _ => {}
        }
        for e in events {
            let e: &RangeEvent = &e;
            say(e);
        }
    });
}

fn centre(window: &Window) {
    let size = window.inner_size();
    let _ = window.set_cursor_position(tao::dpi::PhysicalPosition::new(size.width / 2, size.height / 2));
}

/// Shows the pixels in the window, scaled to fit.
#[cfg(not(windows))]
struct Screen {
    area: gtk::DrawingArea,
}

#[cfg(not(windows))]
impl Screen {
    fn new(window: &Rc<Window>, pixels: Rc<RefCell<Vec<u32>>>) -> Screen {
        use gtk::prelude::*;
        use tao::platform::unix::WindowExtUnix;
        let area = gtk::DrawingArea::new();
        area.set_hexpand(true);
        area.set_vexpand(true);
        area.connect_draw(move |area, cr| {
            let Ok(mut surface) =
                gtk::cairo::ImageSurface::create(gtk::cairo::Format::Rgb24, WIDTH as i32, HEIGHT as i32)
            else {
                return gtk::glib::Propagation::Stop;
            };
            let stride = surface.stride() as usize;
            if let Ok(mut data) = surface.data() {
                let px = pixels.borrow();
                for y in 0..HEIGHT {
                    for x in 0..WIDTH {
                        let at = y * stride + x * 4;
                        data[at..at + 4].copy_from_slice(&px[y * WIDTH + x].to_ne_bytes());
                    }
                }
            }
            let (w, h) = (area.allocated_width() as f64, area.allocated_height() as f64);
            cr.scale(w / WIDTH as f64, h / HEIGHT as f64);
            let _ = cr.set_source_surface(&surface, 0.0, 0.0);
            cr.source().set_filter(gtk::cairo::Filter::Nearest);
            let _ = cr.paint();
            gtk::glib::Propagation::Stop
        });
        if let Some(vbox) = window.default_vbox() {
            vbox.pack_start(&area, true, true, 0);
        }
        area.show_all();
        Screen { area }
    }

    fn show(&mut self) {
        use gtk::prelude::*;
        self.area.queue_draw();
    }
}

#[cfg(windows)]
struct Screen {
    surface: softbuffer::Surface<Rc<Window>, Rc<Window>>,
    window: Rc<Window>,
    pixels: Rc<RefCell<Vec<u32>>>,
}

#[cfg(windows)]
impl Screen {
    fn new(window: &Rc<Window>, pixels: Rc<RefCell<Vec<u32>>>) -> Screen {
        let window = window.clone();
        let context = softbuffer::Context::new(window.clone()).expect("a drawing context");
        let surface = softbuffer::Surface::new(&context, window.clone()).expect("a drawing surface");
        Screen { surface, window, pixels }
    }

    fn show(&mut self) {
        use std::num::NonZeroU32;
        let size = self.window.inner_size();
        let (Some(w), Some(h)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height)) else { return };
        if self.surface.resize(w, h).is_err() {
            return;
        }
        let Ok(mut buf) = self.surface.buffer_mut() else { return };
        let px = self.pixels.borrow();
        let (w, h) = (w.get() as usize, h.get() as usize);
        for y in 0..h {
            let sy = y * HEIGHT / h;
            for x in 0..w {
                buf[y * w + x] = px[sy * WIDTH + x * WIDTH / w];
            }
        }
        let _ = buf.present();
    }
}
