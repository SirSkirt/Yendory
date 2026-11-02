//! Real input and capture, end to end, on the test range: Tools starts the range (its own
//! process and window), finds its window, sends input through the safety layer, and checks what
//! the range says it received. Needs an X11 display (CI runs it under Xvfb); without one it
//! says so and passes.
#![cfg(target_os = "linux")]
use serde_json::Value;
use std::sync::atomic::Ordering;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use yendory_tools_lib::live::controls::Control;
use yendory_tools_lib::live::profile::range_profile;
use yendory_tools_lib::live::safety::{Controller, STOPS};
use yendory_tools_lib::live::{script, sys, RangeProcess};

const EXE: &str = env!("CARGO_BIN_EXE_yendory-tools");

/// The tests share one display: one at a time.
static DISPLAY: Mutex<()> = Mutex::new(());

/// Starts the range with its error output kept, for the failure message.
fn start_range(seed: u64) -> RangeProcess {
    let log = std::env::temp_dir().join(format!("tools-range-{}-{seed}.log", std::process::id()));
    std::env::set_var("YENDORY_RANGE_LOG", &log);
    RangeProcess::start(EXE.as_ref(), seed, |_| {}).unwrap()
}

fn wait_for_window(range: &mut RangeProcess, seed: u64) -> u64 {
    let profile = range_profile();
    let pid = range.pid();
    let deadline = Instant::now() + Duration::from_secs(40);
    loop {
        let all = sys::open().unwrap().windows().unwrap();
        if let Some(w) = all.iter().find(|w| profile.matches(w) && w.pid == pid) {
            return w.id;
        }
        if Instant::now() > deadline || !range.running() {
            let log = std::env::temp_dir().join(format!("tools-range-{}-{seed}.log", std::process::id()));
            panic!(
                "the range's window did not appear (range running: {}; windows seen: {:?}); the range said:\n{}",
                range.running(),
                all.iter().map(|w| (&w.title, w.pid, &w.process)).collect::<Vec<_>>(),
                std::fs::read_to_string(log).unwrap_or_default()
            );
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn front(window: u64) {
    let mut d = sys::open().unwrap();
    d.bring_to_front(window).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !d.in_front(window) {
        assert!(Instant::now() < deadline, "the window did not come to the front");
        std::thread::sleep(Duration::from_millis(30));
    }
    // The range centres the cursor when it gets the focus.
    std::thread::sleep(Duration::from_millis(300));
}

fn num(v: &Value, k: &str) -> f64 {
    v[k].as_f64().unwrap_or_else(|| panic!("{k} in {v}"))
}

#[test]
fn the_range_receives_exactly_what_the_safety_layer_sends() {
    if std::env::var_os("DISPLAY").is_none() {
        eprintln!("skipped: no X11 display (run under xvfb-run)");
        return;
    }
    let _one = DISPLAY.lock().unwrap_or_else(|e| e.into_inner());
    let mut range = start_range(5);
    let window = wait_for_window(&mut range, 5);
    front(window);

    // The picture: not blank, and it changes when the view turns.
    let mut desk = sys::open().unwrap();
    let before = desk.grab(window, 128, 72).unwrap();
    assert_eq!(before.len(), 128 * 72 * 3);
    assert!(before.iter().any(|&b| b != before[0]), "the picture is blank");

    let p = range_profile();
    let c =
        Mutex::new(Controller::new(sys::open().unwrap(), window, p.permitted().unwrap(), p.limits.clone()));
    let s0 = range.state().unwrap();
    assert_eq!(s0["focused"], true);

    script::run(
        &c,
        &script::parse("hold w 300; move 100 0; wait 100; click left; hold w+shift 200; press e; tap 1")
            .unwrap(),
    );
    std::thread::sleep(Duration::from_millis(200));
    let s1 = range.state().unwrap();
    // W, then the click, then W and Shift: four presses arrived; E and 1 are not permitted.
    assert_eq!(num(&s1, "presses") - num(&s0, "presses"), 4.0, "{s1}");
    assert_eq!(num(&s1, "mouseX") - num(&s0, "mouseX"), 100.0, "{s1}");
    let turned = num(&s1, "angle") - num(&s0, "angle");
    let want = 100.0 * yendory_tools_lib::range::TURN_PER_UNIT as f64;
    assert!((turned.rem_euclid(std::f64::consts::TAU) - want).abs() < 1e-3, "turned {turned}");
    assert_eq!(num(&s1, "shots") - num(&s0, "shots"), 1.0);
    let moved = (num(&s1, "x") - num(&s0, "x")).hypot(num(&s1, "y") - num(&s0, "y"));
    assert!(moved > 0.3, "moved {moved}");
    // Nothing is left held, in Tools or in the game.
    assert!(c.lock().unwrap().held().is_empty());
    assert_eq!(s1["held"]["w"], false);
    let refused: Vec<String> =
        c.lock().unwrap().log.iter().filter(|e| !e.sent).map(|e| e.action.clone()).collect();
    assert_eq!(refused, ["press e", "press 1"]);

    let after = desk.grab(window, 128, 72).unwrap();
    assert_ne!(before, after, "the picture did not change after turning");

    // Another window comes to the front: everything is released and input pauses.
    assert!(c.lock().unwrap().press(Control::Key("w")));
    let mut other = start_range(6);
    let other_window = wait_for_window(&mut other, 6);
    front(other_window);
    c.lock().unwrap().tick();
    assert!(c.lock().unwrap().held().is_empty());
    assert!(c.lock().unwrap().paused.is_some());
    assert!(!c.lock().unwrap().press(Control::Key("w")));
    assert_eq!(other.state().unwrap()["presses"], 0, "the other window got input");
    drop(other);

    // The emergency stop keys, pressed on the keyboard (here: XTEST), stop everything.
    front(window);
    let stop_pressed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = stop_pressed.clone();
    sys::stop_key_watch(Box::new(move || {
        STOPS.fetch_add(1, Ordering::AcqRel);
        flag.store(true, Ordering::Release);
    }))
    .unwrap();
    let c =
        Mutex::new(Controller::new(sys::open().unwrap(), window, p.permitted().unwrap(), p.limits.clone()));
    assert!(c.lock().unwrap().press(Control::Key("w")));
    let mut keyboard = sys::open().unwrap();
    let k = |n| yendory_tools_lib::live::controls::key(n).unwrap();
    for (n, down) in
        [("ctrl", true), ("alt", true), ("end", true), ("end", false), ("alt", false), ("ctrl", false)]
    {
        keyboard.key(k(n), down).unwrap();
    }
    let deadline = Instant::now() + Duration::from_secs(3);
    while !stop_pressed.load(Ordering::Acquire) {
        assert!(Instant::now() < deadline, "the stop keys were not seen");
        std::thread::sleep(Duration::from_millis(20));
    }
    c.lock().unwrap().tick();
    assert!(c.lock().unwrap().stopped());
    assert!(c.lock().unwrap().held().is_empty());
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(range.state().unwrap()["held"]["w"], false);
}

#[test]
fn a_person_playing_the_range_is_recorded_in_sync() {
    use yendory_tools_lib::live::record::{record, Setup};
    use yendory_tools_lib::live::recording::{open, validate, Split};
    if std::env::var_os("DISPLAY").is_none() {
        eprintln!("skipped: no X11 display (run under xvfb-run)");
        return;
    }
    let _one = DISPLAY.lock().unwrap_or_else(|e| e.into_inner());
    let mut range = start_range(9);
    let window = wait_for_window(&mut range, 9);
    front(window);
    let k = |n| yendory_tools_lib::live::controls::key(n).unwrap();

    // Tools' own listener never hears virtual input (XTEST): scripted actions can't become a
    // person's demonstration.
    let (tx, rx) = std::sync::mpsc::channel();
    let listening = sys::listen(false, tx).unwrap();
    let mut hands = sys::open().unwrap();
    hands.key(k("w"), true).unwrap();
    hands.key(k("w"), false).unwrap();
    hands.motion(10, 0).unwrap();
    std::thread::sleep(Duration::from_millis(200));
    drop(listening);
    assert_eq!(rx.try_iter().count(), 0, "virtual input was heard as a person's");

    // The person (played through XTEST, which only tests may listen to).
    let (tx, rx) = std::sync::mpsc::channel();
    let listening = sys::listen(true, tx).unwrap();
    let data = std::env::temp_dir().join(format!("tools-live-rec-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data);
    let mut profile = range_profile();
    profile.capture.fps = 20;
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop2 = stop.clone();
    let person = std::thread::spawn(move || {
        let mut hands = sys::open().unwrap();
        let k = |n| yendory_tools_lib::live::controls::key(n).unwrap();
        std::thread::sleep(Duration::from_millis(300));
        hands.key(k("w"), true).unwrap();
        std::thread::sleep(Duration::from_millis(500));
        hands.key(k("w"), false).unwrap();
        for _ in 0..5 {
            hands.motion(20, 0).unwrap();
            std::thread::sleep(Duration::from_millis(30));
        }
        hands.button(yendory_tools_lib::live::controls::Button::Left, true).unwrap();
        std::thread::sleep(Duration::from_millis(40));
        hands.button(yendory_tools_lib::live::controls::Button::Left, false).unwrap();
        // A key outside the profile.
        hands.key(k("e"), true).unwrap();
        hands.key(k("e"), false).unwrap();
        std::thread::sleep(Duration::from_millis(300));
        stop2.store(true, Ordering::Release);
    });
    let mut desk = sys::open().unwrap();
    let setup = Setup {
        data: &data,
        profile: &profile,
        window,
        split: Split::Train,
        notes: String::new(),
        room: u64::MAX,
    };
    let info = record(setup, desk.as_mut(), rx, &stop, |_| {}).unwrap();
    person.join().unwrap();
    drop(listening);

    let v = validate(&data, &profile.id, &info.id).unwrap();
    assert!(v.ok, "{v:?}");
    let count = |n: &str| v.active_ticks.iter().find(|x| x.0 == n).map(|x| x.1).unwrap_or(0);
    // 20 pictures a second: W held 500 ms is about 10 ticks.
    assert!((9..=12).contains(&count("w")), "{v:?}");
    assert!((1..=2).contains(&count("left")), "{v:?}");
    assert_eq!(v.mouse_x, 100.0, "{v:?}");
    assert!(v.fps > 15.0, "{v:?}");
    assert!((info.duration_ms as f64 / 1000.0 - info.ticks as f64 / 20.0).abs() < 0.3, "{info:?}");
    let mut r = open(&data, &profile.id, &info.id).unwrap();
    let first = r.frame(0).unwrap();
    let last = r.frame(r.ticks.len() - 1).unwrap();
    assert_eq!(first.len(), (profile.capture.width * profile.capture.height * 3) as usize);
    assert!(first.iter().any(|&b| b != first[0]), "the first picture is blank");
    assert_ne!(first, last, "walking and turning did not change the picture");
    let events = std::fs::read_to_string(r.events_path()).unwrap();
    assert!(!events.contains("\"e\""));
    let _ = std::fs::remove_dir_all(&data);
}
