// Prevents an extra console window on Windows in release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // The test range runs as its own process from the same program (docs/GAME_TRAINING.md).
    if args.iter().any(|a| a == "--range") {
        return yendory_tools_lib::range::app::run(&args);
    }
    yendory_tools_lib::run()
}
