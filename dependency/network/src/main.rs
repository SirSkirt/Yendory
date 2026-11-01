// No console window on Windows: the Network runs in the background.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--version") {
        // What the Hub reads after installing, to check the right program arrived.
        println!("{}", yendory_network::version_json());
        return;
    }
    let idle = std::env::var("YENDORY_NETWORK_IDLE").ok().and_then(|s| s.parse().ok()).unwrap_or(300);
    if let Err(e) = yendory_network::serve(yendory_network::Options { idle_seconds: idle }) {
        yendory_network::log(&format!("stopped: {e}"));
        std::process::exit(1);
    }
}
