fn main() {
    // The Engine's version, shown in the Hub's sidebar and Settings.
    let manifest = "../../../crates/yendory-engine/Cargo.toml";
    println!("cargo:rerun-if-changed={manifest}");
    let text = std::fs::read_to_string(manifest).expect("the Engine's Cargo.toml");
    let version = text
        .lines()
        .find_map(|l| l.strip_prefix("version = \""))
        .and_then(|v| v.strip_suffix('"'))
        .expect("the Engine's version");
    println!("cargo:rustc-env=YENDORY_ENGINE_VERSION={version}");
    tauri_build::build()
}
