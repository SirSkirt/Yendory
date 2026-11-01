fn main() {
    // The Engine's version, reported by `hello` and `--version`.
    let manifest = "../engine/Cargo.toml";
    println!("cargo:rerun-if-changed={manifest}");
    let text = std::fs::read_to_string(manifest).expect("the Engine's Cargo.toml");
    let version = text
        .lines()
        .find_map(|l| l.strip_prefix("version = \""))
        .and_then(|v| v.strip_suffix('"'))
        .expect("the Engine's version");
    println!("cargo:rustc-env=YENDORY_ENGINE_VERSION={version}");
}
