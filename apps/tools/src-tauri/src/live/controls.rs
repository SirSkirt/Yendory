//! The controls a network can be given: keyboard keys and mouse buttons, by name, with their
//! codes on each system. Keys that only reach the system (the Windows key, the menu key, Print
//! Screen, media keys) are not in the table at all, so no profile can permit them.
use serde::Serialize;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KeyInfo {
    /// The name in profiles and scripts (lower case).
    pub name: &'static str,
    /// The name shown in the window.
    pub label: &'static str,
    /// Windows: the set-1 scan code, and whether it needs the extended (E0) prefix.
    #[serde(skip)]
    pub scan: u16,
    #[serde(skip)]
    pub extended: bool,
    /// X11: the keysym.
    #[serde(skip)]
    pub keysym: u32,
    pub group: &'static str,
}

const fn k(
    name: &'static str,
    label: &'static str,
    scan: u16,
    extended: bool,
    keysym: u32,
    group: &'static str,
) -> KeyInfo {
    KeyInfo { name, label, scan, extended, keysym, group }
}

pub const KEYS: &[KeyInfo] = &[
    k("w", "W", 0x11, false, 0x77, "Letters"),
    k("a", "A", 0x1E, false, 0x61, "Letters"),
    k("s", "S", 0x1F, false, 0x73, "Letters"),
    k("d", "D", 0x20, false, 0x64, "Letters"),
    k("q", "Q", 0x10, false, 0x71, "Letters"),
    k("e", "E", 0x12, false, 0x65, "Letters"),
    k("r", "R", 0x13, false, 0x72, "Letters"),
    k("f", "F", 0x21, false, 0x66, "Letters"),
    k("g", "G", 0x22, false, 0x67, "Letters"),
    k("c", "C", 0x2E, false, 0x63, "Letters"),
    k("v", "V", 0x2F, false, 0x76, "Letters"),
    k("x", "X", 0x2D, false, 0x78, "Letters"),
    k("z", "Z", 0x2C, false, 0x7A, "Letters"),
    k("b", "B", 0x30, false, 0x62, "Letters"),
    k("t", "T", 0x14, false, 0x74, "Letters"),
    k("y", "Y", 0x15, false, 0x79, "Letters"),
    k("u", "U", 0x16, false, 0x75, "Letters"),
    k("i", "I", 0x17, false, 0x69, "Letters"),
    k("o", "O", 0x18, false, 0x6F, "Letters"),
    k("p", "P", 0x19, false, 0x70, "Letters"),
    k("h", "H", 0x23, false, 0x68, "Letters"),
    k("j", "J", 0x24, false, 0x6A, "Letters"),
    k("k", "K", 0x25, false, 0x6B, "Letters"),
    k("l", "L", 0x26, false, 0x6C, "Letters"),
    k("n", "N", 0x31, false, 0x6E, "Letters"),
    k("m", "M", 0x32, false, 0x6D, "Letters"),
    k("1", "1", 0x02, false, 0x31, "Numbers"),
    k("2", "2", 0x03, false, 0x32, "Numbers"),
    k("3", "3", 0x04, false, 0x33, "Numbers"),
    k("4", "4", 0x05, false, 0x34, "Numbers"),
    k("5", "5", 0x06, false, 0x35, "Numbers"),
    k("6", "6", 0x07, false, 0x36, "Numbers"),
    k("7", "7", 0x08, false, 0x37, "Numbers"),
    k("8", "8", 0x09, false, 0x38, "Numbers"),
    k("9", "9", 0x0A, false, 0x39, "Numbers"),
    k("0", "0", 0x0B, false, 0x30, "Numbers"),
    k("space", "Space", 0x39, false, 0x20, "Common"),
    k("shift", "Shift", 0x2A, false, 0xFFE1, "Common"),
    k("ctrl", "Ctrl", 0x1D, false, 0xFFE3, "Common"),
    k("alt", "Alt", 0x38, false, 0xFFE9, "Common"),
    k("tab", "Tab", 0x0F, false, 0xFF09, "Common"),
    k("enter", "Enter", 0x1C, false, 0xFF0D, "Common"),
    k("escape", "Esc", 0x01, false, 0xFF1B, "Common"),
    k("backspace", "Backspace", 0x0E, false, 0xFF08, "Common"),
    k("capslock", "Caps Lock", 0x3A, false, 0xFFE5, "Common"),
    k("up", "Up", 0x48, true, 0xFF52, "Arrows"),
    k("down", "Down", 0x50, true, 0xFF54, "Arrows"),
    k("left", "Left", 0x4B, true, 0xFF51, "Arrows"),
    k("right", "Right", 0x4D, true, 0xFF53, "Arrows"),
    k("insert", "Insert", 0x52, true, 0xFF63, "Arrows"),
    k("delete", "Delete", 0x53, true, 0xFFFF, "Arrows"),
    k("home", "Home", 0x47, true, 0xFF50, "Arrows"),
    k("end", "End", 0x4F, true, 0xFF57, "Arrows"),
    k("pageup", "Page Up", 0x49, true, 0xFF55, "Arrows"),
    k("pagedown", "Page Down", 0x51, true, 0xFF56, "Arrows"),
    k("f1", "F1", 0x3B, false, 0xFFBE, "Function"),
    k("f2", "F2", 0x3C, false, 0xFFBF, "Function"),
    k("f3", "F3", 0x3D, false, 0xFFC0, "Function"),
    k("f4", "F4", 0x3E, false, 0xFFC1, "Function"),
    k("f5", "F5", 0x3F, false, 0xFFC2, "Function"),
    k("f6", "F6", 0x40, false, 0xFFC3, "Function"),
    k("f7", "F7", 0x41, false, 0xFFC4, "Function"),
    k("f8", "F8", 0x42, false, 0xFFC5, "Function"),
    k("f9", "F9", 0x43, false, 0xFFC6, "Function"),
    k("f10", "F10", 0x44, false, 0xFFC7, "Function"),
    k("f11", "F11", 0x57, false, 0xFFC8, "Function"),
    k("f12", "F12", 0x58, false, 0xFFC9, "Function"),
    k("grave", "`", 0x29, false, 0x60, "Symbols"),
    k("minus", "-", 0x0C, false, 0x2D, "Symbols"),
    k("equals", "=", 0x0D, false, 0x3D, "Symbols"),
    k("bracketleft", "[", 0x1A, false, 0x5B, "Symbols"),
    k("bracketright", "]", 0x1B, false, 0x5D, "Symbols"),
    k("backslash", "\\", 0x2B, false, 0x5C, "Symbols"),
    k("semicolon", ";", 0x27, false, 0x3B, "Symbols"),
    k("quote", "'", 0x28, false, 0x27, "Symbols"),
    k("comma", ",", 0x33, false, 0x2C, "Symbols"),
    k("period", ".", 0x34, false, 0x2E, "Symbols"),
    k("slash", "/", 0x35, false, 0x2F, "Symbols"),
];

pub fn key(name: &str) -> Option<&'static KeyInfo> {
    let name = name.trim().to_ascii_lowercase();
    let name = match name.as_str() {
        "esc" => "escape",
        "return" => "enter",
        "control" => "ctrl",
        other => other,
    };
    KEYS.iter().find(|k| k.name == name)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Button {
    Left,
    Right,
    Middle,
}

impl Button {
    pub const ALL: [Button; 3] = [Button::Left, Button::Right, Button::Middle];

    pub fn name(self) -> &'static str {
        match self {
            Button::Left => "left",
            Button::Right => "right",
            Button::Middle => "middle",
        }
    }

    pub fn parse(name: &str) -> Option<Button> {
        match name.trim().to_ascii_lowercase().as_str() {
            "left" | "mouse1" | "lmb" => Some(Button::Left),
            "right" | "mouse2" | "rmb" => Some(Button::Right),
            "middle" | "mouse3" | "mmb" => Some(Button::Middle),
            _ => None,
        }
    }
}

/// A key or a mouse button.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Control {
    Key(&'static str),
    Button(Button),
}

impl Control {
    pub fn parse(name: &str) -> Option<Control> {
        Button::parse(name).map(Control::Button).or_else(|| key(name).map(|k| Control::Key(k.name)))
    }

    pub fn name(self) -> &'static str {
        match self {
            Control::Key(k) => k,
            Control::Button(b) => b.name(),
        }
    }
}

/// The emergency stop: Ctrl + Alt + End, on the physical keyboard. No profile can make the
/// network press it: Ctrl and Alt are never held together (see [`refused_combination`]).
pub const STOP_KEYS: &str = "Ctrl+Alt+End";

/// Why pressing `next` while `held` is held would reach the system instead of the game, if it
/// would: Alt+Tab, Alt+F4, Alt+Esc, Alt+Space, Alt+Enter, Ctrl+Esc, and Ctrl with Alt at all.
pub fn refused_combination(held: &[&str], next: &str) -> Option<String> {
    let has = |k: &str| held.contains(&k) || next == k;
    if has("ctrl") && has("alt") {
        return Some("Ctrl and Alt together reach the system (and the emergency stop)".into());
    }
    if has("alt") {
        for k in ["tab", "f4", "escape", "space", "enter"] {
            if has(k) {
                return Some(format!("Alt+{} reaches the system", key(k).map(|i| i.label).unwrap_or(k)));
            }
        }
    }
    if has("ctrl") && has("escape") {
        return Some("Ctrl+Esc opens the system's menu".into());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_unique_and_parse() {
        for (i, a) in KEYS.iter().enumerate() {
            assert!(KEYS[i + 1..].iter().all(|b| b.name != a.name), "{}", a.name);
            assert_eq!(key(&a.name.to_uppercase()).unwrap().name, a.name);
        }
        assert_eq!(Control::parse("Shift"), Some(Control::Key("shift")));
        assert_eq!(Control::parse("Esc"), Some(Control::Key("escape")));
        assert_eq!(Control::parse("mouse2"), Some(Control::Button(Button::Right)));
        assert_eq!(Control::parse("win"), None);
        assert_eq!(Control::parse("printscreen"), None);
    }

    #[test]
    fn system_combinations_are_refused() {
        assert!(refused_combination(&["alt"], "tab").is_some());
        assert!(refused_combination(&["tab"], "alt").is_some());
        assert!(refused_combination(&["alt"], "f4").is_some());
        assert!(refused_combination(&["ctrl"], "alt").is_some());
        assert!(refused_combination(&["ctrl"], "escape").is_some());
        assert!(refused_combination(&["ctrl", "shift"], "w").is_none());
        assert!(refused_combination(&["alt"], "w").is_none());
        assert!(refused_combination(&["w", "shift"], "space").is_none());
    }
}
