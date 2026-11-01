//! Shelves: the owner's own groups of models, kept in the Hub's data folder (`shelves.json`).
//! Every model stands on exactly one shelf; a model on no shelf (new, or made by another
//! program) stands on the first one. Shelves never change the models folder.
use super::*;
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Shelf {
    id: String,
    name: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Shelves {
    shelves: Vec<Shelf>,
    /// Model id → shelf id.
    places: BTreeMap<String, String>,
}

impl Default for Shelves {
    fn default() -> Self {
        Shelves {
            shelves: vec![Shelf { id: "models".into(), name: "Models".into() }],
            places: BTreeMap::new(),
        }
    }
}

impl Shelves {
    fn has(&self, shelf: &str) -> bool {
        self.shelves.iter().any(|s| s.id == shelf)
    }
    fn add(&mut self, name: &str) -> Res<String> {
        let name = check_shelf_name(name)?;
        if self.shelves.iter().any(|s| s.name.eq_ignore_ascii_case(&name)) {
            return Err(format!("There is already a shelf called {name}"));
        }
        let id = (1..).map(|n| format!("s{n}")).find(|id| !self.has(id)).expect("a free id");
        self.shelves.push(Shelf { id: id.clone(), name });
        Ok(id)
    }
    fn rename(&mut self, shelf: &str, name: &str) -> Res<()> {
        let name = check_shelf_name(name)?;
        if self.shelves.iter().any(|s| s.id != shelf && s.name.eq_ignore_ascii_case(&name)) {
            return Err(format!("There is already a shelf called {name}"));
        }
        self.shelves.iter_mut().find(|s| s.id == shelf).ok_or("No such shelf")?.name = name;
        Ok(())
    }
    /// Removes a shelf; its models go to the first remaining shelf.
    fn delete(&mut self, shelf: &str) -> Res<()> {
        if !self.has(shelf) {
            return Err("No such shelf".into());
        }
        if self.shelves.len() == 1 {
            return Err("The last shelf can't be removed".into());
        }
        self.shelves.retain(|s| s.id != shelf);
        self.places.retain(|_, s| s != shelf);
        Ok(())
    }
    fn place(&mut self, model: &str, shelf: &str) -> Res<()> {
        if !self.has(shelf) {
            return Err("No such shelf".into());
        }
        self.places.insert(model.into(), shelf.into());
        Ok(())
    }
}

fn check_shelf_name(name: &str) -> Res<String> {
    let name = name.trim();
    if name.is_empty() || name.chars().count() > 40 || name.chars().any(char::is_control) {
        return Err("Give the shelf a name of 1 to 40 characters".into());
    }
    Ok(name.to_string())
}

fn file(app: &AppHandle) -> Res<PathBuf> {
    Ok(hub_dir(app)?.join("shelves.json"))
}

fn load(app: &AppHandle) -> Shelves {
    file(app)
        .ok()
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| serde_json::from_slice::<Shelves>(&b).ok())
        .filter(|s| !s.shelves.is_empty())
        .unwrap_or_default()
}

fn save(app: &AppHandle, s: &Shelves) -> Res<()> {
    write_atomic(&file(app)?, &serde_json::to_vec_pretty(s).map_err(|e| e.to_string())?)
}

fn change(app: &AppHandle, f: impl FnOnce(&mut Shelves) -> Res<()>) -> Res<Shelves> {
    let mut s = load(app);
    f(&mut s)?;
    save(app, &s)?;
    Ok(s)
}

/// Puts a new model on a shelf (the first one when none is given or it no longer exists).
pub(crate) fn place(app: &AppHandle, model: &str, shelf: Option<&str>) {
    let _ = change(app, |s| {
        let shelf = shelf.filter(|x| s.has(x)).map(str::to_string).unwrap_or_else(|| s.shelves[0].id.clone());
        s.place(model, &shelf)
    });
}

/// A deleted model leaves its shelf.
pub(crate) fn forget(app: &AppHandle, model: &str) {
    let _ = change(app, |s| {
        s.places.remove(model);
        Ok(())
    });
}

#[tauri::command]
pub fn shelves(app: AppHandle) -> Shelves {
    load(&app)
}

#[tauri::command]
pub fn add_shelf(app: AppHandle, name: String) -> Res<Shelves> {
    change(&app, |s| s.add(&name).map(|_| ()))
}

#[tauri::command]
pub fn rename_shelf(app: AppHandle, shelf: String, name: String) -> Res<Shelves> {
    change(&app, |s| s.rename(&shelf, &name))
}

#[tauri::command]
pub fn delete_shelf(app: AppHandle, shelf: String) -> Res<Shelves> {
    change(&app, |s| s.delete(&shelf))
}

#[tauri::command]
pub fn move_model(app: AppHandle, model: String, shelf: String) -> Res<Shelves> {
    change(&app, |s| s.place(&model, &shelf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shelves_keep_every_model_somewhere() {
        let mut s = Shelves::default();
        let lab = s.add("Experiments").unwrap();
        assert!(s.add("experiments").is_err(), "names are unique, ignoring case");
        s.place("m1", &lab).unwrap();
        s.place("m2", "models").unwrap();
        assert!(s.place("m3", "nope").is_err());
        s.rename(&lab, "Lab").unwrap();
        assert_eq!(s.shelves[1].name, "Lab");
        // Removing a shelf sends its models back to the first shelf (they lose their place).
        s.delete(&lab).unwrap();
        assert_eq!(s.places.get("m1"), None);
        assert_eq!(s.places.get("m2").map(String::as_str), Some("models"));
        assert!(s.delete("models").is_err(), "the last shelf stays");
        // A fresh id never reuses one in use.
        let a = s.add("A").unwrap();
        let b = s.add("B").unwrap();
        assert_ne!(a, b);
        assert!(check_shelf_name(" ").is_err());
    }
}
