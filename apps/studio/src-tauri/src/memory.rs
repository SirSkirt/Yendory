//! Training memory (docs/GROWTH.md, section 3): what training may use on the GPU and in system
//! RAM (saved with the other settings, given to the Engine at startup and when changed), and
//! what the running training uses, for the memory panel and the progress cards.
use super::*;
use std::sync::atomic::AtomicU8;
use yendory_engine::learn::memory::{self, MemoryReport, MemorySettings};

/// The GPU safety margin in force: the GPU is opened once, with the margin of that moment.
static MARGIN_IN_FORCE: AtomicU8 = AtomicU8::new(0);

pub(crate) fn stored(app: &AppHandle) -> MemorySettings {
    read_settings(app).get("memory").and_then(|v| serde_json::from_value(v.clone()).ok()).unwrap_or_default()
}

/// Gives the saved settings to the Engine (before the GPU is first opened).
pub(crate) fn apply_stored(app: &AppHandle) {
    let s = stored(app);
    let _ = MARGIN_IN_FORCE.compare_exchange(0, s.gpu_margin_percent, Ordering::AcqRel, Ordering::Acquire);
    memory::set_memory_settings(s);
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryView {
    settings: MemorySettings,
    ram_total: u64,
    ram_available: u64,
    /// System RAM offloading may use now (within the limit, leaving the reserve free).
    ram_allowance: u64,
    /// The GPU budget training plans with (the GPU's memory less the margin), if known.
    gpu_budget: Option<u64>,
    /// The margin changed since the GPU was opened: it applies after Studio restarts.
    margin_after_restart: bool,
    /// The latest training's memory use, if training ran on the GPU since Studio started.
    report: Option<MemoryReport>,
    /// What deserves attention now (slow offloading, little RAM left).
    warnings: Vec<String>,
}

fn view(s: MemorySettings) -> MemoryView {
    let (ram_total, ram_available) = memory::system_ram();
    let report = memory::memory_report();
    let mut warnings = Vec::new();
    if let Some(r) = report.as_ref() {
        if s.warn_slowdown && r.optimizer_in_ram && r.transfer_share() > 0.2 {
            warnings.push(format!(
                "Offloading slows training: moving the gradient and weights between the GPU and system RAM takes {:.0}% of each step.",
                100. * r.transfer_share()
            ));
        }
        if s.warn_slowdown && r.micro_rows.is_some_and(|k| k <= 2) {
            warnings.push(
                "Training runs in micro-batches of one or two rows to fit in GPU memory; the GPU is used less efficiently than with larger parts."
                    .into(),
            );
        }
    }
    if ram_total > 0 && ram_available < s.ram_reserve_bytes {
        warnings.push(format!(
            "Only {} of system RAM is free, less than the {} kept in reserve: nothing more is moved to RAM.",
            memory::gib(ram_available),
            memory::gib(s.ram_reserve_bytes)
        ));
    }
    let in_force = MARGIN_IN_FORCE.load(Ordering::Acquire);
    MemoryView {
        ram_allowance: s.ram_allowance(),
        gpu_budget: s.gpu_budget(),
        margin_after_restart: in_force != 0 && in_force != s.gpu_margin_percent,
        settings: s,
        ram_total,
        ram_available,
        report,
        warnings,
    }
}

#[tauri::command]
pub fn training_memory_view() -> MemoryView {
    view(memory::memory_settings())
}

#[tauri::command]
pub fn set_training_memory(
    app: AppHandle,
    studio: State<'_, Studio>,
    settings: MemorySettings,
) -> Res<MemoryView> {
    not_teaching(&studio)?;
    check(&settings)?;
    write_setting(&app, "memory", serde_json::to_value(&settings).map_err(|e| e.to_string())?)?;
    memory::set_memory_settings(settings.clone());
    Ok(view(settings))
}

fn check(s: &MemorySettings) -> Res<()> {
    const MB: u64 = 1 << 20;
    if s.gpu_memory_bytes.is_some_and(|b| !(256 * MB..=(1u64 << 40)).contains(&b)) {
        return Err("Give the GPU's memory between 0.25 GB and 1 TB, or leave it empty".into());
    }
    if !(1..=50).contains(&s.gpu_margin_percent) {
        return Err("Keep a GPU safety margin of 1-50%".into());
    }
    let (total, _) = memory::system_ram();
    if total > 0 && s.ram_reserve_bytes >= total {
        return Err(format!("The RAM reserve must be less than this computer's {}", memory::gib(total)));
    }
    if s.micro_rows.is_some_and(|k| !(1..=1024).contains(&k)) {
        return Err("Choose 1-1024 rows per micro-batch, or leave it empty for whole batches".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_are_checked_and_round_trip() {
        let ok =
            MemorySettings { gpu_memory_bytes: Some(8 << 30), micro_rows: Some(4), ..Default::default() };
        assert!(check(&ok).is_ok());
        assert!(check(&MemorySettings { gpu_margin_percent: 0, ..ok.clone() }).is_err());
        assert!(check(&MemorySettings { gpu_memory_bytes: Some(1 << 20), ..ok.clone() }).is_err());
        assert!(check(&MemorySettings { micro_rows: Some(0), ..ok.clone() }).is_err());
        let json = serde_json::to_value(&ok).unwrap();
        assert_eq!(json["mode"], "automatic");
        assert_eq!(json["gpuMemoryBytes"], 8u64 << 30);
        assert_eq!(serde_json::from_value::<MemorySettings>(json).unwrap(), ok);
        // Older settings files have no memory settings: the defaults apply.
        assert_eq!(
            serde_json::from_value::<MemorySettings>(serde_json::json!({})).unwrap(),
            MemorySettings::default()
        );
        let v = view(ok);
        assert_eq!(v.gpu_budget, Some((8u64 << 30) / 100 * 90));
    }
}
