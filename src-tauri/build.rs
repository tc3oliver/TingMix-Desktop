// Every command the frontend may call is listed here; capabilities/default.json allows exactly these.
const COMMANDS: &[&str] = &[
    "list_input_devices",
    "live_start",
    "live_stop",
    "live_snapshot",
    "file_pick",
    "file_start",
    "file_cancel",
    "file_jobs",
    "export_text",
    "settings_get",
    "settings_set",
    "history_list",
    "history_load",
    "history_delete",
    "diagnostics",
    "models_overview",
    "models_import",
    "models_begin_install",
    "models_install",
    "models_cancel",
    "models_activate",
    "models_rollback",
    "models_remove",
    "models_verify",
    "models_probe",
    "backend_info",
    "updater_check",
    "updater_install",
];

fn main() {
    tauri_build::try_build(
        tauri_build::Attributes::new()
            .app_manifest(tauri_build::AppManifest::new().commands(COMMANDS)),
    )
    .expect("tauri build failed");
}
