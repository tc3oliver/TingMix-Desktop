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
    // MLX is loaded from Contents/Frameworks in the bundle (libmlx, libjaccl, mlx.metallib, see tools/bundle-mlx.sh).
    // Cargo applies a dependency's rpath only to that dependency's own targets, so the app binary adds its own.
    // `TINGMIX_MLX_DEV_RPATH=1` additionally embeds the wheel's lib directory (searched second) for `cargo test`; never
    // set it for a bundle build, the app binary must not carry an absolute path.
    println!("cargo:rerun-if-env-changed=MLX_PREFIX");
    println!("cargo:rerun-if-env-changed=TINGMIX_MLX_DEV_RPATH");
    if std::env::var_os("CARGO_FEATURE_MLX").is_some() {
        if let Some(prefix) = std::env::var_os("MLX_PREFIX") {
            let dev = std::env::var_os("TINGMIX_MLX_DEV_RPATH").is_some();
            if dev {
                println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../Frameworks");
                println!(
                    "cargo:rustc-link-arg=-Wl,-rpath,{}",
                    std::path::Path::new(&prefix).join("lib").display()
                );
            } else {
                println!("cargo:rustc-link-arg-bins=-Wl,-rpath,@executable_path/../Frameworks");
            }
        }
    }
    tauri_build::try_build(
        tauri_build::Attributes::new()
            .app_manifest(tauri_build::AppManifest::new().commands(COMMANDS)),
    )
    .expect("tauri build failed");
}
