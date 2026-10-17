//! Tauri host: owns the audio/streaming pipeline and exposes typed commands and events to the React UI.

mod commands;
mod dto;
mod files;
mod live;
mod models;
mod state;
mod store;

use tauri::menu::{Menu, MenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{DragDropEvent, Emitter, Manager, RunEvent, WindowEvent};

use state::AppState;
use store::Store;

fn show_main(app: &tauri::AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.show();
        let _ = w.unminimize();
        let _ = w.set_focus();
    }
}

fn stop_live(app: &tauri::AppHandle) {
    let live = app.state::<AppState>().live.lock().unwrap().take();
    if let Some(mut l) = live {
        l.stop();
    }
}

pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let store = Store::new(app.path().app_config_dir()?, app.path().app_data_dir()?);
            let models = models::Models::open(app.path().app_data_dir()?.join("models"));
            app.manage(AppState::new(store, models));

            let menu = Menu::with_items(
                app,
                &[
                    &MenuItem::with_id(app, "show", "Show TingMix", true, None::<&str>)?,
                    &MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?,
                ],
            )?;
            let mut tray = TrayIconBuilder::new()
                .menu(&menu)
                .show_menu_on_left_click(false)
                .tooltip("TingMix")
                .on_menu_event(|app, ev| match ev.id().as_ref() {
                    "show" => show_main(app),
                    "quit" => app.exit(0),
                    _ => {}
                })
                .on_tray_icon_event(|tray, ev| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = ev
                    {
                        show_main(tray.app_handle());
                    }
                });
            if let Some(icon) = app.default_window_icon() {
                tray = tray.icon(icon.clone());
            }
            tray.build(app)?;
            Ok(())
        })
        .on_window_event(|window, ev| match ev {
            // recording continues in the tray; with nothing running, closing the window quits
            WindowEvent::CloseRequested { api, .. } => {
                let recording = window
                    .state::<AppState>()
                    .live
                    .lock()
                    .unwrap()
                    .as_ref()
                    .is_some_and(live::Live::is_running);
                if recording {
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
            WindowEvent::DragDrop(DragDropEvent::Enter { .. }) => {
                let _ = window.emit("tingmix://drag", true);
            }
            WindowEvent::DragDrop(DragDropEvent::Leave) => {
                let _ = window.emit("tingmix://drag", false);
            }
            WindowEvent::DragDrop(DragDropEvent::Drop { paths, .. }) => {
                let _ = window.emit("tingmix://drag", false);
                let st = window.state::<AppState>();
                let refs: Vec<_> = paths
                    .iter()
                    .filter(|p| p.is_file())
                    .map(|p| st.register_file(p.clone()))
                    .collect();
                let _ = window.emit("tingmix://files-dropped", refs);
            }
            _ => {}
        })
        .invoke_handler(tauri::generate_handler![
            commands::list_input_devices,
            commands::live_start,
            commands::live_stop,
            commands::live_snapshot,
            commands::file_pick,
            commands::file_start,
            commands::file_cancel,
            commands::file_jobs,
            commands::export_text,
            commands::settings_get,
            commands::settings_set,
            commands::history_list,
            commands::history_load,
            commands::history_delete,
            commands::diagnostics,
            commands::models_overview,
            commands::models_import,
            commands::models_begin_install,
            commands::models_install,
            commands::models_cancel,
            commands::models_activate,
            commands::models_rollback,
            commands::models_remove,
            commands::models_verify,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|app, ev| {
        if let RunEvent::Exit = ev {
            stop_live(app);
        }
    });
}
