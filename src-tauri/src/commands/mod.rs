//! Tauri IPC commands exposed to the React frontend.
//!
//! Each command is callable via `invoke("command_name", { args })` from
//! the frontend TypeScript code.

pub mod chat;
pub mod filesystem;
pub mod hardware;
pub mod model_download;
pub mod ollama;
pub mod python_env;
pub mod python_env_startup;
pub mod session;
pub mod settings;

/// Placeholder IPC command for initial Tauri shell verification.
#[tauri::command]
pub fn greet(name: &str) -> String {
    format!("Hello, {}! LocalCowork is running.", name)
}
