//! Tauri IPC commands for Ollama integration.
//!
//! Provides model listing, status checking, and pull (download)
//! operations against a running Ollama instance.

use serde::{Deserialize, Serialize};
use tauri::Emitter;

/// Base URL for the Ollama HTTP API.
const OLLAMA_API_BASE: &str = "http://localhost:11434";

/// Information about a single Ollama model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OllamaModelInfo {
    pub name: String,
    pub size_bytes: u64,
    pub parameter_size: String,
    pub quantization_level: String,
}

/// Progress update emitted while pulling an Ollama model.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OllamaPullProgress {
    pub status: String,
    pub total: u64,
    pub completed: u64,
    pub percent: f64,
}

/// Raw Ollama `/api/tags` response shape.
#[derive(Debug, Deserialize)]
struct OllamaTagsResponse {
    models: Option<Vec<OllamaTagModel>>,
}

/// Raw model entry from Ollama tags API.
#[derive(Debug, Deserialize)]
struct OllamaTagModel {
    name: String,
    size: u64,
    details: Option<OllamaTagModelDetails>,
}

/// Details sub-object from Ollama tags API.
#[derive(Debug, Deserialize)]
struct OllamaTagModelDetails {
    parameter_size: Option<String>,
    quantization_level: Option<String>,
}

/// Raw progress line from Ollama `/api/pull` streaming response.
#[derive(Debug, Deserialize)]
struct OllamaPullLine {
    status: Option<String>,
    total: Option<u64>,
    completed: Option<u64>,
}

/// Base URL for the llama.cpp health endpoint (matches _models/config.yaml).
const LLAMA_SERVER_HEALTH: &str = "http://localhost:8080/health";

/// Check whether a llama-server instance is running on port 8080.
///
/// Returns `true` if the llama.cpp `/health` endpoint responds with 2xx.
#[tauri::command]
pub async fn check_llama_server_status() -> Result<bool, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .map_err(|e| format!("HTTP client error: {e}"))?;

    match client.get(LLAMA_SERVER_HEALTH).send().await {
        Ok(resp) => Ok(resp.status().is_success()),
        Err(_) => Ok(false),
    }
}

/// Check whether Ollama is running and reachable.
///
/// Returns `true` if the Ollama API at localhost:11434 responds.
#[tauri::command]
pub async fn check_ollama_status() -> Result<bool, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .map_err(|e| format!("HTTP client error: {e}"))?;

    match client.get(format!("{OLLAMA_API_BASE}/api/tags")).send().await {
        Ok(resp) => Ok(resp.status().is_success()),
        Err(_) => Ok(false),
    }
}

/// List all models currently available in the local Ollama instance.
///
/// Queries `GET /api/tags` and returns a simplified model list.
#[tauri::command]
pub async fn list_ollama_models() -> Result<Vec<OllamaModelInfo>, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .map_err(|e| format!("HTTP client error: {e}"))?;

    let response = client
        .get(format!("{OLLAMA_API_BASE}/api/tags"))
        .send()
        .await
        .map_err(|e| format!("Cannot reach Ollama: {e}"))?;

    if !response.status().is_success() {
        return Err(format!(
            "Ollama API returned status: {}",
            response.status()
        ));
    }

    let tags: OllamaTagsResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse Ollama response: {e}"))?;

    let models = tags
        .models
        .unwrap_or_default()
        .into_iter()
        .map(|m| {
            let details = m.details.unwrap_or(OllamaTagModelDetails {
                parameter_size: None,
                quantization_level: None,
            });
            OllamaModelInfo {
                name: m.name,
                size_bytes: m.size,
                parameter_size: details.parameter_size.unwrap_or_default(),
                quantization_level: details.quantization_level.unwrap_or_default(),
            }
        })
        .collect();

    Ok(models)
}

/// Pull (download) a model via Ollama's streaming API.
///
/// Streams progress events as `ollama-pull-progress` Tauri events.
/// The model name should be in Ollama format (e.g., "gpt-oss:20b").
#[tauri::command]
pub async fn pull_ollama_model(
    model_name: String,
    app_handle: tauri::AppHandle,
) -> Result<bool, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3600))
        .build()
        .map_err(|e| format!("HTTP client error: {e}"))?;

    let response = client
        .post(format!("{OLLAMA_API_BASE}/api/pull"))
        .json(&serde_json::json!({ "name": model_name, "stream": true }))
        .send()
        .await
        .map_err(|e| format!("Cannot reach Ollama: {e}"))?;

    if !response.status().is_success() {
        return Err(format!(
            "Ollama pull failed with status: {}",
            response.status()
        ));
    }

    // Stream the NDJSON response line by line
    use futures::StreamExt;
    let mut stream = response.bytes_stream();
    let mut buffer = Vec::new();

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| format!("Stream error: {e}"))?;
        buffer.extend_from_slice(&chunk);

        // Process complete lines (NDJSON — each JSON object ends with \n)
        while let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
            let line_bytes: Vec<u8> = buffer.drain(..=pos).collect();
            let line = String::from_utf8_lossy(&line_bytes);
            let trimmed = line.trim();

            if trimmed.is_empty() {
                continue;
            }

            if let Ok(pull_line) = serde_json::from_str::<OllamaPullLine>(trimmed) {
                let total = pull_line.total.unwrap_or(0);
                let completed = pull_line.completed.unwrap_or(0);
                let percent = if total > 0 {
                    (completed as f64 / total as f64) * 100.0
                } else {
                    0.0
                };

                let progress = OllamaPullProgress {
                    status: pull_line.status.unwrap_or_default(),
                    total,
                    completed,
                    percent: (percent * 10.0).round() / 10.0,
                };

                let _ = app_handle.emit("ollama-pull-progress", &progress);
            }
        }
    }

    Ok(true)
}
