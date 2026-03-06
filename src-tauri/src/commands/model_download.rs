//! Tauri IPC commands for model downloading and verification.
//!
//! Provides streaming download with progress events, SHA-256
//! verification, and model directory management.

use futures::StreamExt;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::time::Instant;
use tauri::Emitter;
use tokio::io::AsyncWriteExt;

/// Progress update emitted during model download.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelDownloadProgress {
    pub bytes_downloaded: u64,
    pub bytes_total: u64,
    pub percent: f64,
    pub speed_mbps: f64,
    pub eta_seconds: u64,
}

/// Result of a completed model download.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelDownloadResult {
    pub success: bool,
    pub model_path: String,
    pub sha256: String,
    pub size_bytes: u64,
}

/// Get the default model directory (platform-standard data dir / models/).
///
/// Creates the directory if it does not exist.
#[tauri::command]
pub async fn get_model_dir() -> Result<String, String> {
    let model_dir = crate::data_dir().join("models");
    tokio::fs::create_dir_all(&model_dir)
        .await
        .map_err(|e| format!("Failed to create model directory: {e}"))?;
    Ok(model_dir.to_string_lossy().to_string())
}

/// Download a model file with streaming progress events.
///
/// Fetches the model from the given URL, writes to `target_dir`, and
/// emits `model-download-progress` events throughout. After download
/// completes, computes the SHA-256 hash of the file.
#[tauri::command]
pub async fn download_model(
    url: String,
    target_dir: String,
    app_handle: tauri::AppHandle,
) -> Result<ModelDownloadResult, String> {
    let client = reqwest::Client::new();

    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Failed to start download: {e}"))?;

    if !response.status().is_success() {
        return Err(format!(
            "Download failed with status: {}",
            response.status()
        ));
    }

    let bytes_total = response.content_length().unwrap_or(0);

    let filename = url
        .split('/')
        .next_back()
        .unwrap_or("model.gguf")
        .to_string();

    let target_path = PathBuf::from(&target_dir).join(&filename);

    tokio::fs::create_dir_all(&target_dir)
        .await
        .map_err(|e| format!("Failed to create target directory: {e}"))?;

    let mut file = tokio::fs::File::create(&target_path)
        .await
        .map_err(|e| format!("Failed to create file: {e}"))?;

    let mut stream = response.bytes_stream();
    let mut bytes_downloaded: u64 = 0;
    let start_time = Instant::now();
    let mut last_emit = Instant::now();

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result
            .map_err(|e| format!("Download stream error: {e}"))?;

        file.write_all(&chunk)
            .await
            .map_err(|e| format!("Failed to write chunk: {e}"))?;

        bytes_downloaded += chunk.len() as u64;

        // Emit progress at most every 100ms to avoid flooding
        if last_emit.elapsed().as_millis() >= 100
            || bytes_downloaded == bytes_total
        {
            let elapsed_secs = start_time.elapsed().as_secs_f64();
            let speed_mbps = if elapsed_secs > 0.0 {
                (bytes_downloaded as f64 / (1024.0 * 1024.0)) / elapsed_secs
            } else {
                0.0
            };

            let percent = if bytes_total > 0 {
                (bytes_downloaded as f64 / bytes_total as f64) * 100.0
            } else {
                0.0
            };

            let eta_seconds = if speed_mbps > 0.0 && bytes_total > 0 {
                let remaining_mb =
                    (bytes_total - bytes_downloaded) as f64 / (1024.0 * 1024.0);
                (remaining_mb / speed_mbps) as u64
            } else {
                0
            };

            let progress = ModelDownloadProgress {
                bytes_downloaded,
                bytes_total,
                percent: (percent * 10.0).round() / 10.0,
                speed_mbps: (speed_mbps * 100.0).round() / 100.0,
                eta_seconds,
            };

            let _ = app_handle.emit("model-download-progress", &progress);
            last_emit = Instant::now();
        }
    }

    file.flush()
        .await
        .map_err(|e| format!("Failed to flush file: {e}"))?;
    drop(file);

    let sha256 = compute_sha256(&target_path)
        .await
        .map_err(|e| format!("Failed to compute SHA-256: {e}"))?;

    Ok(ModelDownloadResult {
        success: true,
        model_path: target_path.to_string_lossy().to_string(),
        sha256,
        size_bytes: bytes_downloaded,
    })
}

/// Verify a downloaded model file against an expected SHA-256 hash.
#[tauri::command]
pub async fn verify_model(
    path: String,
    expected_sha256: String,
) -> Result<bool, String> {
    let file_path = PathBuf::from(&path);
    if !file_path.exists() {
        return Err(format!("File not found: {path}"));
    }

    let actual = compute_sha256(&file_path)
        .await
        .map_err(|e| format!("Failed to compute SHA-256: {e}"))?;

    Ok(actual.to_lowercase() == expected_sha256.to_lowercase())
}

/// Compute the SHA-256 hash of a file, reading in 8 KB chunks.
async fn compute_sha256(path: &PathBuf) -> Result<String, String> {
    use tokio::io::AsyncReadExt;

    let mut file = tokio::fs::File::open(path)
        .await
        .map_err(|e| format!("Failed to open file for hashing: {e}"))?;

    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 8192];

    loop {
        let bytes_read = file
            .read(&mut buffer)
            .await
            .map_err(|e| format!("Failed to read file: {e}"))?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    let hash = hasher.finalize();
    Ok(format!("{hash:x}"))
}
