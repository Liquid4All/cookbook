//! Tauri IPC commands for hardware detection.
//!
//! Detects CPU, RAM, GPU, and OS details to recommend the optimal
//! inference runtime and quantization level for the local LLM.

use serde::Serialize;
use sysinfo::System;

/// GPU information detected on the system.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuInfo {
    pub vendor: String,
    pub model: String,
    pub vram_gb: Option<f64>,
}

/// Complete hardware profile for the local machine.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareInfo {
    pub cpu_vendor: String,
    pub cpu_model: String,
    pub cpu_cores: u32,
    pub cpu_threads: u32,
    pub ram_total_gb: f64,
    pub ram_available_gb: f64,
    pub os_name: String,
    pub os_version: String,
    pub arch: String,
    pub gpu: Option<GpuInfo>,
    pub recommended_runtime: String,
    pub recommended_quantization: String,
}

/// Detect whether the system has Apple Silicon.
fn is_apple_silicon() -> bool {
    cfg!(target_os = "macos") && std::env::consts::ARCH == "aarch64"
}

/// Recommend an inference runtime based on detected hardware.
fn recommend_runtime(gpu: &Option<GpuInfo>) -> String {
    if is_apple_silicon() {
        return "MLX".to_string();
    }
    if let Some(ref g) = gpu {
        let vendor_lower = g.vendor.to_lowercase();
        if vendor_lower.contains("nvidia") {
            return "vLLM".to_string();
        }
    }
    "llama.cpp".to_string()
}

/// Recommend a quantization level based on total RAM.
fn recommend_quantization(ram_total_gb: f64) -> String {
    if ram_total_gb >= 32.0 {
        "Q8_0".to_string()
    } else if ram_total_gb >= 16.0 {
        "Q4_K_M".to_string()
    } else {
        "Q4_0".to_string()
    }
}

/// Detect GPU information.
///
/// - macOS Apple Silicon: reports the integrated GPU.
/// - Windows: queries WMI via `wmic` for GPU name and VRAM.
/// - Linux: parses `lspci` output for VGA controllers.
fn detect_gpu() -> Option<GpuInfo> {
    if is_apple_silicon() {
        return Some(GpuInfo {
            vendor: "Apple".to_string(),
            model: "Apple Silicon (Unified Memory)".to_string(),
            vram_gb: None,
        });
    }

    #[cfg(target_os = "windows")]
    {
        return detect_gpu_windows();
    }

    #[cfg(target_os = "linux")]
    {
        return detect_gpu_linux();
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        None
    }
}

/// Windows GPU detection via `wmic`.
///
/// Parses `wmic path win32_VideoController get Name,AdapterRAM /format:csv`.
#[cfg(target_os = "windows")]
fn detect_gpu_windows() -> Option<GpuInfo> {
    let output = std::process::Command::new("wmic")
        .args(["path", "win32_VideoController", "get", "Name,AdapterRAM", "/format:csv"])
        .output()
        .ok()?;

    let text = String::from_utf8_lossy(&output.stdout);
    // CSV format: Node,AdapterRAM,Name  (first non-empty data line)
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("Node") {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 3 {
            let adapter_ram_str = parts[1].trim();
            let name = parts[2].trim().to_string();
            if name.is_empty() {
                continue;
            }
            let vram_bytes: u64 = adapter_ram_str.parse().unwrap_or(0);
            let vram_gb = if vram_bytes > 0 {
                Some((vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0) * 10.0).round() / 10.0)
            } else {
                None
            };
            let vendor = if name.to_lowercase().contains("nvidia") {
                "NVIDIA"
            } else if name.to_lowercase().contains("amd") || name.to_lowercase().contains("radeon")
            {
                "AMD"
            } else if name.to_lowercase().contains("intel") {
                "Intel"
            } else {
                "Unknown"
            };
            return Some(GpuInfo {
                vendor: vendor.to_string(),
                model: name,
                vram_gb,
            });
        }
    }
    None
}

/// Linux GPU detection via `lspci`.
#[cfg(target_os = "linux")]
fn detect_gpu_linux() -> Option<GpuInfo> {
    let output = std::process::Command::new("lspci")
        .output()
        .ok()?;

    let text = String::from_utf8_lossy(&output.stdout);
    for line in text.lines() {
        if line.contains("VGA") || line.contains("3D controller") {
            // Format: "01:00.0 VGA compatible controller: NVIDIA Corporation ..."
            let desc = line.splitn(2, ": ").nth(1).unwrap_or(line).trim();
            let vendor = if desc.to_lowercase().contains("nvidia") {
                "NVIDIA"
            } else if desc.to_lowercase().contains("amd") || desc.to_lowercase().contains("radeon")
            {
                "AMD"
            } else if desc.to_lowercase().contains("intel") {
                "Intel"
            } else {
                "Unknown"
            };
            return Some(GpuInfo {
                vendor: vendor.to_string(),
                model: desc.to_string(),
                vram_gb: None,
            });
        }
    }
    None
}

/// Detect hardware capabilities of the local machine.
///
/// Returns CPU, RAM, GPU, OS details, and recommendations for
/// the optimal inference runtime and model quantization.
#[tauri::command]
pub async fn detect_hardware() -> Result<HardwareInfo, String> {
    let mut sys = System::new_all();
    sys.refresh_all();

    let cpus = sys.cpus();
    let (cpu_vendor, cpu_model) = if let Some(cpu) = cpus.first() {
        (cpu.vendor_id().to_string(), cpu.brand().to_string())
    } else {
        ("Unknown".to_string(), "Unknown".to_string())
    };

    let cpu_cores = sys.physical_core_count().unwrap_or(0) as u32;
    let cpu_threads = cpus.len() as u32;

    let ram_total_gb = sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
    let ram_available_gb = sys.available_memory() as f64 / (1024.0 * 1024.0 * 1024.0);

    let os_name = System::name().unwrap_or_else(|| "Unknown".to_string());
    let os_version = System::os_version().unwrap_or_else(|| "Unknown".to_string());
    let arch = std::env::consts::ARCH.to_string();

    let gpu = detect_gpu();
    let recommended_runtime = recommend_runtime(&gpu);
    let recommended_quantization = recommend_quantization(ram_total_gb);

    Ok(HardwareInfo {
        cpu_vendor,
        cpu_model,
        cpu_cores,
        cpu_threads,
        ram_total_gb: (ram_total_gb * 10.0).round() / 10.0,
        ram_available_gb: (ram_available_gb * 10.0).round() / 10.0,
        os_name,
        os_version,
        arch,
        gpu,
        recommended_runtime,
        recommended_quantization,
    })
}
