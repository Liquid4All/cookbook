//! Tauri IPC commands for filesystem browsing.
//!
//! These stub commands enable the File Browser UI to be developed
//! and tested independently of the MCP server integration.
//! In the full integration, these will dispatch to the filesystem
//! MCP server via the McpClient.

use serde::Serialize;

/// Check if a file is hidden (cross-platform).
///
/// On Unix: files starting with '.' are hidden by convention.
/// On Windows: files with the `FILE_ATTRIBUTE_HIDDEN` attribute are hidden.
fn is_hidden(name: &str, _metadata: &std::fs::Metadata) -> bool {
    #[cfg(not(target_os = "windows"))]
    {
        name.starts_with('.')
    }
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::fs::MetadataExt;
        const FILE_ATTRIBUTE_HIDDEN: u32 = 0x2;
        _metadata.file_attributes() & FILE_ATTRIBUTE_HIDDEN != 0
    }
}

/// A single file/directory entry.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FileEntry {
    pub name: String,
    pub path: String,
    pub entry_type: String,
    pub size: u64,
    pub modified: String,
}

/// List directory contents.
///
/// Returns entries sorted: directories first, then files, both alphabetically.
#[tauri::command]
pub fn list_directory(path: String) -> Result<Vec<FileEntry>, String> {
    let dir_path = if path.starts_with('~') {
        let home = dirs::home_dir().ok_or("Cannot resolve home directory")?;
        home.join(path.strip_prefix("~/").unwrap_or(&path))
    } else {
        std::path::PathBuf::from(&path)
    };

    if !dir_path.exists() {
        return Err(format!("Directory not found: {path}"));
    }
    if !dir_path.is_dir() {
        return Err(format!("Not a directory: {path}"));
    }

    let mut entries = Vec::new();
    let read_dir = std::fs::read_dir(&dir_path)
        .map_err(|e| format!("Failed to read directory: {e}"))?;

    for entry_result in read_dir {
        let entry = entry_result.map_err(|e| format!("Failed to read entry: {e}"))?;
        let metadata = entry.metadata().map_err(|e| format!("Failed to read metadata: {e}"))?;

        let name = entry.file_name().to_string_lossy().to_string();

        // Skip hidden files
        // Unix: dot-prefix convention. Windows: FILE_ATTRIBUTE_HIDDEN attribute.
        if is_hidden(&name, &metadata) {
            continue;
        }

        let entry_type = if metadata.is_dir() {
            "dir".to_string()
        } else if metadata.file_type().is_symlink() {
            "symlink".to_string()
        } else {
            "file".to_string()
        };

        let size = metadata.len();
        let modified = metadata
            .modified()
            .map(|t| {
                chrono::DateTime::<chrono::Utc>::from(t).to_rfc3339()
            })
            .unwrap_or_default();

        entries.push(FileEntry {
            name,
            path: entry.path().to_string_lossy().to_string(),
            entry_type,
            size,
            modified,
        });
    }

    // Sort: directories first, then files, both alphabetically
    entries.sort_by(|a, b| {
        let a_is_dir = a.entry_type == "dir";
        let b_is_dir = b.entry_type == "dir";
        b_is_dir
            .cmp(&a_is_dir)
            .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
    });

    Ok(entries)
}

/// Get the user's home directory path.
#[tauri::command]
pub fn get_home_dir() -> Result<String, String> {
    let home = dirs::home_dir().ok_or("Cannot resolve home directory")?;
    Ok(home.to_string_lossy().to_string())
}
