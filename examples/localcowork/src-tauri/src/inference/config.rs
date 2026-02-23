//! Model configuration loading and validation.
//!
//! Reads `_models/config.yaml` and resolves environment variables.
//! Config is the single source of truth for model endpoints, formats, and
//! fallback chains.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::errors::InferenceError;

// ─── Public Types ────────────────────────────────────────────────────────────

/// Which tool-call format the model emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallFormat {
    /// Standard OpenAI JSON tool calls (Qwen, GPT, etc.).
    NativeJson,
    /// Pythonic `Tool: … Arguments: …` format (LFM2.5).
    Pythonic,
    /// Bracket format: `[server.tool(args)]` or `<|tool_call_start|>…<|tool_call_end|>` (LFM2-24B-A2B).
    Bracket,
}

/// A single model's runtime configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub display_name: String,
    pub runtime: String,
    #[serde(default)]
    pub model_name: Option<String>,
    #[serde(default)]
    pub model_path: Option<String>,
    pub base_url: String,
    pub context_window: u32,
    pub tool_call_format: ToolCallFormat,
    pub temperature: f32,
    pub max_tokens: u32,
    pub estimated_vram_gb: Option<f32>,
    #[serde(default)]
    pub capabilities: Vec<String>,
    /// When `true`, sends `response_format: {"type":"json_object"}` on tool-calling
    /// turns. This triggers Ollama's GBNF grammar enforcement for valid JSON output.
    /// Disabled by default — enable after live testing with the target model.
    #[serde(default)]
    pub force_json_response: bool,
    /// Optional role hint (e.g., "tool_router"). Used by the orchestrator to
    /// identify models by purpose rather than name.
    #[serde(default)]
    pub role: Option<String>,
}

/// Dual-model orchestrator configuration (ADR-009).
///
/// When enabled, GPT-OSS-20B plans multi-step workflows and LFM2-1.2B-Tool
/// executes each step with a RAG pre-filtered tool set.
#[derive(Debug, Clone, Deserialize)]
pub struct OrchestratorConfig {
    /// Whether the dual-model orchestrator is active.
    #[serde(default)]
    pub enabled: bool,
    /// Model key for the planner (e.g., "gpt-oss-20b").
    #[serde(default)]
    pub planner_model: String,
    /// Model key for the tool router (e.g., "lfm2-1.2b-tool").
    #[serde(default)]
    pub router_model: String,
    /// Top-K tools for RAG pre-filter per step (default: 15).
    #[serde(default = "default_router_top_k")]
    pub router_top_k: u32,
    /// Maximum number of steps the planner can produce (default: 10).
    #[serde(default = "default_max_plan_steps")]
    pub max_plan_steps: u32,
    /// Maximum retries per step if the router fails to produce a tool call.
    #[serde(default = "default_step_retries")]
    pub step_retries: u32,
}

fn default_router_top_k() -> u32 {
    15
}
fn default_max_plan_steps() -> u32 {
    10
}
fn default_step_retries() -> u32 {
    3
}

/// Top-level model registry (mirrors `_models/config.yaml`).
#[derive(Debug, Clone, Deserialize)]
pub struct ModelsConfig {
    pub active_model: String,
    #[serde(default)]
    pub models_dir: Option<String>,
    pub models: HashMap<String, ModelConfig>,
    #[serde(default)]
    pub fallback_chain: Vec<String>,
    /// Dual-model orchestrator settings (ADR-009). When absent, orchestration
    /// is disabled and the single-model agent loop runs as before.
    #[serde(default)]
    pub orchestrator: Option<OrchestratorConfig>,
    /// Enable two-pass category-based tool selection (Tier 1.5).
    ///
    /// When `true` and >20 MCP tools are registered, the first agent turn
    /// sends ~15 category meta-tools (~1,500 tokens) instead of all tools
    /// (~8,670 tokens). The model selects 2-3 categories, then subsequent
    /// turns use only those categories' real tools.
    ///
    /// Default: `false` (flat mode — all tools every turn).
    #[serde(default)]
    pub two_pass_tool_selection: Option<bool>,
    /// Optional allowlist of MCP server names to start.
    ///
    /// When set, only servers whose names appear in this list are started.
    /// All others are skipped during discovery. This reduces the tool count
    /// sent to the model, improving accuracy and reducing token usage.
    ///
    /// Example: `["security", "audit", "document", "ocr", "email", "system", "clipboard", "filesystem"]`
    ///
    /// Default: `None` (all discovered servers are started).
    #[serde(default)]
    pub enabled_servers: Option<Vec<String>>,
    /// Optional allowlist of fully-qualified tool names to expose to the model.
    ///
    /// When set, only tools whose names appear in this list are kept in the
    /// registry after server startup. All other tools are removed. This allows
    /// curating a tight, high-accuracy tool surface from servers that each
    /// expose more tools than needed for a specific demo or deployment.
    ///
    /// Tool names are fully-qualified: `"server.tool"` (e.g., `"filesystem.list_dir"`).
    ///
    /// Default: `None` (all tools from started servers are exposed).
    #[serde(default)]
    pub enabled_tools: Option<Vec<String>>,
}

// ─── Loading ─────────────────────────────────────────────────────────────────

/// Resolve a config path relative to the project root.
///
/// Searches upward from `start` for `_models/config.yaml`. Falls back to
/// `LOCALCOWORK_PROJECT_ROOT` env var if set.
pub fn find_config_path(start: &Path) -> Result<PathBuf, InferenceError> {
    // 1. Check env var
    if let Ok(root) = std::env::var("LOCALCOWORK_PROJECT_ROOT") {
        let candidate = PathBuf::from(&root).join("_models/config.yaml");
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    // 2. Walk upward from `start`
    let mut dir = start.to_path_buf();
    loop {
        let candidate = dir.join("_models/config.yaml");
        if candidate.exists() {
            return Ok(candidate);
        }
        if !dir.pop() {
            break;
        }
    }

    Err(InferenceError::ConfigError {
        reason: "could not find _models/config.yaml".into(),
    })
}

/// Load and parse the models configuration file.
///
/// Performs environment-variable interpolation on string values matching
/// `${VAR_NAME}` or `${VAR_NAME:-default}`.
pub fn load_models_config(path: &Path) -> Result<ModelsConfig, InferenceError> {
    let raw = std::fs::read_to_string(path).map_err(|e| InferenceError::ConfigError {
        reason: format!("failed to read {}: {e}", path.display()),
    })?;

    let interpolated = interpolate_env_vars(&raw);

    let config: ModelsConfig =
        serde_yaml::from_str(&interpolated).map_err(|e| InferenceError::ConfigError {
            reason: format!("failed to parse config: {e}"),
        })?;

    Ok(config)
}

/// Resolve the active model configuration, respecting the fallback chain.
///
/// Returns `(model_key, ModelConfig)` for the first available model.
/// "Available" here means it exists in the config — actual connectivity is
/// checked at runtime by the client.
pub fn resolve_active_model(config: &ModelsConfig) -> Result<(String, ModelConfig), InferenceError> {
    // Try the explicitly active model first
    if let Some(model) = config.models.get(&config.active_model) {
        return Ok((config.active_model.clone(), model.clone()));
    }

    // Walk the fallback chain
    for key in &config.fallback_chain {
        if key == "static_response" {
            continue; // handled by the client as a special case
        }
        if let Some(model) = config.models.get(key) {
            return Ok((key.clone(), model.clone()));
        }
    }

    Err(InferenceError::ConfigError {
        reason: format!(
            "active model '{}' not found in config and no fallback available",
            config.active_model
        ),
    })
}

// ─── Env-var interpolation ───────────────────────────────────────────────────

/// Replace `${VAR}` and `${VAR:-default}` in a string.
fn interpolate_env_vars(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '$' && chars.peek() == Some(&'{') {
            chars.next(); // consume '{'
            let mut var_expr = String::new();
            for c in chars.by_ref() {
                if c == '}' {
                    break;
                }
                var_expr.push(c);
            }
            let resolved = resolve_var_expr(&var_expr);
            result.push_str(&resolved);
        } else {
            result.push(ch);
        }
    }

    result
}

/// Resolve a variable expression like `VAR` or `VAR:-default`.
fn resolve_var_expr(expr: &str) -> String {
    if let Some(idx) = expr.find(":-") {
        let var_name = &expr[..idx];
        let default = &expr[idx + 2..];
        std::env::var(var_name).unwrap_or_else(|_| expand_tilde(default))
    } else {
        std::env::var(expr).unwrap_or_default()
    }
}

/// Expand a leading `~` to the user's home directory.
///
/// Uses `dirs::home_dir()` for cross-platform support (works on macOS,
/// Linux, and Windows where `$HOME` may not be set).
fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix('~') {
        if let Some(home) = dirs::home_dir() {
            return format!("{}{rest}", home.display());
        }
    }
    path.to_string()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolate_env_vars_with_default() {
        // When env var is NOT set, use default
        std::env::remove_var("__TEST_NONEXISTENT_VAR__");
        let input = "${__TEST_NONEXISTENT_VAR__:-/fallback/path}";
        let result = interpolate_env_vars(input);
        assert_eq!(result, "/fallback/path");
    }

    #[test]
    fn test_interpolate_env_vars_with_value() {
        std::env::set_var("__TEST_INFERENCE_VAR__", "/custom/path");
        let input = "${__TEST_INFERENCE_VAR__:-/fallback/path}";
        let result = interpolate_env_vars(input);
        assert_eq!(result, "/custom/path");
        std::env::remove_var("__TEST_INFERENCE_VAR__");
    }

    #[test]
    fn test_interpolate_no_vars() {
        let input = "plain text with no variables";
        assert_eq!(interpolate_env_vars(input), input);
    }

    #[test]
    fn test_expand_tilde() {
        let result = expand_tilde("~/Documents");
        assert!(!result.starts_with('~'), "tilde should be expanded");
        assert!(result.ends_with("/Documents"));
    }

    #[test]
    fn test_resolve_active_model_not_found() {
        let config = ModelsConfig {
            active_model: "nonexistent".into(),
            models_dir: None,
            models: HashMap::new(),
            fallback_chain: vec![],
            orchestrator: None,
            two_pass_tool_selection: None,
            enabled_servers: None,
            enabled_tools: None,
        };
        let result = resolve_active_model(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_force_json_response_default_is_false() {
        // Config YAML without force_json_response should default to false
        let yaml = r#"
            active_model: test
            models:
              test:
                display_name: "Test Model"
                runtime: ollama
                base_url: "http://localhost:11434/v1"
                context_window: 4096
                tool_call_format: native_json
                temperature: 0.7
                max_tokens: 1024
        "#;
        let config: ModelsConfig = serde_yaml::from_str(yaml).unwrap();
        let model = config.models.get("test").unwrap();
        assert!(!model.force_json_response, "force_json_response should default to false");
    }
}
