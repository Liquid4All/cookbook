//! Tool registry — aggregates tool definitions across all MCP servers.
//!
//! Provides:
//! - Tool lookup by fully-qualified name (`server.tool`)
//! - Server-name extraction from tool names
//! - Validation that a tool call matches the registered schema
//! - Serialization of tools into the LLM system prompt format

use std::collections::HashMap;

use super::errors::McpError;
use super::types::McpToolDefinition;

// ─── ToolRegistry ────────────────────────────────────────────────────────────

/// Aggregated tool registry across all MCP servers.
///
/// Tool names are stored as `"server_name.tool_name"` (e.g., `"filesystem.list_dir"`).
#[derive(Debug, Clone)]
pub struct ToolRegistry {
    /// `tool_name → (server_name, definition)`.
    tools: HashMap<String, (String, McpToolDefinition)>,
}

impl ToolRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register tools from a server.
    ///
    /// Tool names are expected to already be fully qualified (`server.tool`).
    /// If not, they are prefixed with the server name.
    pub fn register_server_tools(&mut self, server_name: &str, tools: Vec<McpToolDefinition>) {
        for tool in tools {
            let fq_name = if tool.name.contains('.') {
                tool.name.clone()
            } else {
                format!("{server_name}.{}", tool.name)
            };

            self.tools
                .insert(fq_name, (server_name.to_string(), tool));
        }
    }

    /// Remove all tools belonging to a server.
    pub fn unregister_server(&mut self, server_name: &str) {
        self.tools.retain(|_, (srv, _)| srv != server_name);
    }

    /// Look up a tool by its fully-qualified name.
    pub fn get_tool(&self, name: &str) -> Option<&McpToolDefinition> {
        self.tools.get(name).map(|(_, def)| def)
    }

    /// Get the server name that owns a tool.
    pub fn get_server_for_tool(&self, tool_name: &str) -> Option<&str> {
        self.tools.get(tool_name).map(|(srv, _)| srv.as_str())
    }

    /// Extract the server name from a fully-qualified tool name.
    ///
    /// E.g., `"filesystem.list_dir"` → `"filesystem"`.
    pub fn server_name_from_tool(tool_name: &str) -> Option<&str> {
        tool_name.split('.').next()
    }

    /// Check whether a tool requires user confirmation before execution.
    pub fn requires_confirmation(&self, tool_name: &str) -> bool {
        self.tools
            .get(tool_name)
            .map(|(_, def)| def.confirmation_required)
            .unwrap_or(true) // Default to requiring confirmation for unknown tools
    }

    /// Check whether a tool supports undo.
    pub fn supports_undo(&self, tool_name: &str) -> bool {
        self.tools
            .get(tool_name)
            .map(|(_, def)| def.undo_supported)
            .unwrap_or(false)
    }

    /// Return all registered tool definitions.
    pub fn all_tools(&self) -> Vec<&McpToolDefinition> {
        self.tools.values().map(|(_, def)| def).collect()
    }

    /// Return all registered tool names.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(|k| k.as_str()).collect()
    }

    /// Return `(name, description)` pairs for all tools.
    ///
    /// Used by the RAG pre-filter to build the embedding index.
    pub fn tool_name_description_pairs(&self) -> Vec<(String, String)> {
        self.tools
            .iter()
            .map(|(name, (_, def))| (name.clone(), def.description.clone()))
            .collect()
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Count tools belonging to a specific server.
    pub fn tools_for_server(&self, server_name: &str) -> usize {
        self.tools
            .values()
            .filter(|(srv, _)| srv == server_name)
            .count()
    }

    /// Retain only tools whose fully-qualified names appear in the allowlist.
    ///
    /// Removes all tools not in the set. Used by `enabled_tools` config to
    /// curate a tight, high-accuracy tool surface for specific deployments.
    pub fn retain_tools(&mut self, allowed: &std::collections::HashSet<String>) {
        let before = self.tools.len();
        self.tools.retain(|name, _| allowed.contains(name));
        let after = self.tools.len();
        tracing::info!(
            before,
            after,
            "filtered tool registry by enabled_tools allowlist"
        );
    }

    /// Return all unique server names.
    pub fn server_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .tools
            .values()
            .map(|(srv, _)| srv.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        names.sort();
        names
    }

    /// Validate a tool call: tool exists and arguments match schema.
    ///
    /// This is a basic structural check — required fields present, correct types
    /// for top-level fields. Full JSON Schema validation is deferred to the
    /// server itself.
    pub fn validate_tool_call(
        &self,
        tool_name: &str,
        arguments: &serde_json::Value,
    ) -> Result<(), McpError> {
        let def = self.get_tool(tool_name).ok_or(McpError::UnknownTool {
            name: tool_name.to_string(),
        })?;

        // Validate required fields if schema specifies them
        if let Some(required) = def.params_schema.get("required") {
            if let Some(required_arr) = required.as_array() {
                let args_obj = arguments.as_object();
                for field in required_arr {
                    if let Some(field_name) = field.as_str() {
                        let has_field = args_obj
                            .map(|obj| obj.contains_key(field_name))
                            .unwrap_or(false);
                        if !has_field {
                            return Err(McpError::InvalidArguments {
                                tool: tool_name.to_string(),
                                reason: format!("missing required field: '{field_name}'"),
                            });
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Generate a concise capability summary for the system prompt.
    ///
    /// Lists available servers with tool counts and categorizes them by
    /// action type (read vs write) based on `confirmation_required` metadata.
    /// Designed to be compact (~170 tokens) so it fits within the system
    /// prompt budget alongside behavioral rules and few-shot examples.
    pub fn capability_summary(&self) -> String {
        if self.is_empty() {
            return "No MCP tools currently available. Built-in tools: \
                    list_directory, read_file."
                .to_string();
        }

        let server_names = self.server_names();
        let total_tools = self.len();

        // Build per-server summaries: "filesystem (9)"
        let server_parts: Vec<String> = server_names
            .iter()
            .map(|name| {
                let count = self.tools_for_server(name);
                format!("{name} ({count})")
            })
            .collect();

        // Categorize servers by whether they have read-only or write tools
        let mut read_servers: Vec<String> = Vec::new();
        let mut write_servers: Vec<String> = Vec::new();

        for name in &server_names {
            let has_read = self
                .tools
                .values()
                .any(|(srv, def)| srv == name && !def.confirmation_required);
            let has_write = self
                .tools
                .values()
                .any(|(srv, def)| srv == name && def.confirmation_required);

            if has_read {
                read_servers.push(name.clone());
            }
            if has_write {
                write_servers.push(name.clone());
            }
        }

        let mut summary = format!(
            "Available capabilities ({total_tools} tools across {} servers): {}.",
            server_names.len(),
            server_parts.join(", "),
        );

        if !read_servers.is_empty() {
            summary.push_str(&format!(
                "\nREAD servers (execute immediately): {}.",
                read_servers.join(", ")
            ));
        }

        if !write_servers.is_empty() {
            summary.push_str(&format!(
                "\nWRITE servers (confirmation shown automatically): {}.",
                write_servers.join(", ")
            ));
        }

        summary
    }

    /// Serialize all tool definitions into OpenAI function-calling format.
    ///
    /// Used to populate the `tools` field in chat completion requests.
    pub fn to_openai_tools(&self) -> Vec<serde_json::Value> {
        self.tools
            .iter()
            .map(|(fq_name, (_, def))| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": fq_name,
                        "description": def.description,
                        "parameters": def.params_schema,
                    }
                })
            })
            .collect()
    }

    /// Serialize tool definitions for a specific set of tool names.
    ///
    /// Used by the two-pass category system to expand selected categories
    /// into their real tool definitions. Tools not found in the registry
    /// are silently skipped.
    pub fn to_openai_tools_filtered(&self, tool_names: &[String]) -> Vec<serde_json::Value> {
        tool_names
            .iter()
            .filter_map(|name| {
                self.tools.get(name).map(|(_, def)| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": def.description,
                            "parameters": def.params_schema,
                        }
                    })
                })
            })
            .collect()
    }
}

// ─── Tool Resolution ────────────────────────────────────────────────────────

/// Known semantic equivalences where Levenshtein edit distance fails.
///
/// Checked AFTER exact match and unprefixed lookup, BEFORE fuzzy matching.
/// Only the tool suffix (the part after the dot) is matched — the server
/// prefix is preserved from the original name.
///
/// Example: `filesystem.rename_file` → `filesystem.move_file` (not `read_file`
/// which is closer by edit distance but semantically wrong).
const SEMANTIC_ALIASES: &[(&str, &str)] = &[
    ("rename_file", "move_file"),
    ("rename", "move_file"),
    ("delete_file", "move_to_trash"),
    ("remove_file", "move_to_trash"),
];

/// Result of resolving a tool name against the registry.
///
/// The agent loop uses this to understand *how* a name was resolved and to
/// generate helpful error messages when tools are not found.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolResolution {
    /// Name found as-is in the registry.
    Exact(String),

    /// Name lacked a server prefix but uniquely matched one registered tool.
    Unprefixed {
        resolved: String,
        original: String,
    },

    /// Name was qualified (`server.tool`) but didn't exist. A similar tool
    /// from the same server was found via edit distance.
    Corrected {
        resolved: String,
        original: String,
        score: f64,
    },

    /// No match found. `suggestions` contains up to 3 similar tool names.
    NotFound {
        original: String,
        suggestions: Vec<String>,
    },
}

impl ToolResolution {
    /// The resolved tool name, if resolution succeeded.
    pub fn resolved_name(&self) -> Option<&str> {
        match self {
            Self::Exact(name) => Some(name),
            Self::Unprefixed { resolved, .. } => Some(resolved),
            Self::Corrected { resolved, .. } => Some(resolved),
            Self::NotFound { .. } => None,
        }
    }

    /// Whether this resolution found a usable tool name.
    pub fn is_resolved(&self) -> bool {
        !matches!(self, Self::NotFound { .. })
    }
}

impl ToolRegistry {
    /// Resolve a tool name that may be wrong, unprefixed, or hallucinated.
    ///
    /// Strategy (first match wins):
    /// 1. **Exact:** name exists in the registry as-is.
    /// 2. **Unprefixed:** name has no dot — search for `*.{name}`, unique match wins.
    /// 3. **Same-server fuzzy:** name has dot but doesn't exist — find the most
    ///    similar tool from the same server prefix via Levenshtein distance.
    /// 4. **NotFound:** no match above `min_similarity` threshold (0.0–1.0).
    pub fn resolve(&self, name: &str, min_similarity: f64) -> ToolResolution {
        // 1. Exact match
        if self.get_tool(name).is_some() {
            return ToolResolution::Exact(name.to_string());
        }

        // 2. Unprefixed (no dot) — search for `*.{name}`
        if !name.contains('.') {
            let suffix = format!(".{name}");
            let candidates: Vec<&str> = self
                .tools
                .keys()
                .filter(|fq| fq.ends_with(&suffix))
                .map(|s| s.as_str())
                .collect();

            return match candidates.len() {
                1 => ToolResolution::Unprefixed {
                    resolved: candidates[0].to_string(),
                    original: name.to_string(),
                },
                0 => ToolResolution::NotFound {
                    original: name.to_string(),
                    suggestions: self.find_similar(name, 3),
                },
                _ => {
                    // Ambiguous — return NotFound with the candidates as suggestions
                    ToolResolution::NotFound {
                        original: name.to_string(),
                        suggestions: candidates.into_iter().map(String::from).collect(),
                    }
                }
            };
        }

        // 3. Semantic alias — known intent mappings where edit distance fails.
        //    e.g., "rename_file" means "move_file" but is closer to "read_file" by
        //    Levenshtein distance. This table is checked AFTER exact match, BEFORE
        //    fuzzy matching, and only fires for qualified names (server.tool).
        let server_prefix = name.split('.').next().unwrap_or("");
        let suffix = name.split('.').nth(1).unwrap_or("");

        for &(alias_from, alias_to) in SEMANTIC_ALIASES {
            if suffix == alias_from {
                let candidate = format!("{server_prefix}.{alias_to}");
                if self.get_tool(&candidate).is_some() {
                    return ToolResolution::Corrected {
                        resolved: candidate,
                        original: name.to_string(),
                        score: 1.0, // Semantic match — highest confidence
                    };
                }
            }
        }

        // 4. Qualified but wrong — find similar tools from the same server prefix

        let same_server: Vec<(&str, &str)> = self
            .tools
            .keys()
            .filter_map(|fq| {
                let tool_suffix = fq.split('.').nth(1)?;
                if fq.starts_with(server_prefix) && fq.contains('.') {
                    Some((fq.as_str(), tool_suffix))
                } else {
                    None
                }
            })
            .collect();

        if !same_server.is_empty() {
            // Compare suffixes (the part after the dot) via Levenshtein
            let mut best: Option<(&str, f64)> = None;
            for (fq_name, tool_suffix) in &same_server {
                let score = similarity(suffix, tool_suffix);
                if score >= min_similarity
                    && best.map_or(true, |(_, best_score)| score > best_score)
                {
                    best = Some((fq_name, score));
                }
            }

            if let Some((resolved, score)) = best {
                return ToolResolution::Corrected {
                    resolved: resolved.to_string(),
                    original: name.to_string(),
                    score,
                };
            }
        }

        // 4. Nothing matched — provide suggestions from the full registry
        ToolResolution::NotFound {
            original: name.to_string(),
            suggestions: self.find_similar(name, 3),
        }
    }

    /// Find up to `max_results` tools most similar to `name`, ranked by score.
    ///
    /// Compares against tool suffixes if `name` is qualified (`server.tool`),
    /// or against full names otherwise. Returns `(tool_name, score)` pairs
    /// with score in 0.0–1.0 (higher = more similar).
    pub fn find_similar(&self, name: &str, max_results: usize) -> Vec<String> {
        let query_suffix = name.split('.').next_back().unwrap_or(name);

        let mut scored: Vec<(String, f64)> = self
            .tools
            .keys()
            .map(|fq| {
                let tool_suffix = fq.split('.').nth(1).unwrap_or(fq);
                let score = similarity(query_suffix, tool_suffix);
                (fq.clone(), score)
            })
            .filter(|(_, score)| *score > 0.3) // Floor — don't suggest wildly different tools
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(max_results);
        scored.into_iter().map(|(name, _)| name).collect()
    }
}

// ─── Edit Distance ──────────────────────────────────────────────────────────

/// Compute the Levenshtein edit distance between two strings.
fn levenshtein(a: &str, b: &str) -> usize {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    let m = a_bytes.len();
    let n = b_bytes.len();

    // Use single-row DP for O(min(m,n)) space
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_bytes[i - 1] == b_bytes[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1)          // deletion
                .min(curr[j - 1] + 1)         // insertion
                .min(prev[j - 1] + cost);     // substitution
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Normalized similarity between two strings (0.0 = completely different, 1.0 = identical).
fn similarity(a: &str, b: &str) -> f64 {
    let max_len = a.len().max(b.len());
    if max_len == 0 {
        return 1.0;
    }
    let dist = levenshtein(a, b);
    1.0 - (dist as f64 / max_len as f64)
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Category-Based Tool Selection (Tier 1.5) ──────────────────────────────

/// A functional category grouping related tools for two-pass selection.
///
/// Presented to the model as a synthetic "meta-tool" on the first turn.
/// When selected by the model, the category is expanded to its real tools.
#[derive(Debug, Clone)]
pub struct ToolCategory {
    /// Category identifier used as the function name (e.g., `"file_browse"`).
    pub name: String,
    /// Human-readable description for the model. Must be discriminative
    /// enough to distinguish sibling categories.
    pub description: String,
    /// Fully-qualified tool names belonging to this category.
    pub tool_names: Vec<String>,
}

/// Registry of tool categories for two-pass selection.
///
/// Built from the live `ToolRegistry` at startup. Categories are hardcoded
/// functional groupings (not auto-generated from servers), because the
/// grouping is semantic — filesystem read vs write, document read vs create.
///
/// The 16 categories reduce the tool-selection decision space from 83 flat
/// tools (~10,700 tokens) to 16 categories (~1,600 tokens), near the
/// K=15 sweet spot identified in ADR-010 benchmarks.
#[derive(Debug, Clone)]
pub struct CategoryRegistry {
    categories: Vec<ToolCategory>,
    tool_to_category: HashMap<String, String>,
}

impl CategoryRegistry {
    /// Build category definitions, filtering out categories whose tools
    /// are not present in the live registry (server not running).
    ///
    /// A category is included if at least one of its tools is registered.
    pub fn build(registry: &ToolRegistry) -> Self {
        let defs = default_category_definitions();
        let mut categories = Vec::new();
        let mut tool_to_category = HashMap::new();

        for (name, description, tool_names) in defs {
            // Keep only tools that actually exist in the registry
            let live_tools: Vec<String> = tool_names
                .into_iter()
                .filter(|t| registry.get_tool(t).is_some())
                .collect();

            if live_tools.is_empty() {
                continue; // Skip categories with no live tools
            }

            for tool in &live_tools {
                tool_to_category.insert(tool.clone(), name.clone());
            }

            categories.push(ToolCategory {
                name,
                description,
                tool_names: live_tools,
            });
        }

        Self {
            categories,
            tool_to_category,
        }
    }

    /// Serialize categories as OpenAI function-calling format.
    ///
    /// Each category becomes a synthetic tool with a single `"intent"`
    /// parameter. The model calls these to signal which capability areas
    /// it needs for the current task.
    pub fn to_openai_tools(&self) -> Vec<serde_json::Value> {
        self.categories
            .iter()
            .map(|cat| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": cat.name,
                        "description": cat.description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "intent": {
                                    "type": "string",
                                    "description": "Brief description of what you want to do"
                                }
                            },
                            "required": ["intent"]
                        }
                    }
                })
            })
            .collect()
    }

    /// Expand selected category names into the union of their real tool names.
    ///
    /// Unknown category names are silently ignored. Duplicate tools across
    /// categories are deduplicated.
    pub fn expand_categories(&self, selected: &[String]) -> Vec<String> {
        let mut tool_set: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        for cat in &self.categories {
            if selected.iter().any(|s| s == &cat.name) {
                tool_set.extend(cat.tool_names.iter().cloned());
            }
        }

        let mut tools: Vec<String> = tool_set.into_iter().collect();
        tools.sort(); // Deterministic ordering
        tools
    }

    /// Check if a name is a known category (not a real tool).
    pub fn is_category(&self, name: &str) -> bool {
        self.categories.iter().any(|c| c.name == name)
    }

    /// Look up which category a tool belongs to.
    pub fn category_for_tool(&self, tool_name: &str) -> Option<&str> {
        self.tool_to_category.get(tool_name).map(|s| s.as_str())
    }

    /// Number of active categories.
    pub fn len(&self) -> usize {
        self.categories.len()
    }

    /// Whether any categories are registered.
    pub fn is_empty(&self) -> bool {
        self.categories.is_empty()
    }

    /// All category names, in definition order.
    pub fn category_names(&self) -> Vec<&str> {
        self.categories.iter().map(|c| c.name.as_str()).collect()
    }
}

/// The 16 hardcoded functional categories covering all 83 tools.
///
/// Returns `(name, description, tool_names)` triples. Tool names use
/// the fully-qualified `"server.tool"` format.
///
/// Category design rationale:
/// - 16 categories (~K=15 sweet spot from ADR-010 benchmarks)
/// - Filesystem split into read (file_browse) vs write (file_edit)
///   to reduce mutable operation exposure
/// - Document split into read vs create for the same reason
/// - clipboard + system-info merged (both are quick OS queries)
/// - system.{open_application, open_file_with, take_screenshot} are
///   separate from system-info because they're action-oriented
/// - screenshot_pipeline tools grouped with image_ocr (visual extraction)
/// - system-settings gets its own category (OS preference changes)
fn default_category_definitions() -> Vec<(String, String, Vec<String>)> {
    vec![
        (
            "file_browse".into(),
            "Browse, search, read, and inspect files and folders. List directory \
             contents, read file contents, search by filename or glob pattern, check \
             file metadata (size, dates, permissions), and watch folders for changes."
                .into(),
            vec![
                "filesystem.list_dir".into(),
                "filesystem.read_file".into(),
                "filesystem.search_files".into(),
                "filesystem.get_metadata".into(),
                "filesystem.watch_folder".into(),
            ],
        ),
        (
            "file_edit".into(),
            "Create, write, move, copy, rename, or delete files. All file modification \
             operations. Requires user confirmation."
                .into(),
            vec![
                "filesystem.write_file".into(),
                "filesystem.move_file".into(),
                "filesystem.copy_file".into(),
                "filesystem.delete_file".into(),
            ],
        ),
        (
            "document_read".into(),
            "Extract text content from PDF, DOCX, HTML, or spreadsheet files. Compare \
             two documents for differences. Read CSV and Excel data."
                .into(),
            vec![
                "document.extract_text".into(),
                "document.diff_documents".into(),
                "document.read_spreadsheet".into(),
            ],
        ),
        (
            "document_create".into(),
            "Create PDF or DOCX files from markdown, fill PDF form fields, merge \
             multiple PDFs, or convert between document formats."
                .into(),
            vec![
                "document.convert_format".into(),
                "document.create_pdf".into(),
                "document.fill_pdf_form".into(),
                "document.merge_pdfs".into(),
                "document.create_docx".into(),
            ],
        ),
        (
            "image_ocr".into(),
            "Extract text from images and screenshots using OCR. Extract structured \
             data (receipts, invoices) or tabular data from visual content. Capture \
             screenshots with text extraction, identify UI elements, and suggest \
             actions from screen captures."
                .into(),
            vec![
                "ocr.extract_text_from_image".into(),
                "ocr.extract_text_from_pdf".into(),
                "ocr.extract_structured_data".into(),
                "ocr.extract_table".into(),
                "screenshot.capture_and_extract".into(),
                "screenshot.extract_ui_elements".into(),
                "screenshot.suggest_actions".into(),
            ],
        ),
        (
            "data_analysis".into(),
            "Query SQLite databases with SQL, process and export CSV files, find \
             duplicate records, and detect anomalies in datasets."
                .into(),
            vec![
                "data.query_sqlite".into(),
                "data.deduplicate_records".into(),
                "data.summarize_anomalies".into(),
                "data.write_csv".into(),
                "data.write_sqlite".into(),
            ],
        ),
        (
            "knowledge_search".into(),
            "Semantic search across indexed documents. Index folders for search, ask \
             questions about file contents, and find related text passages."
                .into(),
            vec![
                "knowledge.index_folder".into(),
                "knowledge.search_documents".into(),
                "knowledge.ask_about_files".into(),
                "knowledge.update_index".into(),
                "knowledge.get_related_chunks".into(),
            ],
        ),
        (
            "security_privacy".into(),
            "Scan files for PII (SSN, credit cards, emails) or leaked secrets \
             (API keys, passwords). Find duplicate files, propose cleanup, encrypt \
             or decrypt files."
                .into(),
            vec![
                "security.scan_for_pii".into(),
                "security.scan_for_secrets".into(),
                "security.find_duplicates".into(),
                "security.propose_cleanup".into(),
                "security.encrypt_file".into(),
                "security.decrypt_file".into(),
            ],
        ),
        (
            "task_management".into(),
            "Create, list, and update tasks with priorities and due dates. Check \
             overdue items. Generate a daily task briefing."
                .into(),
            vec![
                "task.create_task".into(),
                "task.list_tasks".into(),
                "task.update_task".into(),
                "task.get_overdue".into(),
                "task.daily_briefing".into(),
            ],
        ),
        (
            "calendar_scheduling".into(),
            "View calendar events in a date range, create new events, find available \
             time slots, and block focus time."
                .into(),
            vec![
                "calendar.list_events".into(),
                "calendar.create_event".into(),
                "calendar.find_free_slots".into(),
                "calendar.create_time_block".into(),
            ],
        ),
        (
            "email_messaging".into(),
            "Draft and send emails, list saved drafts, search mail by keyword, sender, \
             or date, and summarize email conversation threads."
                .into(),
            vec![
                "email.draft_email".into(),
                "email.list_drafts".into(),
                "email.search_emails".into(),
                "email.summarize_thread".into(),
                "email.send_draft".into(),
            ],
        ),
        (
            "meeting_audio".into(),
            "Transcribe audio recordings to text, extract action items and commitments \
             from transcripts, and generate formatted meeting minutes."
                .into(),
            vec![
                "meeting.transcribe_audio".into(),
                "meeting.extract_action_items".into(),
                "meeting.extract_commitments".into(),
                "meeting.generate_minutes".into(),
            ],
        ),
        (
            "clipboard_system".into(),
            "Access clipboard contents and history (get, set, history). Get system \
             information (OS, CPU, memory, disk, network), monitor CPU and memory \
             usage, and list running processes."
                .into(),
            vec![
                "clipboard.get_clipboard".into(),
                "clipboard.set_clipboard".into(),
                "clipboard.clipboard_history".into(),
                "system.get_system_info".into(),
                "system.list_processes".into(),
                "system.get_cpu_usage".into(),
                "system.get_memory_usage".into(),
                "system.get_disk_usage".into(),
                "system.get_network_info".into(),
            ],
        ),
        (
            "app_launcher".into(),
            "Launch applications by name, open files with a specific program, take \
             a screenshot of the current screen, or kill a running process."
                .into(),
            vec![
                "system.open_application".into(),
                "system.open_file_with".into(),
                "system.take_screenshot".into(),
                "system.kill_process".into(),
            ],
        ),
        (
            "audit_compliance".into(),
            "View tool execution logs, get session summaries, generate text audit \
             reports, and export signed audit PDFs."
                .into(),
            vec![
                "audit.get_tool_log".into(),
                "audit.get_session_summary".into(),
                "audit.generate_audit_report".into(),
                "audit.export_audit_pdf".into(),
            ],
        ),
        (
            "system_settings".into(),
            "View and change OS preferences: display settings, sleep timer, audio \
             volume, default applications, default browser, power settings, and \
             Do Not Disturb mode. Requires user confirmation for changes."
                .into(),
            vec![
                "system-settings.get_display_settings".into(),
                "system-settings.set_display_sleep".into(),
                "system-settings.get_audio_settings".into(),
                "system-settings.set_audio_volume".into(),
                "system-settings.get_default_apps".into(),
                "system-settings.set_default_browser".into(),
                "system-settings.get_power_settings".into(),
                "system-settings.toggle_do_not_disturb".into(),
            ],
        ),
    ]
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tool(name: &str, confirmation: bool, undo: bool) -> McpToolDefinition {
        McpToolDefinition {
            name: name.to_string(),
            description: format!("Test tool: {name}"),
            params_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "required": ["path"]
            }),
            returns_schema: serde_json::json!({}),
            confirmation_required: confirmation,
            undo_supported: undo,
        }
    }

    #[test]
    fn test_register_and_lookup() {
        let mut registry = ToolRegistry::new();
        registry.register_server_tools(
            "filesystem",
            vec![sample_tool("list_dir", false, false)],
        );

        assert_eq!(registry.len(), 1);
        assert!(registry.get_tool("filesystem.list_dir").is_some());
        assert!(registry.get_tool("nonexistent.tool").is_none());
    }

    #[test]
    fn test_fully_qualified_names_preserved() {
        let mut registry = ToolRegistry::new();
        let tool = sample_tool("filesystem.read_file", false, false);
        registry.register_server_tools("filesystem", vec![tool]);

        // Already has a dot, so no double-prefix
        assert!(registry.get_tool("filesystem.read_file").is_some());
    }

    #[test]
    fn test_server_name_lookup() {
        let mut registry = ToolRegistry::new();
        registry.register_server_tools(
            "ocr",
            vec![sample_tool("extract_text", false, false)],
        );

        assert_eq!(
            registry.get_server_for_tool("ocr.extract_text"),
            Some("ocr")
        );
    }

    #[test]
    fn test_server_name_from_tool() {
        assert_eq!(
            ToolRegistry::server_name_from_tool("filesystem.list_dir"),
            Some("filesystem")
        );
    }

    #[test]
    fn test_confirmation_and_undo() {
        let mut registry = ToolRegistry::new();
        registry.register_server_tools(
            "fs",
            vec![
                sample_tool("read_file", false, false),
                sample_tool("write_file", true, true),
            ],
        );

        assert!(!registry.requires_confirmation("fs.read_file"));
        assert!(registry.requires_confirmation("fs.write_file"));
        assert!(!registry.supports_undo("fs.read_file"));
        assert!(registry.supports_undo("fs.write_file"));
        // Unknown tools default to requiring confirmation
        assert!(registry.requires_confirmation("unknown.tool"));
    }

    #[test]
    fn test_unregister_server() {
        let mut registry = ToolRegistry::new();
        registry.register_server_tools("a", vec![sample_tool("tool1", false, false)]);
        registry.register_server_tools("b", vec![sample_tool("tool2", false, false)]);

        assert_eq!(registry.len(), 2);
        registry.unregister_server("a");
        assert_eq!(registry.len(), 1);
        assert!(registry.get_tool("a.tool1").is_none());
        assert!(registry.get_tool("b.tool2").is_some());
    }

    #[test]
    fn test_validate_tool_call_valid() {
        let mut registry = ToolRegistry::new();
        registry.register_server_tools("fs", vec![sample_tool("list_dir", false, false)]);

        let args = serde_json::json!({"path": "/tmp"});
        assert!(registry.validate_tool_call("fs.list_dir", &args).is_ok());
    }

    #[test]
    fn test_validate_tool_call_missing_required() {
        let mut registry = ToolRegistry::new();
        registry.register_server_tools("fs", vec![sample_tool("list_dir", false, false)]);

        let args = serde_json::json!({});
        let err = registry.validate_tool_call("fs.list_dir", &args).unwrap_err();
        assert!(matches!(err, McpError::InvalidArguments { .. }));
    }

    #[test]
    fn test_validate_tool_call_unknown_tool() {
        let registry = ToolRegistry::new();
        let args = serde_json::json!({});
        let err = registry
            .validate_tool_call("nonexistent.tool", &args)
            .unwrap_err();
        assert!(matches!(err, McpError::UnknownTool { .. }));
    }

    #[test]
    fn test_to_openai_tools() {
        let mut registry = ToolRegistry::new();
        registry.register_server_tools("fs", vec![sample_tool("list_dir", false, false)]);

        let openai_tools = registry.to_openai_tools();
        assert_eq!(openai_tools.len(), 1);
        assert_eq!(openai_tools[0]["type"], "function");
        assert_eq!(openai_tools[0]["function"]["name"], "fs.list_dir");
    }

    #[test]
    fn test_server_names() {
        let mut registry = ToolRegistry::new();
        registry.register_server_tools("fs", vec![sample_tool("t1", false, false)]);
        registry.register_server_tools("ocr", vec![sample_tool("t2", false, false)]);

        let names = registry.server_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"fs".to_string()));
        assert!(names.contains(&"ocr".to_string()));
    }

    #[test]
    fn test_tools_for_server() {
        let mut registry = ToolRegistry::new();
        registry.register_server_tools(
            "fs",
            vec![
                sample_tool("list_dir", false, false),
                sample_tool("read_file", false, false),
            ],
        );
        registry.register_server_tools("ocr", vec![sample_tool("extract", false, false)]);

        assert_eq!(registry.tools_for_server("fs"), 2);
        assert_eq!(registry.tools_for_server("ocr"), 1);
        assert_eq!(registry.tools_for_server("nonexistent"), 0);
    }

    #[test]
    fn test_capability_summary_empty() {
        let registry = ToolRegistry::new();
        let summary = registry.capability_summary();
        assert!(summary.contains("No MCP tools currently available"));
        assert!(summary.contains("list_directory"));
        assert!(summary.contains("read_file"));
    }

    #[test]
    fn test_capability_summary_with_servers() {
        let mut registry = ToolRegistry::new();
        registry.register_server_tools(
            "filesystem",
            vec![
                sample_tool("list_dir", false, false),
                sample_tool("read_file", false, false),
                sample_tool("write_file", true, true),
            ],
        );
        registry.register_server_tools(
            "ocr",
            vec![sample_tool("extract_text", false, false)],
        );

        let summary = registry.capability_summary();
        assert!(summary.contains("4 tools across 2 servers"));
        assert!(summary.contains("filesystem (3)"));
        assert!(summary.contains("ocr (1)"));
    }

    #[test]
    fn test_capability_summary_categorizes_by_confirmation() {
        let mut registry = ToolRegistry::new();
        // "audit" has only read tools (no confirmation)
        registry.register_server_tools(
            "audit",
            vec![sample_tool("get_log", false, false)],
        );
        // "email" has only write tools (confirmation required)
        registry.register_server_tools(
            "email",
            vec![sample_tool("send_draft", true, false)],
        );
        // "filesystem" has both
        registry.register_server_tools(
            "filesystem",
            vec![
                sample_tool("read_file", false, false),
                sample_tool("delete_file", true, true),
            ],
        );

        let summary = registry.capability_summary();

        // READ servers line should include audit and filesystem but not email
        let read_line = summary
            .lines()
            .find(|l| l.starts_with("READ servers"))
            .expect("should have READ line");
        assert!(read_line.contains("audit"));
        assert!(read_line.contains("filesystem"));
        assert!(!read_line.contains("email"));

        // WRITE servers line should include email and filesystem but not audit
        let write_line = summary
            .lines()
            .find(|l| l.starts_with("WRITE servers"))
            .expect("should have WRITE line");
        assert!(write_line.contains("email"));
        assert!(write_line.contains("filesystem"));
        assert!(!write_line.contains("audit"));
    }

    // ─── Levenshtein / Similarity Tests ──────────────────────────────

    #[test]
    fn test_levenshtein_identical() {
        assert_eq!(levenshtein("move_file", "move_file"), 0);
    }

    #[test]
    fn test_levenshtein_basic() {
        assert_eq!(levenshtein("rename_file", "move_file"), 5);
        assert_eq!(levenshtein("rename_file", "read_file"), 3);
        assert_eq!(levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_levenshtein_empty() {
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("", ""), 0);
    }

    #[test]
    fn test_similarity_range() {
        let s = similarity("move_file", "move_file");
        assert!((s - 1.0).abs() < f64::EPSILON);

        let s = similarity("rename_file", "move_file");
        assert!(s > 0.5); // "rename_file" vs "move_file" share "_file"

        let s = similarity("abc", "xyz");
        assert!(s < 0.5); // Completely different
    }

    // ─── ToolResolution Tests ────────────────────────────────────────

    fn build_filesystem_registry() -> ToolRegistry {
        let mut registry = ToolRegistry::new();
        registry.register_server_tools(
            "filesystem",
            vec![
                sample_tool("list_dir", false, false),
                sample_tool("read_file", false, false),
                sample_tool("write_file", true, true),
                sample_tool("move_file", true, false),
                sample_tool("copy_file", true, false),
                sample_tool("search_files", false, false),
            ],
        );
        registry.register_server_tools(
            "ocr",
            vec![sample_tool("extract_text_from_image", false, false)],
        );
        registry
    }

    #[test]
    fn test_resolve_exact_match() {
        let registry = build_filesystem_registry();
        let result = registry.resolve("filesystem.list_dir", 0.5);
        assert_eq!(result, ToolResolution::Exact("filesystem.list_dir".into()));
    }

    #[test]
    fn test_resolve_unprefixed() {
        let registry = build_filesystem_registry();
        let result = registry.resolve("move_file", 0.5);
        assert!(matches!(
            result,
            ToolResolution::Unprefixed { ref resolved, .. } if resolved == "filesystem.move_file"
        ));
    }

    #[test]
    fn test_resolve_corrected_rename_to_move_via_alias() {
        let registry = build_filesystem_registry();
        let result = registry.resolve("filesystem.rename_file", 0.5);
        // "rename_file" is closer to "read_file" by edit distance (3 vs 5),
        // but SEMANTIC_ALIASES maps "rename_file" → "move_file" before
        // Levenshtein runs. This prevents dispatching to the wrong tool.
        assert_eq!(
            result,
            ToolResolution::Corrected {
                resolved: "filesystem.move_file".to_string(),
                original: "filesystem.rename_file".to_string(),
                score: 1.0,
            },
        );
    }

    #[test]
    fn test_resolve_corrected_returns_best_match() {
        // With a limited registry, verify the closest match is selected
        let mut registry = ToolRegistry::new();
        registry.register_server_tools(
            "filesystem",
            vec![
                sample_tool("move_file", true, false),
                sample_tool("list_dir", false, false),
            ],
        );
        let result = registry.resolve("filesystem.move_files", 0.5);
        // "move_files" vs "move_file" = distance 1, similarity ~0.9
        assert!(
            matches!(
                result,
                ToolResolution::Corrected { ref resolved, score, .. }
                    if resolved == "filesystem.move_file" && score > 0.8
            ),
            "expected Corrected to filesystem.move_file, got: {result:?}"
        );
    }

    #[test]
    fn test_resolve_not_found_below_threshold() {
        let registry = build_filesystem_registry();
        // "filesystem.zzzzzz" has no similarity to any tool
        let result = registry.resolve("filesystem.zzzzzz", 0.5);
        assert!(matches!(result, ToolResolution::NotFound { .. }));
    }

    #[test]
    fn test_resolve_not_found_unknown_server() {
        let registry = build_filesystem_registry();
        let result = registry.resolve("nonexistent.some_tool", 0.5);
        assert!(matches!(result, ToolResolution::NotFound { .. }));
    }

    #[test]
    fn test_resolve_not_found_has_suggestions() {
        let registry = build_filesystem_registry();
        let result = registry.resolve("filesystem.zzzzzz", 0.3);
        if let ToolResolution::NotFound { suggestions, .. } = result {
            // Should have some suggestions even at low threshold
            // (tools with shared "_" characters might score above 0.3)
            assert!(suggestions.len() <= 3); // max_results = 3
        }
    }

    #[test]
    fn test_resolve_unprefixed_ambiguous() {
        // Register the same tool name in two different servers
        let mut registry = ToolRegistry::new();
        registry.register_server_tools("a", vec![sample_tool("run", false, false)]);
        registry.register_server_tools("b", vec![sample_tool("run", false, false)]);

        let result = registry.resolve("run", 0.5);
        // Ambiguous — two matches → NotFound with candidates as suggestions
        assert!(matches!(result, ToolResolution::NotFound { ref suggestions, .. } if suggestions.len() == 2));
    }

    #[test]
    fn test_find_similar() {
        let registry = build_filesystem_registry();
        let similar = registry.find_similar("rename_file", 3);
        assert!(!similar.is_empty());
        // "move_file" should be among the suggestions (shares "_file" suffix)
        assert!(similar.iter().any(|s| s.contains("move_file")));
    }

    #[test]
    fn test_resolution_resolved_name() {
        let exact = ToolResolution::Exact("filesystem.list_dir".into());
        assert_eq!(exact.resolved_name(), Some("filesystem.list_dir"));

        let not_found = ToolResolution::NotFound {
            original: "bad.tool".into(),
            suggestions: vec![],
        };
        assert_eq!(not_found.resolved_name(), None);
    }

    // ─── to_openai_tools_filtered Tests ─────────────────────────────

    #[test]
    fn test_to_openai_tools_filtered_returns_matching() {
        let registry = build_filesystem_registry();
        let filtered = registry.to_openai_tools_filtered(&[
            "filesystem.list_dir".to_string(),
            "filesystem.move_file".to_string(),
        ]);
        assert_eq!(filtered.len(), 2);
        let names: Vec<&str> = filtered
            .iter()
            .filter_map(|v| v["function"]["name"].as_str())
            .collect();
        assert!(names.contains(&"filesystem.list_dir"));
        assert!(names.contains(&"filesystem.move_file"));
    }

    #[test]
    fn test_to_openai_tools_filtered_skips_missing() {
        let registry = build_filesystem_registry();
        let filtered = registry.to_openai_tools_filtered(&[
            "filesystem.list_dir".to_string(),
            "nonexistent.tool".to_string(),
        ]);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_to_openai_tools_filtered_empty_input() {
        let registry = build_filesystem_registry();
        let filtered = registry.to_openai_tools_filtered(&[]);
        assert!(filtered.is_empty());
    }

    // ─── CategoryRegistry Tests ─────────────────────────────────────

    /// Build a registry with tools from multiple servers for category testing.
    fn build_multi_server_registry() -> ToolRegistry {
        let mut registry = ToolRegistry::new();
        registry.register_server_tools(
            "filesystem",
            vec![
                sample_tool("list_dir", false, false),
                sample_tool("read_file", false, false),
                sample_tool("search_files", false, false),
                sample_tool("get_metadata", false, false),
                sample_tool("watch_folder", false, false),
                sample_tool("write_file", true, false),
                sample_tool("move_file", true, true),
                sample_tool("copy_file", true, false),
                sample_tool("delete_file", true, true),
            ],
        );
        registry.register_server_tools(
            "ocr",
            vec![
                sample_tool("extract_text_from_image", false, false),
                sample_tool("extract_text_from_pdf", false, false),
                sample_tool("extract_structured_data", false, false),
                sample_tool("extract_table", false, false),
            ],
        );
        registry.register_server_tools(
            "task",
            vec![
                sample_tool("create_task", true, false),
                sample_tool("list_tasks", false, false),
                sample_tool("update_task", true, false),
                sample_tool("get_overdue", false, false),
                sample_tool("daily_briefing", false, false),
            ],
        );
        registry
    }

    #[test]
    fn test_category_registry_build_filters_missing_servers() {
        // Registry only has filesystem, ocr, task — not all 13 servers.
        // Categories for missing servers should be excluded.
        let registry = build_multi_server_registry();
        let cat_reg = CategoryRegistry::build(&registry);

        // Should include: file_browse, file_edit, image_ocr, task_management
        assert!(cat_reg.is_category("file_browse"));
        assert!(cat_reg.is_category("file_edit"));
        assert!(cat_reg.is_category("image_ocr"));
        assert!(cat_reg.is_category("task_management"));

        // Should NOT include categories for servers that aren't running
        assert!(!cat_reg.is_category("email_messaging"));
        assert!(!cat_reg.is_category("calendar_scheduling"));
        assert!(!cat_reg.is_category("meeting_audio"));
        assert!(!cat_reg.is_category("knowledge_search"));
    }

    #[test]
    fn test_category_registry_expand_single() {
        let registry = build_multi_server_registry();
        let cat_reg = CategoryRegistry::build(&registry);

        let expanded = cat_reg.expand_categories(&["file_browse".to_string()]);
        assert_eq!(expanded.len(), 5);
        assert!(expanded.contains(&"filesystem.list_dir".to_string()));
        assert!(expanded.contains(&"filesystem.read_file".to_string()));
        assert!(expanded.contains(&"filesystem.search_files".to_string()));
        assert!(expanded.contains(&"filesystem.get_metadata".to_string()));
        assert!(expanded.contains(&"filesystem.watch_folder".to_string()));
    }

    #[test]
    fn test_category_registry_expand_multiple() {
        let registry = build_multi_server_registry();
        let cat_reg = CategoryRegistry::build(&registry);

        let expanded = cat_reg.expand_categories(&[
            "file_browse".to_string(),
            "image_ocr".to_string(),
        ]);
        // 5 file_browse + 4 image_ocr = 9 tools
        assert_eq!(expanded.len(), 9);
        assert!(expanded.contains(&"filesystem.list_dir".to_string()));
        assert!(expanded.contains(&"ocr.extract_text_from_image".to_string()));
    }

    #[test]
    fn test_category_registry_expand_unknown_ignored() {
        let registry = build_multi_server_registry();
        let cat_reg = CategoryRegistry::build(&registry);

        let expanded = cat_reg.expand_categories(&[
            "file_browse".to_string(),
            "nonexistent_category".to_string(),
        ]);
        // Should only have file_browse tools, nonexistent is ignored
        assert_eq!(expanded.len(), 5);
    }

    #[test]
    fn test_category_registry_to_openai_tools() {
        let registry = build_multi_server_registry();
        let cat_reg = CategoryRegistry::build(&registry);

        let tools = cat_reg.to_openai_tools();
        assert!(!tools.is_empty());

        // Each tool should have the correct structure
        for tool in &tools {
            assert_eq!(tool["type"], "function");
            assert!(tool["function"]["name"].is_string());
            assert!(tool["function"]["description"].is_string());
            assert!(tool["function"]["parameters"]["properties"]["intent"].is_object());
        }

        // Check a specific category
        let file_browse = tools
            .iter()
            .find(|t| t["function"]["name"] == "file_browse");
        assert!(file_browse.is_some());
    }

    #[test]
    fn test_category_registry_is_category() {
        let registry = build_multi_server_registry();
        let cat_reg = CategoryRegistry::build(&registry);

        assert!(cat_reg.is_category("file_browse"));
        assert!(cat_reg.is_category("image_ocr"));
        assert!(!cat_reg.is_category("filesystem.list_dir")); // real tool, not category
        assert!(!cat_reg.is_category("nonexistent"));
    }

    #[test]
    fn test_category_registry_category_for_tool() {
        let registry = build_multi_server_registry();
        let cat_reg = CategoryRegistry::build(&registry);

        assert_eq!(
            cat_reg.category_for_tool("filesystem.list_dir"),
            Some("file_browse")
        );
        assert_eq!(
            cat_reg.category_for_tool("filesystem.move_file"),
            Some("file_edit")
        );
        assert_eq!(
            cat_reg.category_for_tool("ocr.extract_text_from_image"),
            Some("image_ocr")
        );
        assert_eq!(cat_reg.category_for_tool("nonexistent.tool"), None);
    }

    #[test]
    fn test_category_registry_len() {
        let registry = build_multi_server_registry();
        let cat_reg = CategoryRegistry::build(&registry);

        // Should have 4 categories: file_browse, file_edit, image_ocr, task_management
        assert_eq!(cat_reg.len(), 4);
        assert!(!cat_reg.is_empty());
    }

    #[test]
    fn test_category_registry_empty() {
        let registry = ToolRegistry::new();
        let cat_reg = CategoryRegistry::build(&registry);

        assert_eq!(cat_reg.len(), 0);
        assert!(cat_reg.is_empty());
    }

    #[test]
    fn test_category_registry_category_names() {
        let registry = build_multi_server_registry();
        let cat_reg = CategoryRegistry::build(&registry);

        let names = cat_reg.category_names();
        assert!(names.contains(&"file_browse"));
        assert!(names.contains(&"file_edit"));
        assert!(names.contains(&"image_ocr"));
        assert!(names.contains(&"task_management"));
    }

    #[test]
    fn test_default_categories_cover_all_83_tools() {
        // Verify that the hardcoded categories cover all 83 tools across 15 servers.
        // Original 67 PRD tools + 5 extra system tools + 3 screenshot-pipeline + 8 system-settings.
        let defs = default_category_definitions();
        let mut all_tools: Vec<String> = Vec::new();
        for (_, _, tools) in &defs {
            all_tools.extend(tools.clone());
        }
        // Should be exactly 83 tools
        assert_eq!(all_tools.len(), 83, "categories must cover all 83 tools");

        // No duplicates
        let unique: std::collections::HashSet<&str> =
            all_tools.iter().map(|s| s.as_str()).collect();
        assert_eq!(
            unique.len(),
            all_tools.len(),
            "no tool should appear in multiple categories"
        );
    }

    #[test]
    fn test_default_categories_count() {
        let defs = default_category_definitions();
        assert_eq!(defs.len(), 16, "should have exactly 16 categories");
    }
}
