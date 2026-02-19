//! Shared types for the inference client.
//!
//! These mirror the OpenAI Chat Completions API types, used for both
//! request building and response parsing.

use serde::{Deserialize, Serialize};

// ─── Request Types ───────────────────────────────────────────────────────────

/// A single message in the conversation.
///
/// Serialization notes for OpenAI-compatible local models:
/// - `content` must be `""` (not `null`) for assistant messages with tool calls.
///   Many local models (Ollama, llama.cpp) misinterpret `null` content and fail
///   to recognize the tool call round-trip pattern.
/// - `tool_call_id` and `tool_calls` are skipped when `None`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    #[serde(serialize_with = "serialize_content")]
    pub content: Option<String>,
    /// Tool call results are sent back as `tool` role messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Assistant messages may contain tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallResponse>>,
}

/// Custom serializer for `content`: emit `""` instead of `null` when `None`.
///
/// OpenAI's API accepts `null` content, but many local LLM runtimes
/// (Ollama, llama.cpp, vLLM) reject or mishandle `null` content fields.
/// Using `""` (empty string) is universally safe.
fn serialize_content<S>(value: &Option<String>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match value {
        Some(s) => serializer.serialize_str(s),
        None => serializer.serialize_str(""),
    }
}

/// Message role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// Tool definition sent in the request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub r#type: String,
    pub function: FunctionDefinition,
}

/// Function definition within a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Structured output format hint for the model.
///
/// When set to `json_object`, instructs Ollama to use GBNF grammar
/// enforcement to guarantee valid JSON output. This is opt-in and
/// experimental — only enable after live testing with the target model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFormat {
    /// The format type. Currently only `"json_object"` is supported.
    pub r#type: String,
}

/// Request body for `POST /v1/chat/completions`.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    pub temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    pub max_tokens: u32,
    pub stream: bool,
    /// Optional structured output format. When set, the model backend
    /// (Ollama/llama.cpp) uses grammar constraints to enforce valid output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
}

/// Optional sampling parameter overrides for a single inference call.
///
/// When provided, these override the model config defaults.
/// Used to lower temperature/top_p for tool-calling turns (more deterministic)
/// and raise them for conversational turns (more creative).
#[derive(Debug, Clone, Copy, Default)]
pub struct SamplingOverrides {
    /// Override temperature (0.0 = deterministic, 1.0 = creative).
    pub temperature: Option<f32>,
    /// Override top_p (nucleus sampling threshold).
    pub top_p: Option<f32>,
}

// ─── Response Types ──────────────────────────────────────────────────────────

/// A parsed tool call extracted from the model's response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique ID for this tool call (generated if the model doesn't provide one).
    pub id: String,
    /// Fully qualified tool name, e.g. `"filesystem.list_dir"`.
    pub name: String,
    /// Validated JSON arguments.
    pub arguments: serde_json::Value,
}

/// Tool call as returned in the OpenAI response format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResponse {
    pub id: String,
    pub r#type: String,
    pub function: FunctionCallResponse,
}

/// Function call details in a response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallResponse {
    pub name: String,
    pub arguments: String,
}

/// A single chunk from the streaming response.
#[derive(Debug, Clone)]
pub struct StreamChunk {
    /// Incremental text token (if this chunk carries text).
    pub token: Option<String>,
    /// Tool calls detected in this chunk (accumulated).
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Why the model stopped: `"stop"`, `"tool_calls"`, or `None` (still going).
    pub finish_reason: Option<String>,
}

/// Raw SSE chunk from the OpenAI API.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionChunk {
    #[allow(dead_code)]
    pub id: Option<String>,
    pub choices: Vec<ChunkChoice>,
}

/// A single choice within a streaming chunk.
#[derive(Debug, Clone, Deserialize)]
pub struct ChunkChoice {
    pub delta: ChunkDelta,
    pub finish_reason: Option<String>,
}

/// The delta (incremental update) within a chunk choice.
#[derive(Debug, Clone, Deserialize)]
pub struct ChunkDelta {
    #[serde(default)]
    pub content: Option<String>,
    /// Reasoning/thinking content from models like Qwen3 and GPT-OSS.
    /// Deserialized to prevent serde unknown-field errors, but not used for
    /// streaming output — `content` holds the actual answer after reasoning
    /// completes. Reasoning tokens are silently discarded.
    #[serde(default)]
    #[allow(dead_code)]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ChunkToolCall>>,
}

/// A tool call fragment within a streaming delta.
#[derive(Debug, Clone, Deserialize)]
pub struct ChunkToolCall {
    pub index: Option<u32>,
    pub id: Option<String>,
    pub function: Option<ChunkFunction>,
}

/// A function call fragment within a streaming tool call.
#[derive(Debug, Clone, Deserialize)]
pub struct ChunkFunction {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_p_omitted_when_none() {
        let req = ChatCompletionRequest {
            model: "test".to_string(),
            messages: vec![],
            tools: None,
            tool_choice: None,
            temperature: 0.7,
            top_p: None,
            max_tokens: 1024,
            stream: false,
            response_format: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("top_p"), "top_p should be omitted when None");
    }

    #[test]
    fn test_top_p_included_when_some() {
        let req = ChatCompletionRequest {
            model: "test".to_string(),
            messages: vec![],
            tools: None,
            tool_choice: None,
            temperature: 0.1,
            top_p: Some(0.2),
            max_tokens: 1024,
            stream: false,
            response_format: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"top_p\":0.2"), "top_p should appear in JSON when Some");
    }

    #[test]
    fn test_response_format_omitted_when_none() {
        let req = ChatCompletionRequest {
            model: "test".to_string(),
            messages: vec![],
            tools: None,
            tool_choice: None,
            temperature: 0.7,
            top_p: None,
            max_tokens: 1024,
            stream: false,
            response_format: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(
            !json.contains("response_format"),
            "response_format should be omitted when None"
        );
    }

    #[test]
    fn test_response_format_included_when_set() {
        let req = ChatCompletionRequest {
            model: "test".to_string(),
            messages: vec![],
            tools: None,
            tool_choice: None,
            temperature: 0.7,
            top_p: None,
            max_tokens: 1024,
            stream: false,
            response_format: Some(ResponseFormat {
                r#type: "json_object".to_string(),
            }),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(
            json.contains("\"response_format\""),
            "response_format should appear in JSON when Some"
        );
        assert!(
            json.contains("\"json_object\""),
            "type should be json_object"
        );
    }

    #[test]
    fn test_sampling_overrides_default() {
        let overrides = SamplingOverrides::default();
        assert!(overrides.temperature.is_none());
        assert!(overrides.top_p.is_none());
    }
}
