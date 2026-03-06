//! SSE streaming response parser for OpenAI-compatible chat completions.
//!
//! Reads a `reqwest::Response` as a byte stream, splits on SSE boundaries
//! (`data: …\n\n`), parses each chunk as JSON, and accumulates tool calls
//! across multiple deltas.

use futures::stream::{self, Stream, StreamExt};
use serde::Deserialize;
use uuid::Uuid;

use super::config::ToolCallFormat;
use super::errors::InferenceError;
use super::tool_call_parser::{
    parse_bracket_tool_calls, parse_native_json_tool_call, parse_pythonic_tool_calls,
};
use super::types::{ChatCompletionChunk, StreamChunk, ToolCall};

// ─── SSE line parser ─────────────────────────────────────────────────────────

/// Parse raw SSE bytes into `StreamChunk`s.
///
/// This is the main entry point for streaming. It:
/// 1. Splits the HTTP body into SSE events
/// 2. Parses each `data:` line as a `ChatCompletionChunk`
/// 3. Accumulates tool call fragments across deltas
/// 4. Emits complete `StreamChunk`s for each event
pub fn parse_sse_stream(
    response: reqwest::Response,
    tool_call_format: ToolCallFormat,
) -> impl Stream<Item = Result<StreamChunk, InferenceError>> {
    let byte_stream = response.bytes_stream();

    // Buffer for incomplete SSE lines across chunk boundaries
    let state = StreamState::new(tool_call_format);

    stream::unfold(
        (byte_stream, state, String::new()),
        |(mut byte_stream, mut state, mut buffer)| async move {
            loop {
                // Check if we have a complete SSE event in the buffer
                if let Some(event_end) = buffer.find("\n\n") {
                    let event = buffer[..event_end].to_string();
                    buffer = buffer[event_end + 2..].to_string();

                    match state.process_event(&event) {
                        Ok(Some(chunk)) => return Some((Ok(chunk), (byte_stream, state, buffer))),
                        Ok(None) => continue, // [DONE] or keep-alive
                        Err(e) => return Some((Err(e), (byte_stream, state, buffer))),
                    }
                }

                // Need more data from the stream
                match byte_stream.next().await {
                    Some(Ok(bytes)) => {
                        let text = String::from_utf8_lossy(&bytes);
                        buffer.push_str(&text);
                    }
                    Some(Err(e)) => {
                        return Some((
                            Err(InferenceError::StreamError {
                                reason: format!("stream read error: {e}"),
                            }),
                            (byte_stream, state, buffer),
                        ));
                    }
                    None => {
                        // Stream ended — check for any remaining buffer content
                        if !buffer.trim().is_empty() {
                            match state.process_event(buffer.trim()) {
                                Ok(Some(chunk)) => {
                                    buffer.clear();
                                    return Some((Ok(chunk), (byte_stream, state, buffer)));
                                }
                                Ok(None) => return None,
                                Err(e) => return Some((Err(e), (byte_stream, state, buffer))),
                            }
                        }
                        return None;
                    }
                }
            }
        },
    )
}

// ─── Stream State ────────────────────────────────────────────────────────────

/// Mutable state for accumulating tool call fragments across SSE events.
struct StreamState {
    tool_call_format: ToolCallFormat,
    /// Accumulated content for Pythonic format parsing.
    accumulated_content: String,
    /// In-progress tool calls (native_json): `(index, id, name, arguments_buffer)`.
    pending_tool_calls: Vec<(u32, Option<String>, String, String)>,
}

impl StreamState {
    fn new(tool_call_format: ToolCallFormat) -> Self {
        Self {
            tool_call_format,
            accumulated_content: String::new(),
            pending_tool_calls: Vec::new(),
        }
    }

    /// Process a single SSE event string (may contain multiple `data:` lines).
    fn process_event(&mut self, event: &str) -> Result<Option<StreamChunk>, InferenceError> {
        let mut data_content = String::new();

        for line in event.lines() {
            if let Some(data) = line.strip_prefix("data: ").or_else(|| line.strip_prefix("data:")) {
                let data = data.trim();
                if data == "[DONE]" {
                    // Stream complete — finalize any pending pythonic calls
                    return self.finalize();
                }
                data_content.push_str(data);
            }
            // Ignore non-data lines (comments, event types, etc.)
        }

        if data_content.is_empty() {
            return Ok(None); // Keep-alive or comment
        }

        let chunk: ChatCompletionChunk =
            serde_json::from_str(&data_content).map_err(|e| InferenceError::StreamError {
                reason: format!("failed to parse SSE chunk: {e} (data: {data_content})"),
            })?;

        self.process_chunk(chunk)
    }

    /// Process a parsed `ChatCompletionChunk`.
    fn process_chunk(
        &mut self,
        chunk: ChatCompletionChunk,
    ) -> Result<Option<StreamChunk>, InferenceError> {
        let choice = match chunk.choices.first() {
            Some(c) => c,
            None => return Ok(None),
        };

        let mut result = StreamChunk {
            token: None,
            tool_calls: None,
            finish_reason: choice.finish_reason.clone(),
        };

        // Handle text content — use only `content`, ignore `reasoning`.
        // Reasoning/thinking models (Qwen3, GPT-OSS) stream chain-of-thought
        // in `reasoning` and the actual answer in `content`. We only surface
        // `content` to the user; reasoning tokens are silently discarded.
        if let Some(ref content) = choice.delta.content {
            if !content.is_empty() {
                result.token = Some(content.clone());
                self.accumulated_content.push_str(content);
            }
        }

        // Handle native tool call deltas
        if let Some(ref tool_calls) = choice.delta.tool_calls {
            for tc in tool_calls {
                let index = tc.index.unwrap_or(0);

                // Find or create the pending tool call for this index
                let pending = self
                    .pending_tool_calls
                    .iter_mut()
                    .find(|(idx, _, _, _)| *idx == index);

                match pending {
                    Some((_, ref mut id, ref mut name, ref mut args)) => {
                        // Append to existing
                        if let Some(ref f) = tc.function {
                            if let Some(ref n) = f.name {
                                name.push_str(n);
                            }
                            if let Some(ref a) = f.arguments {
                                args.push_str(a);
                            }
                        }
                        if tc.id.is_some() {
                            *id = tc.id.clone();
                        }
                    }
                    None => {
                        // New tool call
                        let name = tc
                            .function
                            .as_ref()
                            .and_then(|f| f.name.clone())
                            .unwrap_or_default();
                        let args = tc
                            .function
                            .as_ref()
                            .and_then(|f| f.arguments.clone())
                            .unwrap_or_default();
                        self.pending_tool_calls
                            .push((index, tc.id.clone(), name, args));
                    }
                }
            }
        }

        // If finish_reason is "tool_calls" (native) or "stop" (might have text-based calls),
        // finalize the tool calls
        if let Some(ref reason) = result.finish_reason {
            if reason == "tool_calls" {
                result.tool_calls = Some(self.finalize_native_tool_calls()?);
            } else if reason == "stop" {
                // Check accumulated content for text-based tool call formats
                match self.tool_call_format {
                    ToolCallFormat::Pythonic => {
                        let calls = parse_pythonic_tool_calls(&self.accumulated_content)?;
                        if !calls.is_empty() {
                            result.tool_calls = Some(calls);
                            result.finish_reason = Some("tool_calls".into());
                        }
                    }
                    ToolCallFormat::Bracket => {
                        let calls = parse_bracket_tool_calls(&self.accumulated_content)?;
                        if !calls.is_empty() {
                            result.tool_calls = Some(calls);
                            result.finish_reason = Some("tool_calls".into());
                        }
                    }
                    ToolCallFormat::NativeJson => {} // Native handles via structured deltas
                }
            }
        }

        Ok(Some(result))
    }

    /// Finalize accumulated native JSON tool calls.
    fn finalize_native_tool_calls(&mut self) -> Result<Vec<ToolCall>, InferenceError> {
        let pending = std::mem::take(&mut self.pending_tool_calls);
        let mut calls = Vec::with_capacity(pending.len());

        for (_index, id, name, args) in pending {
            calls.push(parse_native_json_tool_call(id.as_deref(), &name, &args)?);
        }

        Ok(calls)
    }

    /// Finalize the stream — emit any remaining tool calls.
    fn finalize(&mut self) -> Result<Option<StreamChunk>, InferenceError> {
        // Check for pending native tool calls
        if !self.pending_tool_calls.is_empty() {
            let calls = self.finalize_native_tool_calls()?;
            return Ok(Some(StreamChunk {
                token: None,
                tool_calls: Some(calls),
                finish_reason: Some("tool_calls".into()),
            }));
        }

        // Check for text-based tool calls in accumulated content
        if !self.accumulated_content.is_empty() {
            let calls = match self.tool_call_format {
                ToolCallFormat::Pythonic => parse_pythonic_tool_calls(&self.accumulated_content)?,
                ToolCallFormat::Bracket => parse_bracket_tool_calls(&self.accumulated_content)?,
                ToolCallFormat::NativeJson => Vec::new(),
            };
            if !calls.is_empty() {
                return Ok(Some(StreamChunk {
                    token: None,
                    tool_calls: Some(calls),
                    finish_reason: Some("tool_calls".into()),
                }));
            }
        }

        Ok(None)
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Parse a non-streaming response body into tool calls and content.
///
/// Used for fallback when streaming is not supported by the endpoint.
pub fn parse_non_streaming_response(
    body: &str,
    format: ToolCallFormat,
) -> Result<StreamChunk, InferenceError> {
    #[derive(Deserialize)]
    struct NonStreamResponse {
        choices: Vec<NonStreamChoice>,
    }

    #[derive(Deserialize)]
    struct NonStreamChoice {
        message: NonStreamMessage,
        finish_reason: Option<String>,
    }

    #[derive(Deserialize)]
    struct NonStreamMessage {
        content: Option<String>,
        /// Reasoning/thinking content from models like Qwen3, GPT-OSS.
        /// Deserialized to prevent serde unknown-field errors, but not used —
        /// `content` holds the actual answer. See ADR comment on reasoning models.
        #[allow(dead_code)]
        reasoning: Option<String>,
        tool_calls: Option<Vec<NonStreamToolCall>>,
    }

    #[derive(Deserialize)]
    struct NonStreamToolCall {
        id: Option<String>,
        function: NonStreamFunction,
    }

    #[derive(Deserialize)]
    struct NonStreamFunction {
        name: String,
        arguments: String,
    }

    let resp: NonStreamResponse =
        serde_json::from_str(body).map_err(|e| InferenceError::StreamError {
            reason: format!("failed to parse non-streaming response: {e}"),
        })?;

    let choice = resp.choices.first().ok_or(InferenceError::StreamError {
        reason: "empty choices array".into(),
    })?;

    // Use `content` only. Reasoning/thinking models (Qwen3, GPT-OSS via Ollama)
    // put chain-of-thought in `reasoning` and the actual answer in `content`.
    // If `content` is empty (model exhausted max_tokens during reasoning), we
    // treat it as an empty response — the caller handles the retry/fallback.
    let content = choice.message.content.clone().filter(|c| !c.is_empty());

    // Check for native tool calls in the response
    let mut tool_calls = Vec::new();
    if let Some(ref tcs) = choice.message.tool_calls {
        for tc in tcs {
            let id = tc
                .id
                .clone()
                .unwrap_or_else(|| format!("call_{}", Uuid::new_v4()));
            let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                .map_err(|e| InferenceError::ToolCallParseError {
                    raw_response: tc.function.arguments.clone(),
                    reason: format!("invalid JSON: {e}"),
                })?;
            tool_calls.push(ToolCall {
                id,
                name: tc.function.name.clone(),
                arguments: args,
            });
        }
    }

    // Check for text-based tool calls in content (pythonic or bracket format)
    if tool_calls.is_empty() {
        if let Some(ref text) = content {
            let parsed = match format {
                ToolCallFormat::Pythonic => parse_pythonic_tool_calls(text)?,
                ToolCallFormat::Bracket => parse_bracket_tool_calls(text)?,
                ToolCallFormat::NativeJson => Vec::new(),
            };
            if !parsed.is_empty() {
                tool_calls = parsed;
            }
        }
    }

    let finish_reason = if !tool_calls.is_empty() {
        Some("tool_calls".into())
    } else {
        choice.finish_reason.clone()
    };

    Ok(StreamChunk {
        token: content,
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
        finish_reason,
    })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_non_streaming_with_content() {
        let body = r#"{
            "choices": [{
                "message": {"role": "assistant", "content": "Hello, world!"},
                "finish_reason": "stop"
            }]
        }"#;

        let chunk = parse_non_streaming_response(body, ToolCallFormat::NativeJson).unwrap();
        assert_eq!(chunk.token.as_deref(), Some("Hello, world!"));
        assert!(chunk.tool_calls.is_none());
        assert_eq!(chunk.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn test_parse_non_streaming_with_tool_calls() {
        let body = r#"{
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "filesystem.list_dir",
                            "arguments": "{\"path\": \"/tmp\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }"#;

        let chunk = parse_non_streaming_response(body, ToolCallFormat::NativeJson).unwrap();
        let calls = chunk.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "filesystem.list_dir");
        assert_eq!(calls[0].arguments["path"], "/tmp");
    }

    #[test]
    fn test_parse_non_streaming_pythonic() {
        let body = r#"{
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Let me check.\n\nTool: filesystem.list_dir\nArguments: {\"path\": \"/tmp\"}"
                },
                "finish_reason": "stop"
            }]
        }"#;

        let chunk = parse_non_streaming_response(body, ToolCallFormat::Pythonic).unwrap();
        let calls = chunk.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "filesystem.list_dir");
        assert_eq!(chunk.finish_reason.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn test_parse_non_streaming_empty_choices() {
        let body = r#"{"choices": []}"#;
        let result = parse_non_streaming_response(body, ToolCallFormat::NativeJson);
        assert!(result.is_err());
    }

    /// Reasoning models (Qwen3, GPT-OSS via Ollama) return both `content` and
    /// `reasoning`. We use `content` only — `reasoning` is chain-of-thought.
    #[test]
    fn test_parse_non_streaming_reasoning_model_uses_content() {
        let body = r#"{
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "{\"needs_tools\":true,\"steps\":[]}",
                    "reasoning": "Let me think about this task... The user wants..."
                },
                "finish_reason": "stop"
            }]
        }"#;

        let chunk = parse_non_streaming_response(body, ToolCallFormat::NativeJson).unwrap();
        assert_eq!(
            chunk.token.as_deref(),
            Some("{\"needs_tools\":true,\"steps\":[]}"),
            "should use content, not reasoning"
        );
    }

    /// When a reasoning model exhausts max_tokens during thinking, `content` is
    /// empty and `reasoning` has the chain-of-thought. We return None for content
    /// (not the reasoning text) so callers can handle the incomplete response.
    #[test]
    fn test_parse_non_streaming_reasoning_model_empty_content() {
        let body = r#"{
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning": "Let me think step by step about this..."
                },
                "finish_reason": "length"
            }]
        }"#;

        let chunk = parse_non_streaming_response(body, ToolCallFormat::NativeJson).unwrap();
        assert!(
            chunk.token.is_none(),
            "empty content should be None, not fallback to reasoning"
        );
        assert_eq!(chunk.finish_reason.as_deref(), Some("length"));
    }

    /// Deserialization should not fail when `reasoning` field is present.
    #[test]
    fn test_parse_non_streaming_reasoning_field_deserialized() {
        let body = r#"{
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                    "reasoning": "Quick response needed."
                },
                "finish_reason": "stop"
            }]
        }"#;

        let chunk = parse_non_streaming_response(body, ToolCallFormat::NativeJson).unwrap();
        assert_eq!(chunk.token.as_deref(), Some("Hello!"));
    }

    #[test]
    fn test_parse_non_streaming_bracket() {
        let body = r#"{
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "I'll list the directory.\n\n[filesystem.list_dir(path=\"/tmp\")]"
                },
                "finish_reason": "stop"
            }]
        }"#;

        let chunk = parse_non_streaming_response(body, ToolCallFormat::Bracket).unwrap();
        let calls = chunk.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "filesystem.list_dir");
        assert_eq!(calls[0].arguments["path"], "/tmp");
        assert_eq!(chunk.finish_reason.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn test_parse_non_streaming_bracket_special_tokens() {
        let body = r#"{
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "<|tool_call_start|>[filesystem.search_files(pattern=\"*.pdf\", path=\"/home\")]<|tool_call_end|>"
                },
                "finish_reason": "stop"
            }]
        }"#;

        let chunk = parse_non_streaming_response(body, ToolCallFormat::Bracket).unwrap();
        let calls = chunk.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "filesystem.search_files");
        assert_eq!(calls[0].arguments["pattern"], "*.pdf");
    }
}
