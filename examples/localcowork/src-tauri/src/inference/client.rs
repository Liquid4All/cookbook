//! OpenAI-compatible inference client.
//!
//! Sends chat completion requests to a local LLM endpoint and streams back
//! tokens and tool calls. Handles the fallback chain when the primary model
//! is unavailable.

use std::time::Duration;

use futures::future::Either;
use futures::Stream;
use reqwest::Client as HttpClient;
use uuid::Uuid;

use super::config::{ModelConfig, ModelsConfig, ToolCallFormat};
use super::errors::InferenceError;
use super::streaming::{parse_non_streaming_response, parse_sse_stream};
use super::tool_call_parser::{extract_tool_call_from_error, repair_malformed_tool_call_json};
use super::types::{
    ChatCompletionRequest, ChatMessage, SamplingOverrides, StreamChunk, ToolCall, ToolDefinition,
};

// ─── Constants ───────────────────────────────────────────────────────────────

/// TCP connection timeout.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// Total request timeout for non-streaming calls.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Total request timeout for streaming calls.
///
/// Streaming responses from local models can take a long time, especially
/// when the context window is large (18+ messages). The model needs time
/// to process the full context before emitting the first token. A 30s
/// timeout causes silent stream termination that looks like "empty response"
/// to the agent loop.
const STREAM_REQUEST_TIMEOUT: Duration = Duration::from_secs(180);

// ─── InferenceClient ─────────────────────────────────────────────────────────

/// Client for the local LLM inference endpoint.
///
/// Created from `ModelsConfig` and holds the current model configuration.
/// Provides streaming and non-streaming chat completion methods.
pub struct InferenceClient {
    /// HTTP client for non-streaming requests (30s timeout).
    http: HttpClient,
    /// HTTP client for streaming requests (180s timeout).
    http_stream: HttpClient,
    /// The full models configuration (for fallback chain).
    config: ModelsConfig,
    /// The current model key (e.g., "qwen25-32b").
    current_model_key: String,
    /// The current model configuration.
    current_model: ModelConfig,
    /// Models that have already been tried and failed.
    exhausted_models: Vec<String>,
}

impl InferenceClient {
    /// Create a new inference client from the models configuration.
    ///
    /// Resolves the active model from config. Does NOT check connectivity —
    /// that happens on the first request.
    pub fn from_config(config: ModelsConfig) -> Result<Self, InferenceError> {
        let (key, model) = super::config::resolve_active_model(&config)?;

        let http = HttpClient::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map_err(|e| InferenceError::ConnectionFailed {
                endpoint: model.base_url.clone(),
                reason: format!("failed to build HTTP client: {e}"),
            })?;

        let http_stream = HttpClient::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(STREAM_REQUEST_TIMEOUT)
            .build()
            .map_err(|e| InferenceError::ConnectionFailed {
                endpoint: model.base_url.clone(),
                reason: format!("failed to build streaming HTTP client: {e}"),
            })?;

        Ok(Self {
            http,
            http_stream,
            config,
            current_model_key: key,
            current_model: model,
            exhausted_models: Vec::new(),
        })
    }

    /// Create an inference client targeting a specific model by key.
    ///
    /// Unlike [`from_config`] which resolves the active model + fallback chain,
    /// this constructor pins the client to a specific model. Used by the
    /// orchestrator (ADR-009) to create separate planner and router clients.
    pub fn from_config_with_model(
        config: ModelsConfig,
        model_key: &str,
    ) -> Result<Self, InferenceError> {
        let model = config
            .models
            .get(model_key)
            .ok_or_else(|| InferenceError::ConfigError {
                reason: format!("model '{model_key}' not found in config"),
            })?
            .clone();

        let http = HttpClient::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map_err(|e| InferenceError::ConnectionFailed {
                endpoint: model.base_url.clone(),
                reason: format!("failed to build HTTP client: {e}"),
            })?;

        let http_stream = HttpClient::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(STREAM_REQUEST_TIMEOUT)
            .build()
            .map_err(|e| InferenceError::ConnectionFailed {
                endpoint: model.base_url.clone(),
                reason: format!("failed to build streaming HTTP client: {e}"),
            })?;

        Ok(Self {
            http,
            http_stream,
            config,
            current_model_key: model_key.to_string(),
            current_model: model,
            exhausted_models: Vec::new(),
        })
    }

    /// The base URL of the current model's endpoint.
    pub fn current_base_url(&self) -> &str {
        &self.current_model.base_url
    }

    /// The name of the currently selected model.
    pub fn current_model_name(&self) -> &str {
        &self.current_model.display_name
    }

    /// The tool call format of the current model.
    pub fn tool_call_format(&self) -> ToolCallFormat {
        self.current_model.tool_call_format
    }

    /// The context window size of the current model.
    pub fn context_window(&self) -> u32 {
        self.current_model.context_window
    }

    // ─── Chat Completion (streaming) ─────────────────────────────────────

    /// Send a streaming chat completion request.
    ///
    /// Returns a `Stream` of `StreamChunk`s. Each chunk contains either a
    /// text token, tool calls, or both.
    ///
    /// If the current model is unavailable, automatically tries the fallback
    /// chain before returning an error. When Ollama returns HTTP 500 due to
    /// malformed JSON in tool call arguments, attempts client-side repair
    /// before triggering the fallback chain.
    pub async fn chat_completion_stream(
        &mut self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<ToolDefinition>>,
        sampling: Option<SamplingOverrides>,
    ) -> Result<
        impl Stream<Item = Result<StreamChunk, InferenceError>>,
        InferenceError,
    > {
        let mut last_error: Option<InferenceError> = None;

        for _attempt in 0..=self.remaining_fallbacks() {
            match self.try_stream_request(&messages, &tools, sampling.as_ref()).await {
                Ok(stream) => return Ok(Either::Left(stream)),
                Err(e) if e.is_tool_call_parse_error() => {
                    // Ollama returned HTTP 500 because the model generated
                    // malformed JSON in tool call arguments. Try to repair the
                    // JSON client-side before falling back to the next model.
                    if let Some(repaired) = Self::try_repair_from_error(&e) {
                        tracing::info!(
                            tool = %repaired.tool_calls.as_ref()
                                .and_then(|tc| tc.first())
                                .map(|tc| tc.name.as_str())
                                .unwrap_or("unknown"),
                            "repaired malformed JSON tool call"
                        );
                        return Ok(Either::Right(futures::stream::once(
                            async { Ok(repaired) },
                        )));
                    }
                    // Repair failed — continue to fallback chain
                    tracing::warn!("tool call JSON repair failed, falling back");
                    last_error = Some(e);
                    if self.try_next_fallback().is_err() {
                        break;
                    }
                }
                Err(e) if Self::is_retriable(&e) => {
                    last_error = Some(e);
                    if self.try_next_fallback().is_err() {
                        break; // No more fallbacks
                    }
                }
                Err(e) => return Err(e), // Non-retriable error
            }
        }

        Err(last_error.unwrap_or(InferenceError::AllModelsUnavailable {
            attempted: self.exhausted_models.clone(),
        }))
    }

    /// Attempt a single streaming request to the current model.
    async fn try_stream_request(
        &self,
        messages: &[ChatMessage],
        tools: &Option<Vec<ToolDefinition>>,
        sampling: Option<&SamplingOverrides>,
    ) -> Result<impl Stream<Item = Result<StreamChunk, InferenceError>>, InferenceError> {
        let url = format!("{}/chat/completions", self.current_model.base_url);
        let model_name = self
            .current_model
            .model_name
            .clone()
            .unwrap_or_else(|| self.current_model_key.clone());

        let temperature = sampling
            .and_then(|s| s.temperature)
            .unwrap_or(self.current_model.temperature);
        let top_p = sampling.and_then(|s| s.top_p);

        // Enable JSON response format when the model config opts in AND
        // tools are present. This sends `response_format: {"type":"json_object"}`
        // which triggers Ollama's GBNF grammar enforcement for valid JSON output.
        let response_format = if self.current_model.force_json_response && tools.is_some() {
            Some(super::types::ResponseFormat {
                r#type: "json_object".to_string(),
            })
        } else {
            None
        };

        let body = ChatCompletionRequest {
            model: model_name,
            messages: messages.to_vec(),
            tools: tools.clone(),
            tool_choice: tools.as_ref().map(|_| "auto".to_string()),
            temperature,
            top_p,
            max_tokens: self.current_model.max_tokens,
            stream: true,
            response_format,
        };

        // Log the request metadata (not the full body — it can be huge)
        tracing::info!(
            url = %url,
            model = %body.model,
            message_count = body.messages.len(),
            has_tools = body.tools.is_some(),
            tool_count = body.tools.as_ref().map(|t| t.len()).unwrap_or(0),
            max_tokens = body.max_tokens,
            stream = body.stream,
            "=== LLM REQUEST ==="
        );

        let response = self
            .http_stream
            .post(&url)
            .json(&body)
            .header("Accept", "text/event-stream")
            .send()
            .await
            .map_err(|e| {
                if e.is_connect() {
                    InferenceError::ConnectionFailed {
                        endpoint: url.clone(),
                        reason: e.to_string(),
                    }
                } else if e.is_timeout() {
                    InferenceError::Timeout { duration_secs: 5 }
                } else {
                    InferenceError::ConnectionFailed {
                        endpoint: url.clone(),
                        reason: e.to_string(),
                    }
                }
            })?;

        let status = response.status();
        if !status.is_success() {
            let body_text = response.text().await.unwrap_or_default();
            return Err(InferenceError::HttpError {
                status: status.as_u16(),
                body: body_text,
            });
        }

        Ok(parse_sse_stream(response, self.current_model.tool_call_format))
    }

    // ─── Tool Call Repair ──────────────────────────────────────────────────

    /// Attempt to repair a malformed tool call from an Ollama HTTP 500 error.
    ///
    /// Extracts the raw JSON from the error body, applies repair heuristics,
    /// and builds a synthetic `StreamChunk` with the repaired tool call.
    /// Returns `None` if the error body doesn't match or repair fails.
    fn try_repair_from_error(err: &InferenceError) -> Option<StreamChunk> {
        let body = err.error_body()?;
        let (_tool_name, raw_args) = extract_tool_call_from_error(body)?;
        let repaired_args = repair_malformed_tool_call_json(&raw_args)?;

        // Build a synthetic tool call. The tool name is empty because
        // Ollama's error body doesn't include it — the agent loop resolves
        // the name from the conversation context (the model declared intent
        // before Ollama attempted to parse the arguments).
        let tool_call = ToolCall {
            id: format!("call_{}", Uuid::new_v4()),
            name: _tool_name,
            arguments: repaired_args,
        };

        Some(StreamChunk {
            token: None,
            tool_calls: Some(vec![tool_call]),
            finish_reason: Some("tool_calls".to_string()),
        })
    }

    // ─── Chat Completion (non-streaming) ─────────────────────────────────

    /// Send a non-streaming chat completion request.
    ///
    /// Returns a single `StreamChunk` with the complete response.
    pub async fn chat_completion(
        &mut self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<ToolDefinition>>,
        sampling: Option<SamplingOverrides>,
    ) -> Result<StreamChunk, InferenceError> {
        let url = format!("{}/chat/completions", self.current_model.base_url);
        let model_name = self
            .current_model
            .model_name
            .clone()
            .unwrap_or_else(|| self.current_model_key.clone());

        let temperature = sampling
            .as_ref()
            .and_then(|s| s.temperature)
            .unwrap_or(self.current_model.temperature);
        let top_p = sampling.as_ref().and_then(|s| s.top_p);

        let response_format = if self.current_model.force_json_response && tools.is_some() {
            Some(super::types::ResponseFormat {
                r#type: "json_object".to_string(),
            })
        } else {
            None
        };

        let body = ChatCompletionRequest {
            model: model_name,
            messages,
            tools: tools.clone(),
            tool_choice: tools.as_ref().map(|_| "auto".to_string()),
            temperature,
            top_p,
            max_tokens: self.current_model.max_tokens,
            stream: false,
            response_format,
        };

        let response = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| InferenceError::ConnectionFailed {
                endpoint: url.clone(),
                reason: e.to_string(),
            })?;

        let status = response.status();
        if !status.is_success() {
            let body_text = response.text().await.unwrap_or_default();
            return Err(InferenceError::HttpError {
                status: status.as_u16(),
                body: body_text,
            });
        }

        let body_text = response.text().await.map_err(|e| InferenceError::StreamError {
            reason: format!("failed to read response body: {e}"),
        })?;

        parse_non_streaming_response(&body_text, self.current_model.tool_call_format)
    }

    // ─── Health Check ────────────────────────────────────────────────────

    /// Check if the current model endpoint is reachable.
    ///
    /// Sends a lightweight request to verify connectivity. Does not consume
    /// inference tokens.
    pub async fn health_check(&self) -> Result<bool, InferenceError> {
        let url = format!("{}/models", self.current_model.base_url);

        match self.http.get(&url).timeout(CONNECT_TIMEOUT).send().await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    // ─── Fallback Chain ──────────────────────────────────────────────────

    /// Move to the next model in the fallback chain.
    ///
    /// Returns `Err` if no more fallbacks are available.
    pub fn try_next_fallback(&mut self) -> Result<(), InferenceError> {
        self.exhausted_models.push(self.current_model_key.clone());

        for key in &self.config.fallback_chain {
            if self.exhausted_models.contains(key) || key == "static_response" {
                continue;
            }
            if let Some(model) = self.config.models.get(key) {
                self.current_model_key = key.clone();
                self.current_model = model.clone();
                return Ok(());
            }
        }

        Err(InferenceError::AllModelsUnavailable {
            attempted: self.exhausted_models.clone(),
        })
    }

    /// Number of remaining fallback models.
    fn remaining_fallbacks(&self) -> usize {
        self.config
            .fallback_chain
            .iter()
            .filter(|k| !self.exhausted_models.contains(k) && k.as_str() != "static_response")
            .count()
    }

    /// Whether an error should trigger a fallback attempt.
    ///
    /// HTTP 404 is included because Ollama returns 404 when a model isn't
    /// pulled/installed — the next model in the chain may still be available.
    ///
    /// HTTP 500 is included because local model servers (Ollama, llama.cpp)
    /// return 500 when the model generates malformed JSON in tool call
    /// arguments — this is a transient model error, not a permanent server
    /// failure. Retrying (or falling back) is the correct behavior.
    fn is_retriable(err: &InferenceError) -> bool {
        matches!(
            err,
            InferenceError::ConnectionFailed { .. }
                | InferenceError::Timeout { .. }
                | InferenceError::HttpError { status: 404, .. }
                | InferenceError::HttpError { status: 500, .. }
                | InferenceError::HttpError { status: 502..=504, .. }
        )
    }
}

// ─── Static Response Fallback ────────────────────────────────────────────────

/// Generate the static response used when all models are unavailable.
pub fn static_fallback_response() -> StreamChunk {
    StreamChunk {
        token: Some(
            "The model server is not running. \
             Start it with: ./scripts/start-model.sh\n\n\
             If using Ollama instead, run: ollama serve"
                .to_string(),
        ),
        tool_calls: None,
        finish_reason: Some("stop".to_string()),
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn test_config() -> ModelsConfig {
        let mut models = HashMap::new();
        models.insert(
            "model-a".to_string(),
            ModelConfig {
                display_name: "Model A".to_string(),
                runtime: "ollama".to_string(),
                model_name: Some("model-a:latest".to_string()),
                model_path: None,
                base_url: "http://localhost:11111/v1".to_string(),
                context_window: 4096,
                tool_call_format: ToolCallFormat::NativeJson,
                temperature: 0.7,
                max_tokens: 1024,
                estimated_vram_gb: None,
                capabilities: vec!["text".to_string()],
                force_json_response: false,
                role: None,
            },
        );
        models.insert(
            "model-b".to_string(),
            ModelConfig {
                display_name: "Model B".to_string(),
                runtime: "ollama".to_string(),
                model_name: Some("model-b:latest".to_string()),
                model_path: None,
                base_url: "http://localhost:22222/v1".to_string(),
                context_window: 8192,
                tool_call_format: ToolCallFormat::Pythonic,
                temperature: 0.5,
                max_tokens: 2048,
                estimated_vram_gb: None,
                capabilities: vec!["text".to_string()],
                force_json_response: false,
                role: None,
            },
        );

        ModelsConfig {
            active_model: "model-a".to_string(),
            models_dir: None,
            models,
            fallback_chain: vec![
                "model-a".to_string(),
                "model-b".to_string(),
                "static_response".to_string(),
            ],
            orchestrator: None,
            two_pass_tool_selection: None,
            enabled_servers: None,
            enabled_tools: None,
        }
    }

    #[test]
    fn test_from_config_selects_active_model() {
        let client = InferenceClient::from_config(test_config()).unwrap();
        assert_eq!(client.current_model_key, "model-a");
        assert_eq!(client.current_model_name(), "Model A");
    }

    #[test]
    fn test_fallback_chain() {
        let mut client = InferenceClient::from_config(test_config()).unwrap();
        assert_eq!(client.current_model_key, "model-a");

        // Fallback to model-b
        client.try_next_fallback().unwrap();
        assert_eq!(client.current_model_key, "model-b");
        assert_eq!(client.tool_call_format(), ToolCallFormat::Pythonic);

        // No more fallbacks
        let result = client.try_next_fallback();
        assert!(result.is_err());
    }

    #[test]
    fn test_remaining_fallbacks() {
        let client = InferenceClient::from_config(test_config()).unwrap();
        // model-a (current, in chain) + model-b = 2 remaining
        assert_eq!(client.remaining_fallbacks(), 2);
    }

    #[test]
    fn test_is_retriable() {
        assert!(InferenceClient::is_retriable(
            &InferenceError::ConnectionFailed {
                endpoint: "".into(),
                reason: "".into()
            }
        ));
        assert!(InferenceClient::is_retriable(&InferenceError::Timeout {
            duration_secs: 5
        }));
        assert!(InferenceClient::is_retriable(&InferenceError::HttpError {
            status: 404,
            body: "model not found".into()
        }));
        assert!(InferenceClient::is_retriable(&InferenceError::HttpError {
            status: 500,
            body: "malformed JSON".into()
        }));
        assert!(InferenceClient::is_retriable(&InferenceError::HttpError {
            status: 503,
            body: "".into()
        }));
        assert!(!InferenceClient::is_retriable(
            &InferenceError::HttpError {
                status: 400,
                body: "".into()
            }
        ));
        assert!(!InferenceClient::is_retriable(
            &InferenceError::ToolCallParseError {
                raw_response: "".into(),
                reason: "".into()
            }
        ));
    }

    #[test]
    fn test_static_fallback_response() {
        let chunk = static_fallback_response();
        assert!(chunk.token.is_some());
        assert!(chunk.tool_calls.is_none());
        assert_eq!(chunk.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn test_try_repair_from_error_success() {
        // Simulate the exact Ollama HTTP 500 error with malformed JSON
        let err = InferenceError::HttpError {
            status: 500,
            body: r#"{"error":{"message":"error parsing tool call: raw='{\"create_dirs\":true,\"destination\":\"\"/Users/chintan/Desktop/file.png\",\"source\":\"/tmp/file.png\"}', err=invalid character '/' after object key:value pair"}}"#.to_string(),
        };

        let result = InferenceClient::try_repair_from_error(&err);
        assert!(result.is_some(), "should repair the malformed JSON");

        let chunk = result.unwrap();
        assert!(chunk.token.is_none());
        assert!(chunk.tool_calls.is_some());
        assert_eq!(chunk.finish_reason.as_deref(), Some("tool_calls"));

        let calls = chunk.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["destination"], "/Users/chintan/Desktop/file.png");
        assert_eq!(calls[0].arguments["create_dirs"], true);
    }

    #[test]
    fn test_try_repair_from_error_non_tool_call_error() {
        // A regular HTTP 500 that isn't a tool call parse error
        let err = InferenceError::HttpError {
            status: 500,
            body: "internal server error".to_string(),
        };
        assert!(InferenceClient::try_repair_from_error(&err).is_none());
    }

    #[test]
    fn test_try_repair_from_error_non_http_error() {
        let err = InferenceError::Timeout { duration_secs: 30 };
        assert!(InferenceClient::try_repair_from_error(&err).is_none());
    }
}
