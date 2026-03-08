//! Inference error types.
//!
//! All errors implement `std::error::Error` via `thiserror`. Structured logging
//! is the caller's responsibility — these types carry the context needed to build
//! meaningful log entries.

use thiserror::Error;

/// Errors that can occur during inference operations.
#[derive(Debug, Error)]
pub enum InferenceError {
    /// TCP/HTTP connection to the model endpoint failed.
    #[error("connection failed to {endpoint}: {reason}")]
    ConnectionFailed {
        endpoint: String,
        reason: String,
    },

    /// The model endpoint did not respond within the configured timeout.
    #[error("inference timeout after {duration_secs}s")]
    Timeout {
        duration_secs: u64,
    },

    /// Failed to parse a tool call from the model's response.
    #[error("tool call parse error: {reason}")]
    ToolCallParseError {
        raw_response: String,
        reason: String,
    },

    /// The model returned a tool name that is not in the registry.
    #[error("unknown tool: {name}")]
    UnknownTool {
        name: String,
    },

    /// Every model in the fallback chain was unavailable.
    #[error("all models unavailable (tried: {})", attempted.join(", "))]
    AllModelsUnavailable {
        attempted: Vec<String>,
    },

    /// Non-2xx HTTP response from the model endpoint.
    #[error("HTTP {status}: {body}")]
    HttpError {
        status: u16,
        body: String,
    },

    /// SSE stream parsing or chunk-level error.
    #[error("stream error: {reason}")]
    StreamError {
        reason: String,
    },

    /// Configuration loading or validation error.
    #[error("config error: {reason}")]
    ConfigError {
        reason: String,
    },
}

impl InferenceError {
    /// Check if this error is an Ollama tool call parse failure (HTTP 500).
    ///
    /// Ollama returns HTTP 500 with `"error parsing tool call"` when the model
    /// generates malformed JSON in tool call arguments. These errors are
    /// candidates for client-side JSON repair.
    pub fn is_tool_call_parse_error(&self) -> bool {
        matches!(
            self,
            InferenceError::HttpError { status: 500, body }
                if body.contains("error parsing tool call")
        )
    }

    /// Extract the error body text, if this is an `HttpError`.
    pub fn error_body(&self) -> Option<&str> {
        match self {
            InferenceError::HttpError { body, .. } => Some(body),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_tool_call_parse_error_true() {
        let err = InferenceError::HttpError {
            status: 500,
            body: r#"{"error":{"message":"error parsing tool call: raw='{...}', err=invalid"}}"#
                .to_string(),
        };
        assert!(err.is_tool_call_parse_error());
    }

    #[test]
    fn test_is_tool_call_parse_error_false_different_body() {
        let err = InferenceError::HttpError {
            status: 500,
            body: "internal server error".to_string(),
        };
        assert!(!err.is_tool_call_parse_error());
    }

    #[test]
    fn test_is_tool_call_parse_error_false_different_status() {
        let err = InferenceError::HttpError {
            status: 404,
            body: "error parsing tool call".to_string(),
        };
        assert!(!err.is_tool_call_parse_error());
    }

    #[test]
    fn test_error_body_http_error() {
        let err = InferenceError::HttpError {
            status: 500,
            body: "test body".to_string(),
        };
        assert_eq!(err.error_body(), Some("test body"));
    }

    #[test]
    fn test_error_body_non_http() {
        let err = InferenceError::Timeout { duration_secs: 5 };
        assert!(err.error_body().is_none());
    }
}
