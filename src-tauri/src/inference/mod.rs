//! Inference Client — OpenAI-compatible API client for local LLM inference.
//!
//! This module handles all communication with the local model endpoint:
//! - Streaming and non-streaming chat completions
//! - Tool call parsing (native JSON + Pythonic formats)
//! - SSE stream parsing
//! - Fallback chain management
//! - Model configuration loading from `_models/config.yaml`
//!
//! The client speaks the OpenAI Chat Completions API, making the model
//! interchangeable via config. Switching from Qwen to LFM2.5 is a config
//! change, not a code change.

pub mod client;
pub mod config;
pub mod errors;
pub mod streaming;
pub mod tool_call_parser;
pub mod types;

// Re-exports for convenience
pub use client::InferenceClient;
pub use config::{ModelConfig, ModelsConfig, ToolCallFormat};
pub use errors::InferenceError;
pub use types::{ChatMessage, Role, SamplingOverrides, StreamChunk, ToolCall, ToolDefinition};
