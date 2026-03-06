//! MCP Client — JSON-RPC over stdio transport for MCP server management.
//!
//! This module handles:
//! - Spawning and managing MCP server child processes
//! - JSON-RPC 2.0 communication over process stdio
//! - Tool discovery and aggregation across all servers
//! - Tool call routing, validation, and execution
//! - Server lifecycle (start, restart with backoff, graceful shutdown)
//!
//! The MCP Client is used by the ToolRouter (WS-2D) to dispatch tool calls
//! from the LLM to the appropriate MCP server.

pub mod client;
pub mod discovery;
pub mod errors;
pub mod lifecycle;
pub mod registry;
pub mod transport;
pub mod types;

// Re-exports for convenience
pub use client::McpClient;
pub use errors::McpError;
pub use registry::{CategoryRegistry, ToolCategory, ToolRegistry, ToolResolution};
pub use types::{McpServersConfig, McpToolDefinition, ServerConfig, ToolCallResult};
