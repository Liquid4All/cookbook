//! RAG pre-filter for tool selection (ADR-010 Phase 2 / ADR-009).
//!
//! Embeds tool descriptions at startup via the router model's `/embeddings`
//! endpoint, then for each user query, embeds the query and selects the
//! top-K tools by cosine similarity.
//!
//! Caching: The embedding index can be cached to disk for fast startup.
//! Cache is invalidated when the model endpoint or tool list changes.
//!
//! Ported from TypeScript `tests/model-behavior/benchmark-lfm.ts` (lines 270-414).

use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[allow(unused_imports)]
use std::io::{self, Write};

// ─── Error Type ─────────────────────────────────────────────────────────────

/// Errors from the tool pre-filter.
#[derive(Debug, thiserror::Error)]
pub enum ToolPreFilterError {
    #[error("embedding request failed (HTTP {status}): {body}")]
    HttpError { status: u16, body: String },

    #[error("embedding request failed: {reason}")]
    RequestFailed { reason: String },

    #[error("empty embedding response for {count} inputs")]
    EmptyResponse { count: usize },

    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

// ─── Embedding Response Types ───────────────────────────────────────────────

/// Raw embedding item from the `/embeddings` endpoint.
#[derive(Debug, Deserialize)]
struct RawEmbeddingItem {
    index: usize,
    embedding: serde_json::Value, // number[] or number[][] (per-token)
}

/// Parsed embedding response (array of items).
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<RawEmbeddingItem>,
}

/// Metadata stored alongside the cached index for validation.
#[derive(Debug, Serialize, Deserialize)]
struct CacheMetadata {
    /// Endpoint used to build the index (e.g., "http://localhost:1234/v1")
    endpoint: String,
    /// Hash of tool names + descriptions to detect tool changes.
    tool_hash: String,
    /// Unix timestamp when the cache was created.
    created_at: u64,
    /// Embedding dimension.
    dimension: usize,
}

/// Full cache file contents.
#[derive(Debug, Serialize, Deserialize)]
struct CachedIndex {
    metadata: CacheMetadata,
    tool_names: Vec<String>,
    embeddings: Vec<Vec<f32>>,
}

// ─── Tool Embedding Index ───────────────────────────────────────────────────

/// Pre-computed tool embedding index for RAG pre-filtering.
///
/// Built once at orchestrator startup, reused for every query.
#[derive(Debug, Clone)]
pub struct ToolEmbeddingIndex {
    /// Tool names in registration order.
    tool_names: Vec<String>,
    /// L2-normalized embeddings, one per tool. Shape: `[n_tools][n_dim]`.
    embeddings: Vec<Vec<f32>>,
}

/// A scored tool result from the pre-filter.
#[derive(Debug, Clone, Serialize)]
pub struct ScoredTool {
    pub name: String,
    pub score: f32,
}

impl ToolEmbeddingIndex {
    /// Build the index by embedding all tool descriptions.
    ///
    /// Each tool is embedded as `"name: description"` text. The embeddings are
    /// mean-pooled (if per-token) and L2-normalized for cosine similarity.
    pub async fn build(
        endpoint: &str,
        tools: &[(String, String)], // (name, description) pairs
    ) -> Result<Self, ToolPreFilterError> {
        if tools.is_empty() {
            return Ok(Self {
                tool_names: Vec::new(),
                embeddings: Vec::new(),
            });
        }

        let texts: Vec<String> = tools
            .iter()
            .map(|(name, desc)| format!("{name}: {desc}"))
            .collect();

        let raw = embed_texts(endpoint, &texts).await?;
        let embeddings: Vec<Vec<f32>> = raw.into_iter().map(l2_normalize).collect();
        let tool_names: Vec<String> = tools.iter().map(|(n, _)| n.clone()).collect();

        Ok(Self {
            tool_names,
            embeddings,
        })
    }

    /// Build the index, trying to load from cache first.
    ///
    /// If a valid cached index exists (same endpoint + tools), loads it.
    /// Otherwise builds fresh and saves to cache.
    pub async fn build_with_cache(
        endpoint: &str,
        tools: &[(String, String)],
        cache_dir: Option<&Path>,
    ) -> Result<Self, ToolPreFilterError> {
        // Try to load from cache first
        if let Some(dir) = cache_dir {
            if let Ok(cached) = Self::load_from_cache(endpoint, tools, dir) {
                tracing::info!(
                    tool_count = cached.len(),
                    "orchestrator: tool index loaded from cache"
                );
                return Ok(cached);
            }
        }

        // Build fresh
        let index = Self::build(endpoint, tools).await?;

        // Save to cache
        if let Some(dir) = cache_dir {
            if let Err(e) = index.save_to_cache(endpoint, tools, dir) {
                tracing::warn!(error = %e, "failed to save tool index cache");
            } else {
                tracing::info!("orchestrator: tool index cached");
            }
        }

        Ok(index)
    }

    /// Get the cache file path for a given endpoint.
    fn cache_path(cache_dir: &Path, endpoint: &str) -> PathBuf {
        // Create a safe filename from the endpoint
        let safe_name = endpoint
            .replace("://", "_")
            .replace("/", "_")
            .replace(":", "_")
            .replace(".", "_");
        cache_dir.join(format!("tool_index_{}.json", safe_name))
    }

    /// Load the index from cache if valid.
    fn load_from_cache(
        endpoint: &str,
        tools: &[(String, String)],
        cache_dir: &Path,
    ) -> Result<Self, ToolPreFilterError> {
        let cache_path = Self::cache_path(cache_dir, endpoint);

        if !cache_path.exists() {
            return Err(ToolPreFilterError::RequestFailed {
                reason: "cache file not found".to_string(),
            });
        }

        let content = fs::read_to_string(&cache_path).map_err(|e| ToolPreFilterError::RequestFailed {
            reason: format!("failed to read cache: {}", e),
        })?;

        let cached: CachedIndex = serde_json::from_str(&content).map_err(|e| ToolPreFilterError::RequestFailed {
            reason: format!("failed to parse cache: {}", e),
        })?;

        // Validate endpoint
        if cached.metadata.endpoint != endpoint {
            return Err(ToolPreFilterError::RequestFailed {
                reason: "endpoint changed".to_string(),
            });
        }

        // Validate tool hash
        let current_hash = compute_tool_hash(tools);
        if cached.metadata.tool_hash != current_hash {
            return Err(ToolPreFilterError::RequestFailed {
                reason: "tools changed".to_string(),
            });
        }

        // Validate embeddings exist
        if cached.embeddings.is_empty() && !cached.tool_names.is_empty() {
            return Err(ToolPreFilterError::RequestFailed {
                reason: "cached embeddings are empty".to_string(),
            });
        }

        Ok(Self {
            tool_names: cached.tool_names,
            embeddings: cached.embeddings,
        })
    }

    /// Save the index to cache.
    fn save_to_cache(
        &self,
        endpoint: &str,
        tools: &[(String, String)],
        cache_dir: &Path,
    ) -> Result<(), ToolPreFilterError> {
        // Create cache directory if needed
        fs::create_dir_all(cache_dir).map_err(|e| ToolPreFilterError::RequestFailed {
            reason: format!("failed to create cache dir: {}", e),
        })?;

        let dimension = self.embeddings.first().map(|e| e.len()).unwrap_or(0);
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let cached = CachedIndex {
            metadata: CacheMetadata {
                endpoint: endpoint.to_string(),
                tool_hash: compute_tool_hash(tools),
                created_at,
                dimension,
            },
            tool_names: self.tool_names.clone(),
            embeddings: self.embeddings.clone(),
        };

        let content = serde_json::to_string_pretty(&cached).map_err(|e| ToolPreFilterError::RequestFailed {
            reason: format!("failed to serialize cache: {}", e),
        })?;

        let cache_path = Self::cache_path(cache_dir, endpoint);
        fs::write(&cache_path, content).map_err(|e| ToolPreFilterError::RequestFailed {
            reason: format!("failed to write cache: {}", e),
        })?;

        Ok(())
    }

    /// Select the top-K tool names by cosine similarity to the query.
    ///
    /// Returns `(selected_names, scored_tools)` where scored_tools is sorted
    /// descending by score for debugging.
    pub async fn filter(
        &self,
        endpoint: &str,
        query: &str,
        top_k: usize,
    ) -> Result<(Vec<String>, Vec<ScoredTool>), ToolPreFilterError> {
        if self.tool_names.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let raw_query = embed_texts(endpoint, &[query.to_string()]).await?;
        let query_emb = l2_normalize(
            raw_query
                .into_iter()
                .next()
                .ok_or(ToolPreFilterError::EmptyResponse { count: 1 })?,
        );

        // Score all tools
        let mut scored: Vec<ScoredTool> = self
            .tool_names
            .iter()
            .enumerate()
            .map(|(i, name)| ScoredTool {
                name: name.clone(),
                score: cosine_similarity(&query_emb, &self.embeddings[i]),
            })
            .collect();

        // Sort descending by score
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-K
        let k = top_k.min(scored.len());
        let selected: Vec<String> = scored[..k].iter().map(|s| s.name.clone()).collect();

        Ok((selected, scored))
    }

    /// Number of tools in the index.
    pub fn len(&self) -> usize {
        self.tool_names.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.tool_names.is_empty()
    }
}

// ─── Embedding Helpers ──────────────────────────────────────────────────────

/// Embed a batch of texts via the `/embeddings` endpoint.
///
/// The LFM2 `/embeddings` endpoint returns per-token embeddings (2D arrays).
/// We mean-pool them into a single vector per text.
async fn embed_texts(
    endpoint: &str,
    texts: &[String],
) -> Result<Vec<Vec<f32>>, ToolPreFilterError> {
    let http = HttpClient::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| ToolPreFilterError::RequestFailed {
            reason: format!("failed to build HTTP client: {e}"),
        })?;

    let url = format!("{endpoint}/embeddings");
    let body = serde_json::json!({ "input": texts });

    let response = http
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| ToolPreFilterError::RequestFailed {
            reason: format!("embedding request to {url}: {e}"),
        })?;

    if !response.status().is_success() {
        let status = response.status().as_u16();
        let body_text = response
            .text()
            .await
            .unwrap_or_else(|_| "unknown".to_string());
        return Err(ToolPreFilterError::HttpError {
            status,
            body: body_text,
        });
    }

    let result: EmbeddingResponse =
        response
            .json()
            .await
            .map_err(|e| ToolPreFilterError::RequestFailed {
                reason: format!("failed to parse embedding response: {e}"),
            })?;

    if result.data.is_empty() {
        return Err(ToolPreFilterError::EmptyResponse {
            count: texts.len(),
        });
    }

    // Sort by index to ensure order matches input
    let mut items = result.data;
    items.sort_by_key(|item| item.index);

    items
        .into_iter()
        .map(|item| mean_pool_embedding(&item.embedding))
        .collect()
}

/// Mean-pool per-token embeddings into a single vector.
///
/// If already pooled (1D array of numbers), returns as-is.
/// If 2D (per-token), averages across the token dimension.
fn mean_pool_embedding(embedding: &serde_json::Value) -> Result<Vec<f32>, ToolPreFilterError> {
    match embedding {
        serde_json::Value::Array(arr) if arr.is_empty() => Ok(Vec::new()),

        // 1D: already pooled — [f32, f32, ...]
        serde_json::Value::Array(arr) if arr[0].is_f64() => {
            Ok(arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
        }

        // 2D: per-token — [[f32, ...], [f32, ...], ...]
        serde_json::Value::Array(arr) if arr[0].is_array() => {
            let tokens: Vec<Vec<f32>> = arr
                .iter()
                .filter_map(|row| {
                    row.as_array().map(|r| {
                        r.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect()
                    })
                })
                .collect();

            if tokens.is_empty() {
                return Ok(Vec::new());
            }

            let n_tokens = tokens.len();
            let n_dim = tokens[0].len();
            let mut result = vec![0.0_f32; n_dim];

            for token in &tokens {
                for (d, val) in token.iter().enumerate() {
                    if d < n_dim {
                        result[d] += val;
                    }
                }
            }

            for val in &mut result {
                *val /= n_tokens as f32;
            }

            Ok(result)
        }

        _ => Err(ToolPreFilterError::RequestFailed {
            reason: "unexpected embedding format (expected number[] or number[][])".to_string(),
        }),
    }
}

/// L2-normalize a vector. Returns the normalized copy.
fn l2_normalize(vec: Vec<f32>) -> Vec<f32> {
    let norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        vec.into_iter().map(|v| v / norm).collect()
    } else {
        vec
    }
}

/// Cosine similarity between two L2-normalized vectors (= dot product).
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute a hash of the tool list for cache validation.
///
/// Uses a simple FNV-like hash for speed.
fn compute_tool_hash(tools: &[(String, String)]) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    for (name, desc) in tools {
        name.hash(&mut hasher);
        desc.hash(&mut hasher);
    }
    format!("{:x}", hasher.finish())
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_vectors() {
        let v = l2_normalize(vec![1.0, 2.0, 3.0]);
        let score = cosine_similarity(&v, &v);
        assert!((score - 1.0).abs() < 1e-5, "identical vectors should have similarity ~1.0");
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = l2_normalize(vec![1.0, 0.0, 0.0]);
        let b = l2_normalize(vec![0.0, 1.0, 0.0]);
        let score = cosine_similarity(&a, &b);
        assert!(score.abs() < 1e-5, "orthogonal vectors should have similarity ~0.0");
    }

    #[test]
    fn cosine_opposite_vectors() {
        let a = l2_normalize(vec![1.0, 0.0]);
        let b = l2_normalize(vec![-1.0, 0.0]);
        let score = cosine_similarity(&a, &b);
        assert!((score + 1.0).abs() < 1e-5, "opposite vectors should have similarity ~-1.0");
    }

    #[test]
    fn l2_normalize_zero_vector() {
        let v = l2_normalize(vec![0.0, 0.0, 0.0]);
        assert_eq!(v, vec![0.0, 0.0, 0.0], "zero vector unchanged");
    }

    #[test]
    fn l2_normalize_unit_vector() {
        let v = l2_normalize(vec![3.0, 4.0]);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "normalized vector should have unit norm");
        assert!((v[0] - 0.6).abs() < 1e-5);
        assert!((v[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn mean_pool_1d_passthrough() {
        let embedding = serde_json::json!([1.0, 2.0, 3.0]);
        let result = mean_pool_embedding(&embedding).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn mean_pool_2d_averaging() {
        let embedding = serde_json::json!([[1.0, 2.0], [3.0, 4.0]]);
        let result = mean_pool_embedding(&embedding).unwrap();
        assert_eq!(result, vec![2.0, 3.0]); // mean of [1,3]=2, mean of [2,4]=3
    }

    #[test]
    fn empty_index_filter_returns_empty() {
        let index = ToolEmbeddingIndex {
            tool_names: Vec::new(),
            embeddings: Vec::new(),
        };
        // Can't call async filter in sync test, but verify construction
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn index_len_and_empty() {
        let index = ToolEmbeddingIndex {
            tool_names: vec!["a".into(), "b".into()],
            embeddings: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        };
        assert_eq!(index.len(), 2);
        assert!(!index.is_empty());
    }

    #[test]
    fn compute_tool_hash_deterministic() {
        let tools = vec![
            ("tool1".to_string(), "description1".to_string()),
            ("tool2".to_string(), "description2".to_string()),
        ];
        let hash1 = compute_tool_hash(&tools);
        let hash2 = compute_tool_hash(&tools);
        assert_eq!(hash1, hash2, "same tools should produce same hash");
    }

    #[test]
    fn compute_tool_hash_different_for_different_tools() {
        let tools1 = vec![
            ("tool1".to_string(), "description1".to_string()),
        ];
        let tools2 = vec![
            ("tool2".to_string(), "description2".to_string()),
        ];
        let hash1 = compute_tool_hash(&tools1);
        let hash2 = compute_tool_hash(&tools2);
        assert_ne!(hash1, hash2, "different tools should produce different hashes");
    }
}
