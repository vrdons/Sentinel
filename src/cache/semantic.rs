use crate::cache::CachedResponse;
use crate::provider::{ChatRequest, ChatResponse};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use libsql::Connection;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;
use tracing::{debug, info};

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{Repo, api::tokio::Api};
use tokenizers::Tokenizer;

/// Configuration for semantic caching
#[derive(Debug, Clone)]
pub struct SemanticCacheConfig {
    pub enabled: bool,
    pub similarity_threshold: f32,
    pub embedding_model: String,
    pub max_cache_size: usize,
    pub ttl_hours: u32,
}

impl Default for SemanticCacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            similarity_threshold: 0.85,
            embedding_model: "all-MiniLM-L6-v2".to_string(),
            max_cache_size: 10000,
            ttl_hours: 24,
        }
    }
}

/// Trait for embedding providers
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    fn dimension(&self) -> usize;
    fn model_name(&self) -> &str;
}

/// Local embedding provider using a lightweight model via Candle
pub struct LocalEmbeddingProvider {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    model_name: String,
    dimension: usize,
}

impl std::fmt::Debug for LocalEmbeddingProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalEmbeddingProvider")
            .field("model_name", &self.model_name)
            .field("dimension", &self.dimension)
            .finish()
    }
}

impl LocalEmbeddingProvider {
    pub async fn new(model_name: String) -> Result<Self> {
        let start = Instant::now();
        let device = Device::Cpu; // Use CPU for portability in gateway

        let (repo_id, revision) = match model_name.as_str() {
            "all-MiniLM-L6-v2" => ("sentence-transformers/all-MiniLM-L6-v2", "refs/pr/21"),
            "gte-small" => ("thenlper/gte-small", "main"),
            _ => ("sentence-transformers/all-MiniLM-L6-v2", "refs/pr/21"),
        };

        info!("Loading embedding model {} from HuggingFace...", repo_id);

        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            repo_id.to_string(),
            hf_hub::RepoType::Model,
            revision.to_string(),
        ));

        let config_filename = repo.get("config.json").await?;
        let tokenizer_filename = repo.get("tokenizer.json").await?;
        let weights_filename = repo.get("model.safetensors").await?;

        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };

        let model = BertModel::load(vb, &config)?;

        let dimension = config.hidden_size;

        info!(
            "Loaded model {} ({} dims) in {:?}",
            model_name,
            dimension,
            start.elapsed()
        );

        Ok(Self {
            model,
            tokenizer,
            device,
            model_name,
            dimension,
        })
    }
}

#[async_trait]
impl EmbeddingProvider for LocalEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let start = Instant::now();

        let tokens = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("Tokenizer error: {}", e))?;
        let token_ids = tokens.get_ids();
        let token_ids_tensor = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;
        let token_type_ids = vec![0u32; token_ids.len()];
        let token_type_ids_tensor = Tensor::new(token_type_ids, &self.device)?.unsqueeze(0)?;

        // Use attention mask if available (optional in BertModel::forward)
        let attention_mask = vec![1u32; token_ids.len()];
        let attention_mask_tensor = Tensor::new(attention_mask, &self.device)?.unsqueeze(0)?;

        let embeddings = self.model.forward(
            &token_ids_tensor,
            &token_type_ids_tensor,
            Some(&attention_mask_tensor),
        )?;

        // Mean pooling
        let (_n_batch, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
        let embeddings = embeddings.get(0)?; // Remove batch dimension

        // Normalize
        let norm = embeddings.sqr()?.sum_all()?.sqrt()?;
        let embeddings = embeddings.broadcast_div(&norm)?;

        let v = embeddings.to_vec1::<f32>()?;

        debug!(
            "Generated embedding for '{}...' in {:?}",
            &text[..text.len().min(20)],
            start.elapsed()
        );
        Ok(v)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

/// Calculates cosine similarity between two embeddings
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

/// Semantic cache using vector similarity
pub struct SemanticCache {
    db: Arc<Connection>,
    embedding_provider: Box<dyn EmbeddingProvider>,
    config: SemanticCacheConfig,
    cleanup_tick: AtomicU32,
}

impl SemanticCache {
    pub async fn new(db: Arc<Connection>, config: SemanticCacheConfig) -> Result<Self> {
        let embedding_provider: Box<dyn EmbeddingProvider> =
            Box::new(LocalEmbeddingProvider::new(config.embedding_model.clone()).await?);

        let cache = Self {
            db,
            embedding_provider,
            config,
            cleanup_tick: AtomicU32::new(0),
        };

        cache.setup_database().await?;
        Ok(cache)
    }

    async fn setup_database(&self) -> Result<()> {
        // Create semantic cache table with vector support
        self.db
            .execute(
                &format!(
                    "CREATE TABLE IF NOT EXISTS semantic_cache (
                    id TEXT PRIMARY KEY,
                    prompt_hash TEXT NOT NULL,
                    embedding F32_BLOB({}),
                    response TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )",
                    self.embedding_provider.dimension()
                ),
                (),
            )
            .await?;

        // Create index for faster similarity search
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_semantic_cache_timestamp ON semantic_cache(timestamp)",
            (),
        ).await?;

        info!(
            "Semantic cache database initialized with {} dimensions",
            self.embedding_provider.dimension()
        );
        Ok(())
    }

    /// Generate a prompt key from the request for semantic comparison
    fn generate_prompt_key(&self, request: &ChatRequest) -> String {
        // Combine all messages into a single prompt for embedding
        let combined_prompt = request
            .messages
            .iter()
            .map(|msg| format!("{}: {}", msg.role, msg.content))
            .collect::<Vec<_>>()
            .join("\n");

        // Include model and temperature for context
        format!(
            "{}|model:{}|temp:{}",
            combined_prompt,
            request.model,
            request.temperature.unwrap_or(1.0)
        )
    }

    /// Check for semantically similar cached responses using vector similarity
    pub async fn get_similar(&self, request: &ChatRequest) -> Result<Option<CachedResponse>> {
        if !self.config.enabled {
            return Ok(None);
        }

        let start = Instant::now();
        let prompt_key = self.generate_prompt_key(request);

        // Generate embedding for the prompt
        let query_embedding = self.embedding_provider.embed(&prompt_key).await?;
        let query_embedding_bytes: Vec<u8> = query_embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        // Use libsql's vector similarity search if available
        let cutoff_time =
            chrono::Utc::now() - chrono::Duration::hours(self.config.ttl_hours as i64);

        // Attempt native vector search using vector_distance
        let mut rows = match self
            .db
            .query(
                "SELECT id, response, timestamp, model, embedding
             FROM semantic_cache
             WHERE timestamp > ? AND model = ?
             ORDER BY vector_distance(embedding, ?)
             LIMIT 1",
                libsql::params![
                    cutoff_time.to_rfc3339(),
                    request.model.clone(),
                    query_embedding_bytes.clone()
                ],
            )
            .await
        {
            Ok(rows) => rows,
            Err(e) => {
                debug!(
                    "Native vector search failed, falling back to manual scan: {}",
                    e
                );
                // Fallback to manual scan if vector_distance is not supported
                return self
                    .get_similar_fallback(request, query_embedding, cutoff_time)
                    .await;
            }
        };

        if let Some(row) = rows.next().await? {
            let cached_embedding: Vec<f32> = row
                .get::<Vec<u8>>(4)?
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                .collect();

            let similarity = cosine_similarity(&query_embedding, &cached_embedding);

            if similarity >= self.config.similarity_threshold {
                let response: ChatResponse = serde_json::from_str(&row.get::<String>(1)?)?;
                let timestamp = chrono::DateTime::parse_from_rfc3339(&row.get::<String>(2)?)?
                    .with_timezone(&chrono::Utc);

                info!(
                    "Semantic cache HIT (native): similarity={:.3}, time={:?}",
                    similarity,
                    start.elapsed()
                );
                return Ok(Some(CachedResponse {
                    response,
                    timestamp,
                    embedding: Some(cached_embedding),
                }));
            }
        }

        debug!("Semantic cache MISS: time={:?}", start.elapsed());
        Ok(None)
    }

    async fn get_similar_fallback(
        &self,
        request: &ChatRequest,
        query_embedding: Vec<f32>,
        cutoff_time: chrono::DateTime<chrono::Utc>,
    ) -> Result<Option<CachedResponse>> {
        let mut rows = self
            .db
            .query(
                "SELECT response, timestamp, embedding
             FROM semantic_cache 
             WHERE timestamp > ? AND model = ?
             LIMIT 100",
                libsql::params![cutoff_time.to_rfc3339(), request.model.clone()],
            )
            .await?;

        let mut best_match: Option<(f32, CachedResponse)> = None;

        while let Some(row) = rows.next().await? {
            let cached_embedding: Vec<f32> = row
                .get::<Vec<u8>>(2)?
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                .collect();

            let similarity = cosine_similarity(&query_embedding, &cached_embedding);

            if similarity >= self.config.similarity_threshold
                && best_match.as_ref().is_none_or(|(s, _)| similarity > *s)
            {
                let response: ChatResponse = serde_json::from_str(&row.get::<String>(0)?)?;
                let timestamp = chrono::DateTime::parse_from_rfc3339(&row.get::<String>(1)?)?
                    .with_timezone(&chrono::Utc);

                best_match = Some((
                    similarity,
                    CachedResponse {
                        response,
                        timestamp,
                        embedding: Some(cached_embedding),
                    },
                ));
            }
        }

        Ok(best_match.map(|(_, r)| r))
    }

    /// Store a response in the semantic cache
    pub async fn store(&self, request: &ChatRequest, response: &ChatResponse) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let start = Instant::now();
        let prompt_key = self.generate_prompt_key(request);

        // Generate embedding
        let embedding = self.embedding_provider.embed(&prompt_key).await?;

        // Convert embedding to bytes
        let embedding_bytes: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();

        let id = uuid::Uuid::new_v4().to_string();
        let response_json = serde_json::to_string(response)?;
        let timestamp = chrono::Utc::now();

        // Store in database
        self.db.execute(
            "INSERT INTO semantic_cache (id, prompt_hash, embedding, response, timestamp, model)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            libsql::params![
                id,
                format!("{:x}", md5::compute(prompt_key.as_bytes())),
                embedding_bytes,
                response_json,
                timestamp.to_rfc3339(),
                request.model.clone(),
            ],
        ).await?;

        debug!("Stored semantic cache entry in {:?}", start.elapsed());

        // Clean up old entries periodically (every ~100 inserts)
        if self
            .cleanup_tick
            .fetch_add(1, Ordering::Relaxed)
            .is_multiple_of(100)
        {
            self.cleanup_old_entries().await?;
        }

        Ok(())
    }

    async fn cleanup_old_entries(&self) -> Result<()> {
        let cutoff_time =
            chrono::Utc::now() - chrono::Duration::hours(self.config.ttl_hours as i64 * 2);

        let deleted = self
            .db
            .execute(
                "DELETE FROM semantic_cache WHERE timestamp < ?",
                libsql::params![cutoff_time.to_rfc3339()],
            )
            .await?;

        if deleted > 0 {
            info!("Cleaned up {} old semantic cache entries", deleted);
        }

        Ok(())
    }

    /// Get cache statistics
    pub async fn get_stats(&self) -> Result<serde_json::Value> {
        let mut rows = self
            .db
            .query(
                "SELECT COUNT(*), MAX(timestamp), MIN(timestamp) FROM semantic_cache",
                (),
            )
            .await?;

        if let Some(row) = rows.next().await? {
            Ok(serde_json::json!({
                "total_entries": row.get::<u32>(0)?,
                "latest_entry": row.get::<Option<String>>(1)?,
                "oldest_entry": row.get::<Option<String>>(2)?,
                "similarity_threshold": self.config.similarity_threshold,
                "embedding_model": self.config.embedding_model,
                "dimension": self.embedding_provider.dimension(),
            }))
        } else {
            Ok(serde_json::json!({}))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-6);
    }
}
