use crate::cache::CachedResponse;
use crate::provider::{ChatRequest, ChatResponse};
use crate::storage::db::SharedDrizzleDb;
use crate::storage::schema::{InsertSemanticCacheTable, SelectSemanticCacheTable, SentinelSchema};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use drizzle::core::expr::{and, eq, gt, lt};
use drizzle::sqlite::prelude::*;
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
        let device = Device::Cpu;

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

        let attention_mask = vec![1u32; token_ids.len()];
        let attention_mask_tensor = Tensor::new(attention_mask, &self.device)?.unsqueeze(0)?;

        let embeddings = self.model.forward(
            &token_ids_tensor,
            &token_type_ids_tensor,
            Some(&attention_mask_tensor),
        )?;

        let (_n_batch, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
        let embeddings = embeddings.get(0)?;

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

#[derive(SQLiteFromRow, Default)]
struct CacheStatsRow {
    timestamp: String,
}

#[derive(SQLiteFromRow, Default)]
struct CacheIdRow {
    id: String,
}

pub struct SemanticCache {
    db: SharedDrizzleDb,
    schema: SentinelSchema,
    embedding_provider: Box<dyn EmbeddingProvider>,
    config: SemanticCacheConfig,
    cleanup_tick: AtomicU32,
}

impl SemanticCache {
    pub async fn new(db: SharedDrizzleDb, config: SemanticCacheConfig) -> Result<Self> {
        let embedding_provider: Box<dyn EmbeddingProvider> =
            Box::new(LocalEmbeddingProvider::new(config.embedding_model.clone()).await?);

        let cache = Self {
            db,
            schema: SentinelSchema::new(),
            embedding_provider,
            config,
            cleanup_tick: AtomicU32::new(0),
        };

        cache.setup_database().await?;
        Ok(cache)
    }

    async fn setup_database(&self) -> Result<()> {
        // semantic_cache table/index are created during Database::new.
        info!(
            "Semantic cache database initialized with {} dimensions",
            self.embedding_provider.dimension()
        );
        Ok(())
    }

    fn generate_prompt_key(&self, request: &ChatRequest) -> String {
        let combined_prompt = request
            .messages
            .iter()
            .map(|msg| format!("{}: {}", msg.role, msg.content))
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            "{}|model:{}|temp:{}",
            combined_prompt,
            request.model,
            request.temperature.unwrap_or(1.0)
        )
    }

    pub async fn get_similar(&self, request: &ChatRequest) -> Result<Option<CachedResponse>> {
        if !self.config.enabled {
            return Ok(None);
        }

        let start = Instant::now();
        let prompt_key = self.generate_prompt_key(request);

        let query_embedding = self.embedding_provider.embed(&prompt_key).await?;
        let cutoff_time =
            chrono::Utc::now() - chrono::Duration::hours(self.config.ttl_hours as i64);

        let result = self
            .get_similar_fallback(request, query_embedding, cutoff_time)
            .await?;

        if result.is_some() {
            info!("Semantic cache HIT: time={:?}", start.elapsed());
        } else {
            debug!("Semantic cache MISS: time={:?}", start.elapsed());
        }

        Ok(result)
    }

    async fn get_similar_fallback(
        &self,
        request: &ChatRequest,
        query_embedding: Vec<f32>,
        cutoff_time: chrono::DateTime<chrono::Utc>,
    ) -> Result<Option<CachedResponse>> {
        let db = self
            .db
            .lock()
            .map_err(|_| anyhow!("database mutex poisoned"))?;

        let rows: Vec<SelectSemanticCacheTable> = db
            .select(())
            .from(self.schema.semantic_cache)
            .r#where(and([
                gt(
                    self.schema.semantic_cache.timestamp,
                    cutoff_time.to_rfc3339(),
                ),
                eq(self.schema.semantic_cache.model, request.model.clone()),
            ]))
            .limit(100)
            .all()?;

        let mut best_match: Option<(f32, CachedResponse)> = None;

        for row in rows {
            let cached_embedding: Vec<f32> = row
                .embedding
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                .collect();

            let similarity = cosine_similarity(&query_embedding, &cached_embedding);

            if similarity >= self.config.similarity_threshold
                && best_match.as_ref().is_none_or(|(s, _)| similarity > *s)
            {
                let response: ChatResponse = serde_json::from_str(&row.response)?;
                let timestamp = chrono::DateTime::parse_from_rfc3339(&row.timestamp)?
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

    pub async fn store(&self, request: &ChatRequest, response: &ChatResponse) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let start = Instant::now();
        let prompt_key = self.generate_prompt_key(request);

        let embedding = self.embedding_provider.embed(&prompt_key).await?;
        let embedding_bytes: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();

        let id = uuid::Uuid::new_v4().to_string();
        let response_json = serde_json::to_string(response)?;
        let timestamp = chrono::Utc::now();

        {
            let db = self
                .db
                .lock()
                .map_err(|_| anyhow!("database mutex poisoned"))?;

            db.insert(self.schema.semantic_cache)
                .values([InsertSemanticCacheTable::new(
                    id,
                    format!("{:x}", md5::compute(prompt_key.as_bytes())),
                    embedding_bytes,
                    response_json,
                    timestamp.to_rfc3339(),
                    request.model.clone(),
                    chrono::Utc::now().to_rfc3339(),
                )])
                .execute()?;
        }

        debug!("Stored semantic cache entry in {:?}", start.elapsed());

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

        let db = self
            .db
            .lock()
            .map_err(|_| anyhow!("database mutex poisoned"))?;

        let deleted = db
            .delete(self.schema.semantic_cache)
            .r#where(lt(
                self.schema.semantic_cache.timestamp,
                cutoff_time.to_rfc3339(),
            ))
            .execute()?;

        if deleted > 0 {
            info!("Cleaned up {} old semantic cache entries", deleted);
        }

        if self.config.max_cache_size > 0 {
            let mut offset = self.config.max_cache_size;
            let batch_size = 256usize;

            loop {
                let stale_ids: Vec<CacheIdRow> = db
                    .select(self.schema.semantic_cache.id)
                    .from(self.schema.semantic_cache)
                    .order_by([drizzle::core::OrderBy::desc(
                        self.schema.semantic_cache.timestamp,
                    )])
                    .limit(batch_size)
                    .offset(offset)
                    .all()?;

                if stale_ids.is_empty() {
                    break;
                }

                for row in &stale_ids {
                    db.delete(self.schema.semantic_cache)
                        .r#where(eq(self.schema.semantic_cache.id, row.id.clone()))
                        .execute()?;
                }

                offset += stale_ids.len();
            }
        }

        Ok(())
    }

    pub async fn get_stats(&self) -> Result<serde_json::Value> {
        let db = self
            .db
            .lock()
            .map_err(|_| anyhow!("database mutex poisoned"))?;

        let rows: Vec<CacheStatsRow> = db
            .select(self.schema.semantic_cache.timestamp)
            .from(self.schema.semantic_cache)
            .all()?;

        let total_entries = rows.len() as u32;
        let mut latest_entry: Option<String> = None;
        let mut oldest_entry: Option<String> = None;

        for row in rows {
            if latest_entry
                .as_ref()
                .is_none_or(|current| row.timestamp > *current)
            {
                latest_entry = Some(row.timestamp.clone());
            }
            if oldest_entry
                .as_ref()
                .is_none_or(|current| row.timestamp < *current)
            {
                oldest_entry = Some(row.timestamp);
            }
        }

        Ok(serde_json::json!({
            "total_entries": total_entries,
            "latest_entry": latest_entry,
            "oldest_entry": oldest_entry,
            "similarity_threshold": self.config.similarity_threshold,
            "embedding_model": self.config.embedding_model,
            "dimension": self.embedding_provider.dimension(),
        }))
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
