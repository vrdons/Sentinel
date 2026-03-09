use drizzle::sqlite::prelude::*;

#[SQLiteTable(NAME = "logs")]
pub struct LogsTable {
    #[column(PRIMARY)]
    pub id: String,
    pub timestamp: String,
    pub provider: String,
    pub model: String,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub latency_ms: i64,
    pub cost_usd: f64,
    #[column(DEFAULT = 0.0)]
    pub cost_saved: f64,
    #[column(DEFAULT = 0.5)]
    pub quality_score: f64,
    pub cache_hit: bool,
    #[column(DEFAULT = false)]
    pub pii_redacted: bool,
    pub status: String,
    pub error_message: Option<String>,
}

#[SQLiteTable(NAME = "cost_optimizations")]
pub struct CostOptimizationsTable {
    #[column(PRIMARY)]
    pub id: String,
    pub timestamp: String,
    pub endpoint_pattern: String,
    pub original_model: String,
    pub suggested_model: String,
    pub potential_savings_percent: f64,
    pub confidence_score: f64,
    pub reason: String,
    pub status: String,
}

#[SQLiteTable(NAME = "quality_metrics")]
pub struct QualityMetricsTable {
    #[column(PRIMARY)]
    pub id: String,
    pub timestamp: String,
    pub model: String,
    pub prompt_hash: String,
    pub response_quality: f64,
    pub latency_score: f64,
    pub cost_efficiency: f64,
    pub user_feedback: Option<String>,
}

#[SQLiteTable(NAME = "semantic_cache")]
pub struct SemanticCacheTable {
    #[column(PRIMARY)]
    pub id: String,
    pub prompt_hash: String,
    pub embedding: Vec<u8>,
    pub response: String,
    pub timestamp: String,
    pub model: String,
    pub created_at: String,
}

#[derive(SQLiteSchema)]
pub struct SentinelSchema {
    pub logs: LogsTable,
    pub cost_optimizations: CostOptimizationsTable,
    pub quality_metrics: QualityMetricsTable,
    pub semantic_cache: SemanticCacheTable,
}
