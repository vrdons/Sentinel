use config::{Config, ConfigError, Environment, File};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub providers: ProvidersConfig,
    pub database: DatabaseConfig,
    pub dashboard: DashboardConfig,
    pub cache: CacheConfig,
    pub router: RouterConfig,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct CacheConfig {
    pub semantic: bool,
    pub similarity_threshold: f32,
    pub embedding_model: String,
    pub max_cache_size: usize,
    pub ttl_hours: u32,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ProvidersConfig {
    pub openai_api_key: Option<String>,
    pub anthropic_api_key: Option<String>,
    pub gemini_api_key: Option<String>,
    pub mistral_api_key: Option<String>,
    pub cohere_api_key: Option<String>,
    pub perplexity_api_key: Option<String>,
    pub together_api_key: Option<String>,
    pub ollama_base_url: Option<String>,
    pub primary: String,
    pub fallback: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct DatabaseConfig {
    pub path: Option<PathBuf>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct DashboardConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct RouterConfig {
    pub smart_mode: bool,
    pub quality_threshold: f32,
    pub cost_weight: f32,
    #[serde(default)]
    pub ab_tests: Vec<AbTestConfig>,
    #[serde(default)]
    pub fallback_rules: Vec<FallbackRuleConfig>,
    #[serde(default)]
    pub endpoints: HashMap<String, String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AbTestConfig {
    pub name: String,
    pub models: Vec<String>,
    pub traffic_split: Vec<f32>,
    pub active: bool,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct FallbackRuleConfig {
    pub on_model: String,
    pub fallback_to: String,
    pub condition: String, // "error", "low_quality", "high_latency"
}

impl AppConfig {
    pub fn load() -> Result<Self, ConfigError> {
        let db_default_path = get_default_db_path();

        let s = Config::builder()
            .set_default("server.host", "127.0.0.1")?
            .set_default("server.port", 8080)?
            .set_default("dashboard.host", "127.0.0.1")?
            .set_default("dashboard.port", 3000)?
            .set_default("providers.primary", "openai")?
            .set_default("providers.fallback", Vec::<String>::new())?
            .set_default(
                "database.path",
                db_default_path.to_str().unwrap_or("./sentinel.db"),
            )?
            .set_default("cache.semantic", false)?
            .set_default("cache.similarity_threshold", 0.85)?
            .set_default("cache.embedding_model", "all-MiniLM-L6-v2")?
            .set_default("cache.max_cache_size", 10000)?
            .set_default("cache.ttl_hours", 24)?
            .set_default("router.smart_mode", false)?
            .set_default("router.quality_threshold", 0.7)?
            .set_default("router.cost_weight", 0.3)?
            // Load from sentinel.toml
            .add_source(File::with_name("sentinel").required(false))
            // Load from environment variables (e.g., SENTINEL_SERVER__PORT)
            .add_source(Environment::with_prefix("SENTINEL").separator("__"))
            // Special handling for common API keys
            .add_source(Environment::default().list_separator(","))
            .build()?;

        s.try_deserialize()
    }
}

fn get_default_db_path() -> PathBuf {
    if let Some(proj_dirs) = ProjectDirs::from("", "", "sentinel") {
        let data_dir = proj_dirs.data_dir();
        std::fs::create_dir_all(data_dir).ok();
        data_dir.join("sentinel.db")
    } else {
        PathBuf::from("./sentinel.db")
    }
}
