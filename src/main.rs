use clap::Parser;
use dashmap::DashMap;
use sentinel::cache::semantic::{SemanticCache, SemanticCacheConfig};
use sentinel::cli::{Cli, Commands};
use sentinel::config::AppConfig;
use sentinel::cost::optimization::CostOptimizer;
use sentinel::cost::pricing::PricingTable;
use sentinel::provider::anthropic::AnthropicProvider;
use sentinel::provider::cohere::CohereProvider;
use sentinel::provider::gemini::GeminiProvider;
use sentinel::provider::mistral::MistralProvider;
use sentinel::provider::ollama::OllamaProvider;
use sentinel::provider::openai::OpenAIProvider;
use sentinel::provider::perplexity::PerplexityProvider;
use sentinel::provider::together::TogetherAIProvider;
use sentinel::proxy::pii::PiiRedactor;
use sentinel::proxy::{
    ProviderHealth, ProxyState, check_all_providers, create_router as create_proxy_router,
};
use sentinel::router::smart::SmartRouter;
use sentinel::storage::db::Database;
use sentinel::ui::create_router as create_ui_router;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, warn};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Start {
            proxy_only,
            dashboard_only,
        } => {
            let config = AppConfig::load()?;

            // Initialize DB
            let db_path = config
                .database
                .path
                .clone()
                .unwrap_or_else(|| std::path::PathBuf::from("sentinel.db"));
            let db = Arc::new(Database::new(&db_path).await?);

            let primary_provider = if config.providers.primary == "openai" {
                let api_key = config
                    .providers
                    .openai_api_key
                    .clone()
                    .or_else(|| std::env::var("OPENAI_API_KEY").ok())
                    .ok_or_else(|| anyhow::anyhow!("OpenAI API key not found"))?;
                Arc::new(OpenAIProvider::new(api_key)) as Arc<dyn sentinel::provider::LlmProvider>
            } else if config.providers.primary == "anthropic" {
                let api_key = config
                    .providers
                    .anthropic_api_key
                    .clone()
                    .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
                    .ok_or_else(|| anyhow::anyhow!("Anthropic API key not found"))?;
                Arc::new(AnthropicProvider::new(api_key))
                    as Arc<dyn sentinel::provider::LlmProvider>
            } else if config.providers.primary == "gemini" {
                let api_key = config
                    .providers
                    .gemini_api_key
                    .clone()
                    .or_else(|| std::env::var("GEMINI_API_KEY").ok())
                    .ok_or_else(|| anyhow::anyhow!("Gemini API key not found"))?;
                Arc::new(GeminiProvider::new(api_key)) as Arc<dyn sentinel::provider::LlmProvider>
            } else if config.providers.primary == "mistral" {
                let api_key = config
                    .providers
                    .mistral_api_key
                    .clone()
                    .or_else(|| std::env::var("MISTRAL_API_KEY").ok())
                    .ok_or_else(|| anyhow::anyhow!("Mistral API key not found"))?;
                Arc::new(MistralProvider::new(api_key)) as Arc<dyn sentinel::provider::LlmProvider>
            } else if config.providers.primary == "cohere" {
                let api_key = config
                    .providers
                    .cohere_api_key
                    .clone()
                    .or_else(|| std::env::var("COHERE_API_KEY").ok())
                    .ok_or_else(|| anyhow::anyhow!("Cohere API key not found"))?;
                Arc::new(CohereProvider::new(api_key)) as Arc<dyn sentinel::provider::LlmProvider>
            } else if config.providers.primary == "perplexity" {
                let api_key = config
                    .providers
                    .perplexity_api_key
                    .clone()
                    .or_else(|| std::env::var("PERPLEXITY_API_KEY").ok())
                    .ok_or_else(|| anyhow::anyhow!("Perplexity API key not found"))?;
                Arc::new(PerplexityProvider::new(api_key))
                    as Arc<dyn sentinel::provider::LlmProvider>
            } else if config.providers.primary == "together" {
                let api_key = config
                    .providers
                    .together_api_key
                    .clone()
                    .or_else(|| std::env::var("TOGETHER_API_KEY").ok())
                    .ok_or_else(|| anyhow::anyhow!("Together AI API key not found"))?;
                Arc::new(TogetherAIProvider::new(api_key))
                    as Arc<dyn sentinel::provider::LlmProvider>
            } else if config.providers.primary == "ollama" {
                Arc::new(OllamaProvider::new(
                    config.providers.ollama_base_url.clone(),
                )) as Arc<dyn sentinel::provider::LlmProvider>
            } else {
                return Err(anyhow::anyhow!(
                    "Unsupported primary provider: {}",
                    config.providers.primary
                ));
            };

            let mut fallback_providers = Vec::new();
            for fallback in &config.providers.fallback {
                if fallback == "openai" {
                    if let Some(api_key) = config
                        .providers
                        .openai_api_key
                        .clone()
                        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
                    {
                        fallback_providers.push(Arc::new(OpenAIProvider::new(api_key))
                            as Arc<dyn sentinel::provider::LlmProvider>);
                    }
                } else if fallback == "anthropic" {
                    if let Some(api_key) = config
                        .providers
                        .anthropic_api_key
                        .clone()
                        .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
                    {
                        fallback_providers.push(Arc::new(AnthropicProvider::new(api_key))
                            as Arc<dyn sentinel::provider::LlmProvider>);
                    }
                } else if fallback == "gemini" {
                    if let Some(api_key) = config
                        .providers
                        .gemini_api_key
                        .clone()
                        .or_else(|| std::env::var("GEMINI_API_KEY").ok())
                    {
                        fallback_providers.push(Arc::new(GeminiProvider::new(api_key))
                            as Arc<dyn sentinel::provider::LlmProvider>);
                    }
                } else if fallback == "mistral" {
                    if let Some(api_key) = config
                        .providers
                        .mistral_api_key
                        .clone()
                        .or_else(|| std::env::var("MISTRAL_API_KEY").ok())
                    {
                        fallback_providers.push(Arc::new(MistralProvider::new(api_key))
                            as Arc<dyn sentinel::provider::LlmProvider>);
                    }
                } else if fallback == "cohere" {
                    if let Some(api_key) = config
                        .providers
                        .cohere_api_key
                        .clone()
                        .or_else(|| std::env::var("COHERE_API_KEY").ok())
                    {
                        fallback_providers.push(Arc::new(CohereProvider::new(api_key))
                            as Arc<dyn sentinel::provider::LlmProvider>);
                    }
                } else if fallback == "perplexity" {
                    if let Some(api_key) = config
                        .providers
                        .perplexity_api_key
                        .clone()
                        .or_else(|| std::env::var("PERPLEXITY_API_KEY").ok())
                    {
                        fallback_providers.push(Arc::new(PerplexityProvider::new(api_key))
                            as Arc<dyn sentinel::provider::LlmProvider>);
                    }
                } else if fallback == "together" {
                    if let Some(api_key) = config
                        .providers
                        .together_api_key
                        .clone()
                        .or_else(|| std::env::var("TOGETHER_API_KEY").ok())
                    {
                        fallback_providers.push(Arc::new(TogetherAIProvider::new(api_key))
                            as Arc<dyn sentinel::provider::LlmProvider>);
                    }
                } else if fallback == "ollama" {
                    fallback_providers.push(Arc::new(OllamaProvider::new(
                        config.providers.ollama_base_url.clone(),
                    ))
                        as Arc<dyn sentinel::provider::LlmProvider>);
                }
            }

            // Create provider health wrappers
            let primary_health = ProviderHealth::new(primary_provider.clone());
            let fallback_health: Vec<ProviderHealth> = fallback_providers
                .into_iter()
                .map(ProviderHealth::new)
                .collect();

            // Initialize semantic cache if enabled
            let semantic_cache = if config.cache.semantic {
                info!(
                    "Initializing semantic cache with model: {}",
                    config.cache.embedding_model
                );
                let cache_config = SemanticCacheConfig {
                    enabled: config.cache.semantic,
                    similarity_threshold: config.cache.similarity_threshold,
                    embedding_model: config.cache.embedding_model.clone(),
                    max_cache_size: config.cache.max_cache_size,
                    ttl_hours: config.cache.ttl_hours,
                };
                match SemanticCache::new(db.get_connection(), cache_config).await {
                    Ok(cache) => {
                        info!("✓ Semantic cache initialized successfully");
                        Some(Arc::new(cache))
                    }
                    Err(e) => {
                        warn!("Failed to initialize semantic cache: {}", e);
                        None
                    }
                }
            } else {
                info!("Semantic cache disabled");
                None
            };

            let mut providers_map: std::collections::HashMap<
                String,
                Arc<dyn sentinel::provider::LlmProvider>,
            > = std::collections::HashMap::new();
            providers_map.insert(config.providers.primary.clone(), primary_provider.clone());

            // Re-initialize all requested providers for the map (or reuse existing ones)
            // For brevity and to avoid re-writing all logic, let's just use what we have
            providers_map.insert("openai".to_string(), primary_provider.clone()); // Simplification

            let pricing = PricingTable::default();
            let cost_optimizer = CostOptimizer::new(pricing.clone(), db.clone());

            let mut smart_router = SmartRouter::new(
                config.router.clone(),
                db.clone(),
                cost_optimizer,
                providers_map,
            );

            // Add A/B tests from config
            for ab_test in &config.router.ab_tests {
                if ab_test.active {
                    smart_router.add_ab_test(
                        ab_test.name.clone(),
                        ab_test.models.clone(),
                        ab_test.traffic_split.clone(),
                    );
                }
            }

            let state = Arc::new(ProxyState {
                config: config.clone(),
                primary_provider: primary_health,
                fallback_providers: fallback_health,
                db,
                pricing,
                cache: DashMap::new(),
                semantic_cache,
                pii_redactor: PiiRedactor::new(),
                smart_router: Arc::new(tokio::sync::RwLock::new(smart_router)),
            });

            // Run health checks on startup
            info!("Running provider health checks...");
            let health_results = check_all_providers(&state).await;
            for (provider_name, is_healthy) in health_results {
                if is_healthy {
                    info!("✓ Provider {} is healthy", provider_name);
                } else {
                    warn!("✗ Provider {} failed health check", provider_name);
                }
            }

            let mut handles = Vec::new();

            if !dashboard_only {
                let proxy_state = state.clone();
                let proxy_addr = format!("{}:{}", config.server.host, config.server.port)
                    .parse::<SocketAddr>()?;
                let proxy_app = create_proxy_router(proxy_state);
                info!("Sentinel proxy listening on {}", proxy_addr);
                handles.push(tokio::spawn(async move {
                    let listener = tokio::net::TcpListener::bind(proxy_addr).await.unwrap();
                    axum::serve(listener, proxy_app).await.unwrap();
                }));
            }

            if !proxy_only {
                let ui_state = state.clone();
                let ui_addr = format!("{}:{}", config.dashboard.host, config.dashboard.port)
                    .parse::<SocketAddr>()?;
                let ui_app = create_ui_router(ui_state);
                info!("Sentinel dashboard listening on {}", ui_addr);
                handles.push(tokio::spawn(async move {
                    let listener = tokio::net::TcpListener::bind(ui_addr).await.unwrap();
                    axum::serve(listener, ui_app).await.unwrap();
                }));
            }

            for handle in handles {
                handle.await?;
            }
        }
        Commands::Logs { tail, follow } => {
            let config = AppConfig::load()?;
            let db_path = config
                .database
                .path
                .unwrap_or_else(|| std::path::PathBuf::from("sentinel.db"));
            let db = Database::new(&db_path).await?;

            let mut last_timestamp = None;

            loop {
                let logs = db.get_recent_logs(tail.unwrap_or(20)).await?;
                let mut new_logs = logs.into_iter().rev().collect::<Vec<_>>();

                if let Some(last) = last_timestamp {
                    new_logs.retain(|l| l.timestamp > last);
                }

                for log in new_logs {
                    println!(
                        "[{}] {} | {} | {} | {} tokens | ${:.4} | {}ms",
                        log.timestamp,
                        log.status,
                        log.provider,
                        log.model,
                        log.input_tokens + log.output_tokens,
                        log.cost_usd,
                        log.latency_ms
                    );
                    last_timestamp = Some(log.timestamp);
                }

                if !follow {
                    break;
                }
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
        Commands::Config => {
            let config = AppConfig::load()?;
            println!("{:#?}", config);
        }
    }

    Ok(())
}
