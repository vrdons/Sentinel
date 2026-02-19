use clap::Parser;
use sentinel::cli::{Cli, Commands};
use sentinel::config::AppConfig;
use sentinel::provider::openai::OpenAIProvider;
use sentinel::provider::anthropic::AnthropicProvider;
use sentinel::proxy::{ProxyState, ProviderHealth, create_router as create_proxy_router, check_all_providers};
use sentinel::proxy::pii::PiiRedactor;
use sentinel::storage::db::Database;
use sentinel::cost::pricing::PricingTable;
use sentinel::ui::create_router as create_ui_router;
use dashmap::DashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::info;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};
use std::time::Duration;

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
            let db_path = config.database.path.clone().unwrap_or_else(|| std::path::PathBuf::from("sentinel.db"));
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
                Arc::new(AnthropicProvider::new(api_key)) as Arc<dyn sentinel::provider::LlmProvider>
            } else {
                return Err(anyhow::anyhow!(
                    "Unsupported primary provider: {}",
                    config.providers.primary
                ));
            };

            let mut fallback_providers = Vec::new();
            for fallback in &config.providers.fallback {
                if fallback == "openai" {
                        if let Some(api_key) = config.providers.openai_api_key.clone().or_else(|| std::env::var("OPENAI_API_KEY").ok()) {
                            fallback_providers.push(Arc::new(OpenAIProvider::new(api_key)) as Arc<dyn sentinel::provider::LlmProvider>);
                        }
                } else if fallback == "anthropic" {
                        if let Some(api_key) = config.providers.anthropic_api_key.clone().or_else(|| std::env::var("ANTHROPIC_API_KEY").ok()) {
                            fallback_providers.push(Arc::new(AnthropicProvider::new(api_key)) as Arc<dyn sentinel::provider::LlmProvider>);
                        }
                }
            }

            // Create provider health wrappers
            let primary_health = ProviderHealth::new(primary_provider);
            let fallback_health: Vec<ProviderHealth> = fallback_providers.into_iter()
                .map(ProviderHealth::new)
                .collect();

            let state = Arc::new(ProxyState {
                config: config.clone(),
                primary_provider: primary_health,
                fallback_providers: fallback_health,
                db,
                pricing: PricingTable::default(),
                cache: DashMap::new(),
                pii_redactor: PiiRedactor::new(),
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
                let proxy_addr = format!("{}:{}", config.server.host, config.server.port).parse::<SocketAddr>()?;
                let proxy_app = create_proxy_router(proxy_state);
                info!("Sentinel proxy listening on {}", proxy_addr);
                handles.push(tokio::spawn(async move {
                    let listener = tokio::net::TcpListener::bind(proxy_addr).await.unwrap();
                    axum::serve(listener, proxy_app).await.unwrap();
                }));
            }

            if !proxy_only {
                let ui_state = state.clone();
                let ui_addr = format!("{}:{}", config.dashboard.host, config.dashboard.port).parse::<SocketAddr>()?;
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
            let db_path = config.database.path.unwrap_or_else(|| std::path::PathBuf::from("sentinel.db"));
            let db = Database::new(&db_path).await?;

            let mut last_timestamp = None;

            loop {
                let logs = db.get_recent_logs(tail.unwrap_or(20)).await?;
                let mut new_logs = logs.into_iter().rev().collect::<Vec<_>>();

                if let Some(last) = last_timestamp {
                    new_logs.retain(|l| l.timestamp > last);
                }

                for log in new_logs {
                    println!("[{}] {} | {} | {} | {} tokens | ${:.4} | {}ms",
                        log.timestamp, log.status, log.provider, log.model,
                        log.input_tokens + log.output_tokens, log.cost_usd, log.latency_ms);
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
