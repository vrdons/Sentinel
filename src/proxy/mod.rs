pub mod handlers;
pub mod middleware;
pub mod pii;

use axum::{routing::post, Router};
use std::sync::Arc;
use crate::provider::LlmProvider;
use crate::config::AppConfig;
use crate::storage::db::Database;
use crate::cost::pricing::PricingTable;
use dashmap::DashMap;
use crate::provider::ChatResponse;
use crate::proxy::pii::PiiRedactor;
use crate::cache::semantic::SemanticCache;
use crate::router::smart::SmartRouter;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::RwLock;

pub struct ProviderHealth {
    pub provider: Arc<dyn LlmProvider>,
    pub is_healthy: Arc<AtomicBool>,
}

pub struct ProxyState {
    pub config: AppConfig,
    pub primary_provider: ProviderHealth,
    pub fallback_providers: Vec<ProviderHealth>,
    pub db: Arc<Database>,
    pub pricing: PricingTable,
    pub cache: DashMap<u64, ChatResponse>,
    pub semantic_cache: Option<Arc<SemanticCache>>,
    pub pii_redactor: PiiRedactor,
    pub smart_router: Arc<RwLock<SmartRouter>>,
}

impl ProviderHealth {
    pub fn new(provider: Arc<dyn LlmProvider>) -> Self {
        Self {
            provider,
            is_healthy: Arc::new(AtomicBool::new(true)),
        }
    }

    pub async fn health_check(&self) -> bool {
        match self.provider.health_check().await {
            Ok(()) => {
                self.is_healthy.store(true, Ordering::Relaxed);
                true
            }
            Err(_) => {
                self.is_healthy.store(false, Ordering::Relaxed);
                false
            }
        }
    }

    pub fn is_healthy(&self) -> bool {
        self.is_healthy.load(Ordering::Relaxed)
    }
}

pub async fn check_all_providers(state: &ProxyState) -> Vec<(String, bool)> {
    let mut results = Vec::new();
    
    let primary_healthy = state.primary_provider.health_check().await;
    results.push((state.primary_provider.provider.name().to_string(), primary_healthy));
    
    for fallback in &state.fallback_providers {
        let healthy = fallback.health_check().await;
        results.push((fallback.provider.name().to_string(), healthy));
    }
    
    results
}

pub fn create_router(state: Arc<ProxyState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(handlers::chat_completion))
        .with_state(state)
}
