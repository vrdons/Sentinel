use axum::{routing::get, Router, extract::State, response::Html};
use std::sync::Arc;
use crate::proxy::ProxyState;
use askama::Template;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
struct MonthlySaving {
    month: String,
    saved: f64,
    requests: u32,
    spent: f64,
}

#[derive(Serialize, Deserialize, Clone)]
struct ModelStats {
    model: String,
    requests: u32,
    avg_latency: f64,
    avg_cost: f64,
    avg_quality: f64,
    total_cost: f64,
    quality_per_dollar: f64,
}

#[derive(Template)]
#[template(path = "index.html")]
struct IndexTemplate {}

#[derive(Template)]
#[template(path = "stats.html")]
struct StatsTemplate {
    total_requests: u32,
    total_cost: f64,
    avg_latency: f64,
    cache_hits: u32,
    pii_redacted: u32,
    total_saved: f64,
    monthly_saved: f64,
}

#[derive(Template)]
#[template(path = "logs.html")]
struct LogsTemplate {
    logs: Vec<crate::storage::db::RequestLog>,
}

#[derive(Template)]
#[template(path = "savings.html")]
struct SavingsTemplate {
    total_saved: f64,
    savings_percentage: f64,
    optimized_requests: u32,
    original_cost: f64,
    monthly_data: Vec<MonthlySaving>,
}

#[derive(Template)]
#[template(path = "optimizations.html")]
struct OptimizationsTemplate {
    optimizations: Vec<crate::storage::db::CostOptimization>,
    total_potential_savings: f64,
    top_models: Vec<ModelStats>,
}

pub fn create_router(state: Arc<ProxyState>) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/api/logs", get(get_logs))
        .route("/api/stats", get(get_stats))
        .route("/api/savings", get(get_savings))
        .route("/api/optimizations", get(get_optimizations))
        .with_state(state)
}

async fn index() -> Html<String> {
    let template = IndexTemplate {};
    Html(template.render().unwrap())
}

async fn get_logs(State(state): State<Arc<ProxyState>>) -> Html<String> {
    let logs = state.db.get_recent_logs(20).await.unwrap_or_default();
    let template = LogsTemplate { logs };
    Html(template.render().unwrap())
}

async fn get_savings(State(state): State<Arc<ProxyState>>) -> Html<String> {
    let monthly_savings = state.db.get_monthly_savings(6).await.unwrap_or_else(|_| serde_json::json!({
        "total_saved": 0.0,
        "monthly_breakdown": []
    }));

    let stats = state.db.get_stats().await.unwrap_or_else(|_| serde_json::json!({}));

    let total_cost = stats["total_cost"].as_f64().unwrap_or(0.0);
    let total_saved = stats["monthly_saved"].as_f64().unwrap_or(0.0);
    let original_cost = total_cost + total_saved;
    let savings_percentage = if original_cost > 0.0 { (total_saved / original_cost) * 100.0 } else { 0.0 };

    let monthly_data: Vec<MonthlySaving> = serde_json::from_value(monthly_savings["monthly_breakdown"].clone()).unwrap_or_default();

    let template = SavingsTemplate {
        total_saved,
        savings_percentage,
        optimized_requests: stats["cache_hits"].as_u64().unwrap_or(0) as u32, // Simplified
        original_cost,
        monthly_data,
    };
    Html(template.render().unwrap())
}

async fn get_optimizations(State(state): State<Arc<ProxyState>>) -> Html<String> {
    let optimizations = state.db.get_pending_optimizations().await.unwrap_or_default();
    let performance = state.db.get_model_performance_stats().await.unwrap_or_else(|_| serde_json::json!({
        "models": []
    }));

    let top_models: Vec<ModelStats> = serde_json::from_value(performance["models"].clone()).unwrap_or_default();

    let template = OptimizationsTemplate {
        optimizations,
        total_potential_savings: 0.0, // TODO: calculate
        top_models,
    };
    Html(template.render().unwrap())
}

async fn get_stats(State(state): State<Arc<ProxyState>>) -> Html<String> {
    let stats = state.db.get_stats().await.unwrap_or_else(|_| serde_json::json!({
        "total_requests": 0,
        "total_cost": 0.0,
        "avg_latency": 0.0,
        "cache_hits": 0,
    }));

    let template = StatsTemplate {
        total_requests: stats["total_requests"].as_u64().unwrap_or(0) as u32,
        total_cost: stats["total_cost"].as_f64().unwrap_or(0.0),
        avg_latency: stats["avg_latency"].as_f64().unwrap_or(0.0),
        cache_hits: stats["cache_hits"].as_u64().unwrap_or(0) as u32,
        pii_redacted: stats["pii_redacted"].as_u64().unwrap_or(0) as u32,
        total_saved: stats["total_saved"].as_f64().unwrap_or(0.0),
        monthly_saved: stats["monthly_saved"].as_f64().unwrap_or(0.0),
    };
    Html(template.render().unwrap())
}
