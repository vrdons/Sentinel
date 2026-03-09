use crate::cache::hash_request;
use crate::provider::ChatRequest;
use crate::proxy::ProxyState;
use crate::storage::db::RequestLog;
use axum::{
    Json,
    extract::State,
    response::IntoResponse,
    response::sse::{Event, Sse},
};
use chrono::Utc;
use futures_util::stream::StreamExt;
use std::sync::Arc;
use tracing::{error, info};
use uuid::Uuid;

pub async fn chat_completion(
    State(state): State<Arc<ProxyState>>,
    Json(mut request): Json<ChatRequest>,
) -> Result<axum::response::Response, String> {
    let start_time = Utc::now();
    let request_id = Uuid::new_v4();

    // 0. PII Redaction
    let mut pii_redacted = false;
    for message in &mut request.messages {
        let (redacted, changed) = state.pii_redactor.redact(&message.content);
        if changed {
            message.content = redacted;
            pii_redacted = true;
        }
    }

    // 1. Cache lookup (only for non-streaming)
    if !request.stream {
        let cache_key = hash_request(&request);
        info!("Cache key for request {}: {}", request_id, cache_key);

        // First check exact match cache
        if let Some(cached_response) = state.cache.get(&cache_key) {
            info!("Exact cache hit for request {}", request_id);

            let original_cost = cached_response
                .usage
                .as_ref()
                .map(|u| {
                    state.pricing.calculate_cost(
                        &request.model,
                        u.prompt_tokens,
                        u.completion_tokens,
                    )
                })
                .unwrap_or(0.0);

            // Log cache hit
            let log = RequestLog {
                id: request_id,
                timestamp: Utc::now(),
                provider: "cache".to_string(),
                model: request.model.clone(),
                input_tokens: 0,
                output_tokens: cached_response
                    .usage
                    .as_ref()
                    .map(|u| u.completion_tokens)
                    .unwrap_or(0),
                latency_ms: 0,
                cost_usd: 0.0,
                cost_saved: original_cost,
                quality_score: 1.0,
                cache_hit: true,
                pii_redacted,
                status: "success".to_string(),
                error_message: None,
            };
            let db = state.db.clone();
            tokio::spawn(async move {
                if let Err(e) = db.log_request(&log).await {
                    error!("Failed to log cache hit: {}", e);
                }
            });

            let response = cached_response.value().clone();
            return Ok(Json(response).into_response());
        }

        // Then check semantic cache if available
        if let Some(semantic_cache) = &state.semantic_cache {
            if let Ok(Some(cached_res)) = semantic_cache.get_similar(&request).await {
                info!("Semantic cache hit for request {}", request_id);

                let original_cost = cached_res
                    .response
                    .usage
                    .as_ref()
                    .map(|u| {
                        state.pricing.calculate_cost(
                            &request.model,
                            u.prompt_tokens,
                            u.completion_tokens,
                        )
                    })
                    .unwrap_or(0.0);

                // Log semantic cache hit
                let log = RequestLog {
                    id: request_id,
                    timestamp: Utc::now(),
                    provider: "semantic-cache".to_string(),
                    model: request.model.clone(),
                    input_tokens: 0,
                    output_tokens: cached_res
                        .response
                        .usage
                        .as_ref()
                        .map(|u| u.completion_tokens)
                        .unwrap_or(0),
                    latency_ms: 0,
                    cost_usd: 0.0,
                    cost_saved: original_cost,
                    quality_score: 1.0,
                    cache_hit: true,
                    pii_redacted,
                    status: "success".to_string(),
                    error_message: None,
                };
                let db = state.db.clone();
                tokio::spawn(async move {
                    if let Err(e) = db.log_request(&log).await {
                        error!("Failed to log semantic cache hit: {}", e);
                    }
                });

                return Ok(Json(cached_res.response).into_response());
            }
        }
    }

    if request.stream {
        handle_streaming(state, request, request_id, start_time, pii_redacted).await
    } else {
        handle_smart_routing(state, request, request_id, start_time, pii_redacted).await
    }
}

async fn handle_smart_routing(
    state: Arc<ProxyState>,
    request: ChatRequest,
    request_id: Uuid,
    start_time: chrono::DateTime<Utc>,
    pii_redacted: bool,
) -> Result<axum::response::Response, String> {
    let original_model = request.model.clone();

    let decision = {
        let router = state.smart_router.read().await;
        router
            .route_request(request.clone(), "/v1/chat/completions")
            .await
            .map_err(|e| format!("Routing error: {}", e))?
    };

    let result = {
        let router = state.smart_router.read().await;
        router
            .execute_routing(decision)
            .await
            .map_err(|e| format!("Execution error: {}", e))?
    };

    let latency = Utc::now()
        .signed_duration_since(start_time)
        .num_milliseconds() as u32;

    let actual_cost = result
        .response
        .usage
        .as_ref()
        .map(|u| {
            state.pricing.calculate_cost(
                &result.response.model,
                u.prompt_tokens,
                u.completion_tokens,
            )
        })
        .unwrap_or(0.0);

    let original_cost = result
        .response
        .usage
        .as_ref()
        .map(|u| {
            state
                .pricing
                .calculate_cost(&original_model, u.prompt_tokens, u.completion_tokens)
        })
        .unwrap_or(0.0);

    let cost_saved = (original_cost - actual_cost).max(0.0);

    // Update exact match cache
    let cache_key = hash_request(&request);
    info!("Storing in cache with key: {}", cache_key);
    state.cache.insert(cache_key, result.response.clone());

    // Update semantic cache if enabled
    if let Some(semantic_cache) = &state.semantic_cache {
        let request_copy = request.clone();
        let response_copy = result.response.clone();
        let semantic_cache = semantic_cache.clone();
        tokio::spawn(async move {
            if let Err(e) = semantic_cache.store(&request_copy, &response_copy).await {
                error!("Failed to store in semantic cache: {}", e);
            }
        });
    }

    // Log to DB
    let log = RequestLog {
        id: request_id,
        timestamp: Utc::now(),
        provider: result.routing_type.clone(),
        model: result.response.model.clone(),
        input_tokens: result
            .response
            .usage
            .as_ref()
            .map(|u| u.prompt_tokens)
            .unwrap_or(0),
        output_tokens: result
            .response
            .usage
            .as_ref()
            .map(|u| u.completion_tokens)
            .unwrap_or(0),
        latency_ms: latency,
        cost_usd: actual_cost,
        cost_saved,
        quality_score: result.quality_scores.last().copied().unwrap_or(0.5) as f64,
        cache_hit: false,
        pii_redacted,
        status: "success".to_string(),
        error_message: None,
    };

    let db = state.db.clone();
    tokio::spawn(async move {
        if let Err(e) = db.log_request(&log).await {
            error!("Failed to log request: {}", e);
        }
    });

    Ok(Json(result.response).into_response())
}

async fn handle_streaming(
    state: Arc<ProxyState>,
    mut request: ChatRequest,
    request_id: Uuid,
    _start_time: chrono::DateTime<Utc>,
    _pii_redacted: bool,
) -> Result<axum::response::Response, String> {
    // 1. Get routing decision
    let decision = {
        let router = state.smart_router.read().await;
        router
            .route_request(request.clone(), "/v1/chat/completions")
            .await
            .map_err(|e| format!("Routing error: {}", e))?
    };

    // 2. Extract model from decision
    let routing_type = match decision {
        crate::router::smart::RoutingDecision::Direct(req) => {
            request = req;
            "direct".to_string()
        }
        crate::router::smart::RoutingDecision::Optimized(req, _) => {
            request = req;
            "optimized".to_string()
        }
        crate::router::smart::RoutingDecision::AbTest(req) => {
            request = req;
            "ab_test".to_string()
        }
        crate::router::smart::RoutingDecision::Chain(_, req) => {
            request = req;
            "chain_started".to_string() // Chains are tricky with streaming, we use the first model tier
        }
    };

    // 3. Find provider for the routed model
    let provider = if state.primary_provider.is_healthy() {
        state.primary_provider.provider.clone()
    } else {
        // Try fallback providers
        let mut fallback = None;
        for p in &state.fallback_providers {
            if p.is_healthy() {
                fallback = Some(p.provider.clone());
                break;
            }
        }
        fallback.unwrap_or_else(|| state.primary_provider.provider.clone())
    };

    info!(
        "Streaming request {} routed to {} via {}",
        request_id, request.model, routing_type
    );

    match provider.chat_completion_stream(request).await {
        Ok(stream) => {
            let sse_stream = stream.map(move |chunk_res| match chunk_res {
                Ok(chunk) => match serde_json::to_string(&chunk) {
                    Ok(json) => Ok::<Event, Infallible>(Event::default().data(json)),
                    Err(e) => {
                        Ok::<Event, Infallible>(Event::default().data(format!("Error: {}", e)))
                    }
                },
                Err(e) => Ok::<Event, Infallible>(Event::default().data(format!("Error: {}", e))),
            });
            Ok(Sse::new(sse_stream).into_response())
        }
        Err(e) => Err(format!("Streaming error: {}", e)),
    }
}

use std::convert::Infallible;
