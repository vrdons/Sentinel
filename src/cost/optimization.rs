use crate::cost::pricing::PricingTable;
use crate::provider::ChatRequest;
use crate::storage::db::{CostOptimization, Database};
use anyhow::Result;
use chrono::{DateTime, Utc};
use md5;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tokio::sync::mpsc;
use tokio::time::{Duration, Instant};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct CostOptimizer {
    pricing: PricingTable,
    db: Arc<Database>,
    // Model performance cache - model -> (avg_quality, avg_latency, cost_per_token)
    model_performance: Arc<RwLock<HashMap<String, ModelPerformance>>>,
    last_update: Arc<RwLock<Option<DateTime<Utc>>>>,
    // Fast suggestion cache - (model, complexity_tier) -> suggestion
    suggestion_cache: Arc<RwLock<HashMap<String, OptimizationSuggestion>>>,
    // Async logging channel
    log_sender: Arc<mpsc::UnboundedSender<CostOptimization>>,
}

#[derive(Debug, Clone)]
struct ModelPerformance {
    avg_quality: f32,
    avg_latency: f32,
    cost_per_token: f64,
    total_requests: u32,
    confidence: f32, // How confident we are in these metrics (based on sample size)
}

#[derive(Debug, Clone)]
pub struct OptimizationSuggestion {
    pub original_model: String,
    pub suggested_model: String,
    pub potential_savings_percent: f32,
    pub confidence_score: f32,
    pub reason: String,
    pub quality_impact: f32, // -1.0 to 1.0, negative means quality decrease
}

impl CostOptimizer {
    pub fn new(pricing: PricingTable, db: Arc<Database>) -> Self {
        let (log_sender, mut log_receiver) = mpsc::unbounded_channel();

        // Spawn background task for async logging
        let db_clone = db.clone();
        tokio::spawn(async move {
            let mut batch = Vec::new();
            let mut last_flush = Instant::now();

            while let Some(optimization) = log_receiver.recv().await {
                batch.push(optimization);

                // Flush batch every 100 items or every 5 seconds
                if batch.len() >= 100 || last_flush.elapsed() > Duration::from_secs(5) {
                    for opt in batch.drain(..) {
                        if let Err(e) = db_clone.log_cost_optimization(&opt).await {
                            eprintln!("Failed to log cost optimization: {}", e);
                        }
                    }
                    last_flush = Instant::now();
                }
            }
        });

        let optimizer = Self {
            pricing,
            db,
            model_performance: Arc::new(RwLock::new(HashMap::new())),
            last_update: Arc::new(RwLock::new(None)),
            suggestion_cache: Arc::new(RwLock::new(HashMap::new())),
            log_sender: Arc::new(log_sender),
        };

        // Pre-populate common optimizations
        optimizer.initialize_fast_suggestions();

        optimizer
    }

    /// Pre-populate suggestion cache with common model optimizations
    fn initialize_fast_suggestions(&self) {
        let mut cache = self.suggestion_cache.write().unwrap();

        // Common expensive -> cheap optimizations for simple requests
        let simple_optimizations = vec![
            ("gpt-4", "gpt-4o-mini", 95.0, "Simple task optimization"),
            (
                "gpt-4o",
                "gpt-4o-mini",
                85.0,
                "Cost optimization for basic query",
            ),
            (
                "gpt-4-turbo",
                "gpt-4o-mini",
                90.0,
                "Significant cost reduction",
            ),
            (
                "claude-opus-4.6",
                "claude-haiku-4.5",
                80.0,
                "Claude tier downgrade",
            ),
            (
                "claude-sonnet-4.6",
                "claude-haiku-4.5",
                70.0,
                "Claude simple task optimization",
            ),
            (
                "o1",
                "gpt-4o-mini",
                98.0,
                "Reasoning model unnecessary for simple task",
            ),
            ("o3", "gpt-4o-mini", 97.0, "Advanced reasoning overkill"),
        ];

        for (original, suggested, savings, reason) in simple_optimizations {
            let key = format!("{}:simple", original);
            cache.insert(
                key,
                OptimizationSuggestion {
                    original_model: original.to_string(),
                    suggested_model: suggested.to_string(),
                    potential_savings_percent: savings,
                    confidence_score: 0.9,
                    reason: reason.to_string(),
                    quality_impact: -0.1, // Slight quality decrease acceptable for simple tasks
                },
            );
        }

        // Medium complexity optimizations (fewer, more conservative)
        let medium_optimizations = vec![
            (
                "gpt-4",
                "gpt-4o",
                60.0,
                "Modern model with better efficiency",
            ),
            (
                "claude-opus-4.6",
                "claude-sonnet-4.6",
                40.0,
                "Balanced capability/cost",
            ),
        ];

        for (original, suggested, savings, reason) in medium_optimizations {
            let key = format!("{}:medium", original);
            cache.insert(
                key,
                OptimizationSuggestion {
                    original_model: original.to_string(),
                    suggested_model: suggested.to_string(),
                    potential_savings_percent: savings,
                    confidence_score: 0.8,
                    reason: reason.to_string(),
                    quality_impact: 0.0, // No quality impact
                },
            );
        }
    }

    /// Analyze a request and suggest optimizations (FAST PATH)
    pub async fn analyze_request(
        &self,
        request: &ChatRequest,
        endpoint: &str,
    ) -> Result<Option<OptimizationSuggestion>> {
        let current_model = &request.model;

        // Skip if it's already an optimized model
        if self.is_already_optimized(current_model) {
            return Ok(None);
        }

        // FAST PATH: Check cache first (no async operations)
        let complexity_tier = self.assess_complexity_tier(request);
        let cache_key = format!("{}:{}", current_model, complexity_tier);

        if let Ok(cache) = self.suggestion_cache.read() {
            if let Some(cached_suggestion) = cache.get(&cache_key) {
                // Found cached suggestion - log async and return immediately
                self.log_optimization_async(cached_suggestion, endpoint);
                return Ok(Some(cached_suggestion.clone()));
            }
        }

        // SLOW PATH: Not in cache - compute suggestion (rare)
        // Spawn background task to avoid blocking
        let optimizer = self.clone();
        let request_clone = request.clone();
        let endpoint_str = endpoint.to_string();
        let cache_key_clone = cache_key.clone();

        tokio::spawn(async move {
            if let Ok(suggestion) = optimizer.compute_suggestion_slow(&request_clone).await {
                if let Some(sugg) = suggestion {
                    // Cache the result for future requests
                    if let Ok(mut cache) = optimizer.suggestion_cache.write() {
                        cache.insert(cache_key_clone, sugg.clone());
                    }
                    // Log it
                    optimizer.log_optimization_async(&sugg, &endpoint_str);
                }
            }
        });

        // Return None immediately to avoid blocking
        // The optimization will be cached for next similar request
        Ok(None)
    }

    /// Fast complexity assessment - returns tier instead of exact score
    fn assess_complexity_tier(&self, request: &ChatRequest) -> &'static str {
        let total_chars: usize = request.messages.iter().map(|m| m.content.len()).sum();

        // Fast heuristic based on input length and keywords
        if total_chars < 50 {
            "simple"
        } else if total_chars > 500 || self.has_complex_keywords(&request.messages[0].content) {
            "complex"
        } else {
            "medium"
        }
    }

    /// Quick keyword check for complex tasks
    fn has_complex_keywords(&self, content: &str) -> bool {
        let complex_keywords = [
            "explain",
            "analyze",
            "design",
            "implement",
            "algorithm",
            "strategy",
        ];
        let content_lower = content.to_lowercase();
        complex_keywords
            .iter()
            .any(|&keyword| content_lower.contains(keyword))
    }

    /// Log optimization suggestion asynchronously (non-blocking)
    fn log_optimization_async(&self, suggestion: &OptimizationSuggestion, endpoint: &str) {
        let optimization = CostOptimization {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            endpoint_pattern: endpoint.to_string(),
            original_model: suggestion.original_model.clone(),
            suggested_model: suggestion.suggested_model.clone(),
            potential_savings_percent: suggestion.potential_savings_percent as f64,
            confidence_score: suggestion.confidence_score as f64,
            reason: suggestion.reason.clone(),
            status: "applied".to_string(),
        };

        // Send to background logger - never blocks
        let _ = self.log_sender.send(optimization);
    }

    /// Compute suggestion using slow path (for cache misses)
    async fn compute_suggestion_slow(
        &self,
        request: &ChatRequest,
    ) -> Result<Option<OptimizationSuggestion>> {
        // This can be expensive since it's async
        self.maybe_update_performance_cache().await?;

        let complexity_score = self.assess_complexity(request);
        let current_model = &request.model;

        match complexity_score {
            score if score < 0.3 => {
                self.suggest_cheaper_model(current_model, complexity_score)
                    .await
            }
            score if score > 0.8 => {
                self.suggest_better_model(current_model, complexity_score)
                    .await
            }
            _ => {
                self.suggest_balanced_model(current_model, complexity_score)
                    .await
            }
        }
    }

    /// Calculate actual savings after a request is completed
    pub fn calculate_savings(
        &self,
        original_model: &str,
        used_model: &str,
        tokens: (u32, u32),
    ) -> f64 {
        let original_cost = self
            .pricing
            .calculate_cost(original_model, tokens.0, tokens.1);
        let actual_cost = self.pricing.calculate_cost(used_model, tokens.0, tokens.1);
        original_cost - actual_cost
    }

    /// Update our model performance cache from recent data if needed (every 5 mins)
    async fn maybe_update_performance_cache(&self) -> Result<()> {
        {
            let last_upd = self.last_update.read().unwrap();
            if let Some(last) = *last_upd {
                if Utc::now().signed_duration_since(last).num_minutes() < 5 {
                    return Ok(());
                }
            }
        }

        let stats = self.db.get_model_performance_stats().await?;

        let mut new_cache = HashMap::new();
        if let Some(models) = stats.get("models").and_then(|m| m.as_array()) {
            for model_data in models {
                if let (
                    Some(model),
                    Some(requests),
                    Some(avg_latency),
                    Some(avg_cost),
                    Some(avg_quality),
                ) = (
                    model_data.get("model").and_then(|m| m.as_str()),
                    model_data.get("requests").and_then(|r| r.as_u64()),
                    model_data.get("avg_latency").and_then(|l| l.as_f64()),
                    model_data.get("avg_cost").and_then(|c| c.as_f64()),
                    model_data.get("avg_quality").and_then(|q| q.as_f64()),
                ) {
                    let confidence = self.calculate_confidence(requests as u32);

                    new_cache.insert(
                        model.to_string(),
                        ModelPerformance {
                            avg_quality: avg_quality as f32,
                            avg_latency: avg_latency as f32,
                            cost_per_token: avg_cost,
                            total_requests: requests as u32,
                            confidence,
                        },
                    );
                }
            }
        }

        {
            let mut cache = self.model_performance.write().unwrap();
            *cache = new_cache;
        }
        {
            let mut last_upd = self.last_update.write().unwrap();
            *last_upd = Some(Utc::now());
        }

        Ok(())
    }

    /// Assess the complexity of a request (0.0 = simple, 1.0 = complex)
    pub fn assess_complexity(&self, request: &ChatRequest) -> f32 {
        let mut complexity = 0.0;

        // Message count factor
        let msg_count = request.messages.len() as f32;
        complexity += (msg_count / 10.0).min(0.2); // Up to 0.2 for message count

        // Token count estimation (rough)
        let total_chars: usize = request.messages.iter().map(|m| m.content.len()).sum();
        let estimated_tokens = total_chars / 4; // Rough tokens = chars/4
        complexity += (estimated_tokens as f32 / 4000.0).min(0.3); // Up to 0.3 for length

        // Content analysis
        let combined_content = request
            .messages
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();

        // Look for complex task indicators
        let complex_keywords = [
            "analyze",
            "explain",
            "reasoning",
            "complex",
            "detailed",
            "comprehensive",
            "code",
            "programming",
            "algorithm",
            "math",
            "calculation",
            "proof",
            "research",
            "essay",
            "report",
            "summary",
            "translation",
            "creative",
        ];

        let simple_keywords = [
            "hello", "hi", "yes", "no", "ok", "thanks", "simple", "basic", "quick",
        ];

        for keyword in complex_keywords.iter() {
            if combined_content.contains(keyword) {
                complexity += 0.1;
            }
        }

        for keyword in simple_keywords.iter() {
            if combined_content.contains(keyword) {
                complexity = (complexity - 0.1).max(0.0);
            }
        }

        // Temperature factor
        if let Some(temp) = request.temperature {
            if temp > 0.7 {
                complexity += 0.1; // Creative tasks are usually more complex
            }
        }

        complexity.min(1.0)
    }

    async fn suggest_cheaper_model(
        &self,
        current_model: &str,
        _complexity: f32,
    ) -> Result<Option<OptimizationSuggestion>> {
        // Define model hierarchy (most expensive to least expensive)
        let cheaper_alternatives = match current_model {
            m if m.contains("gpt-4o") && !m.contains("mini") => {
                vec!["gpt-4o-mini", "gpt-3.5-turbo"]
            }
            m if m.contains("gpt-4") => vec!["gpt-4o-mini", "gpt-3.5-turbo"],
            m if m.contains("claude-3-opus") => {
                vec!["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"]
            }
            m if m.contains("claude-3-sonnet") || m.contains("claude-3-5-sonnet") => {
                vec!["claude-3-5-haiku-20241022"]
            }
            m if m.contains("gemini-1.5-pro") => vec!["gemini-1.5-flash"],
            m if m.contains("command-r-plus") => vec!["command-r"],
            m if m.contains("mistral-large") => vec!["mistral-small-latest"],
            _ => vec![],
        };

        for alternative in cheaper_alternatives {
            let original_cost = self.pricing.calculate_cost(current_model, 1000, 1000);
            let alt_cost = self.pricing.calculate_cost(alternative, 1000, 1000);

            if alt_cost < original_cost {
                let savings_percent = ((original_cost - alt_cost) / original_cost * 100.0) as f32;

                // Check if we have performance data for quality impact
                let quality_impact = self.estimate_quality_impact(current_model, alternative);

                if savings_percent > 20.0 {
                    // Only suggest if savings > 20%
                    return Ok(Some(OptimizationSuggestion {
                        original_model: current_model.to_string(),
                        suggested_model: alternative.to_string(),
                        potential_savings_percent: savings_percent,
                        confidence_score: 0.8,
                        reason: format!(
                            "Simple request detected. {} can handle this for {:.0}% less cost",
                            alternative, savings_percent
                        ),
                        quality_impact,
                    }));
                }
            }
        }

        Ok(None)
    }

    async fn suggest_better_model(
        &self,
        current_model: &str,
        _complexity: f32,
    ) -> Result<Option<OptimizationSuggestion>> {
        // For complex tasks, suggest better models if current one is too basic
        let better_alternatives = match current_model {
            m if m.contains("gpt-3.5") => vec!["gpt-4o", "gpt-4o-mini"],
            m if m.contains("gpt-4o-mini") => vec!["gpt-4o"],
            m if m.contains("claude-3-haiku") => vec!["claude-3-5-sonnet-20241022"],
            m if m.contains("gemini-1.5-flash") => vec!["gemini-1.5-pro"],
            m if m.contains("command-r") && !m.contains("plus") => vec!["command-r-plus"],
            m if m.contains("mistral-small") => vec!["mistral-large-latest"],
            _ => vec![],
        };

        for alternative in better_alternatives {
            let quality_impact = self.estimate_quality_impact(current_model, alternative);

            if quality_impact > 0.2 {
                // Only suggest if significant quality improvement
                return Ok(Some(OptimizationSuggestion {
                    original_model: current_model.to_string(),
                    suggested_model: alternative.to_string(),
                    potential_savings_percent: 0.0, // This is about quality, not cost
                    confidence_score: 0.7,
                    reason: format!(
                        "Complex request detected. {} provides better reasoning for complex tasks",
                        alternative
                    ),
                    quality_impact,
                }));
            }
        }

        Ok(None)
    }

    async fn suggest_balanced_model(
        &self,
        current_model: &str,
        _complexity: f32,
    ) -> Result<Option<OptimizationSuggestion>> {
        // For medium complexity, look for the best quality/cost ratio
        let alternatives = vec![
            "gpt-4o-mini",
            "claude-3-5-haiku-20241022",
            "gemini-1.5-flash",
            "mistral-small-latest",
            "command-r",
        ];

        let current_cost = self.pricing.calculate_cost(current_model, 1000, 1000);
        let cache = self.model_performance.read().unwrap();
        let current_performance = cache.get(current_model);

        let mut best_alternative = None;
        let mut best_score = 0.0;

        for alternative in alternatives {
            if alternative == current_model {
                continue;
            }

            let alt_cost = self.pricing.calculate_cost(alternative, 1000, 1000);
            let alt_performance = cache.get(alternative);

            if let (Some(current_perf), Some(alt_perf)) = (current_performance, alt_performance) {
                // Calculate quality per dollar score
                let current_qpd = if current_cost > 0.0 {
                    current_perf.avg_quality as f64 / current_cost
                } else {
                    0.0
                };
                let alt_qpd = if alt_cost > 0.0 {
                    alt_perf.avg_quality as f64 / alt_cost
                } else {
                    0.0
                };

                if alt_qpd > current_qpd && alt_qpd > best_score {
                    best_score = alt_qpd;
                    let savings_percent = if current_cost > 0.0 {
                        ((current_cost - alt_cost) / current_cost * 100.0) as f32
                    } else {
                        0.0
                    };

                    best_alternative = Some(OptimizationSuggestion {
                        original_model: current_model.to_string(),
                        suggested_model: alternative.to_string(),
                        potential_savings_percent: savings_percent,
                        confidence_score: alt_perf.confidence,
                        reason: format!(
                            "{} offers better quality per dollar for medium complexity tasks",
                            alternative
                        ),
                        quality_impact: alt_perf.avg_quality - current_perf.avg_quality,
                    });
                }
            }
        }

        Ok(best_alternative)
    }

    fn estimate_quality_impact(&self, from_model: &str, to_model: &str) -> f32 {
        // If we have actual performance data, use it
        let cache = self.model_performance.read().unwrap();
        if let (Some(from_perf), Some(to_perf)) = (cache.get(from_model), cache.get(to_model)) {
            return to_perf.avg_quality - from_perf.avg_quality;
        }

        // Otherwise, use model tier estimates
        let model_tiers = [
            ("gpt-4o", 0.9),
            ("claude-3-opus-20240229", 0.9),
            ("claude-3-5-sonnet-20241022", 0.85),
            ("gpt-4o-mini", 0.8),
            ("gemini-1.5-pro", 0.8),
            ("mistral-large-latest", 0.75),
            ("claude-3-5-haiku-20241022", 0.7),
            ("gemini-1.5-flash", 0.7),
            ("gpt-3.5-turbo", 0.6),
            ("command-r", 0.6),
            ("mistral-small-latest", 0.5),
        ];

        let from_quality = model_tiers
            .iter()
            .find(|(model, _)| from_model.contains(model))
            .map(|(_, quality)| *quality)
            .unwrap_or(0.5);

        let to_quality = model_tiers
            .iter()
            .find(|(model, _)| to_model.contains(model))
            .map(|(_, quality)| *quality)
            .unwrap_or(0.5);

        to_quality - from_quality
    }

    fn is_already_optimized(&self, model_name: &str) -> bool {
        // Consider these models already optimized
        match model_name {
            "gpt-4o-mini"
            | "claude-3-5-haiku-20241022"
            | "gemini-1.5-flash"
            | "mistral-small-latest"
            | "command-r" => true,
            m if m.contains("ollama") => true,
            _ => false,
        }
    }

    fn calculate_confidence(&self, sample_size: u32) -> f32 {
        // Confidence based on sample size (sigmoid curve)
        let x = sample_size as f32;
        (2.0 / (1.0 + (-x / 50.0).exp()) - 1.0).max(0.1).min(0.95)
    }

    /// Get prompt hash for grouping similar requests
    pub fn get_prompt_hash(&self, request: &ChatRequest) -> String {
        let mut context = md5::Context::new();

        // Hash the essential parts of the request
        context.consume(request.model.as_bytes());
        for msg in &request.messages {
            context.consume(msg.role.as_bytes());
            // Hash first 200 chars to group similar prompts while preserving privacy
            let content_preview = if msg.content.len() > 200 {
                &msg.content[..200]
            } else {
                &msg.content
            };
            context.consume(content_preview.as_bytes());
        }

        format!("{:x}", context.compute())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ChatMessage;

    #[tokio::test]
    async fn test_complexity_assessment() {
        let pricing = PricingTable::default();
        let optimizer = CostOptimizer::new(
            pricing,
            Arc::new(
                Database::new(&std::path::PathBuf::from(":memory:"))
                    .await
                    .unwrap(),
            ),
        );

        // Simple request
        let simple_request = ChatRequest {
            model: "gpt-4o".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "Hi there!".to_string(),
            }],
            stream: false,
            temperature: None,
            max_tokens: None,
        };

        assert!(optimizer.assess_complexity(&simple_request) < 0.3);

        // Complex request
        let complex_request = ChatRequest {
            model: "gpt-4o".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "Please analyze this complex algorithm and provide a detailed explanation of its time complexity, including mathematical proof and code optimization suggestions.".to_string(),
            }],
            stream: false,
            temperature: Some(0.8),
            max_tokens: None,
        };

        assert!(optimizer.assess_complexity(&complex_request) > 0.7);
    }
}
