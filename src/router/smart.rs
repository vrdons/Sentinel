use crate::provider::{ChatRequest, ChatResponse, LlmProvider};
use crate::storage::db::Database;
use crate::cost::optimization::{CostOptimizer, OptimizationSuggestion};
use std::sync::Arc;
use std::collections::HashMap;
use uuid::Uuid;
use chrono::Utc;
use anyhow::Result;
use serde::{Serialize, Deserialize};
use rand::Rng;
use tracing::{info, debug, warn};

pub use crate::config::RouterConfig as SmartRouterConfig;

#[derive(Debug, Clone)]
pub struct ModelCandidate {
    pub model_name: String,
    pub provider: Arc<dyn LlmProvider>,
    pub predicted_quality: f32,
    pub predicted_cost: f64,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainStep {
    pub model: String,
    pub purpose: String,        // "analyze", "simplify", "verify", etc.
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningChain {
    pub name: String,
    pub description: String,
    pub steps: Vec<ChainStep>,
    pub complexity_threshold: f32, // Only use this chain for requests above this complexity
}

pub struct SmartRouter {
    pub config: SmartRouterConfig,
    db: Arc<Database>,
    cost_optimizer: CostOptimizer,
    providers: HashMap<String, Arc<dyn LlmProvider>>,
    reasoning_chains: Vec<ReasoningChain>,
    ab_tests: HashMap<String, AbTest>,
}

#[derive(Debug, Clone)]
struct AbTest {
    name: String,
    models: Vec<String>,
    traffic_split: Vec<f32>, // Should sum to 1.0
    start_time: chrono::DateTime<Utc>,
    end_time: Option<chrono::DateTime<Utc>>,
    active: bool,
}

impl SmartRouter {
    pub fn new(
        config: SmartRouterConfig,
        db: Arc<Database>,
        cost_optimizer: CostOptimizer,
        providers: HashMap<String, Arc<dyn LlmProvider>>,
    ) -> Self {
        let reasoning_chains = Self::default_reasoning_chains();
        
        Self {
            config,
            db,
            cost_optimizer,
            providers,
            reasoning_chains,
            ab_tests: HashMap::new(),
        }
    }

    /// Main routing logic - decides which model(s) to use
    pub async fn route_request(&self, mut request: ChatRequest, endpoint: &str) -> Result<RoutingDecision> {
        // Step 0: Explicit endpoint mapping (Tiered Autonomy - Default)
        if let Some(mapped_model) = self.config.endpoints.get(endpoint) {
            debug!("Using explicit mapping for endpoint {}: {}", endpoint, mapped_model);
            request.model = mapped_model.clone();
            return Ok(RoutingDecision::Direct(request));
        }

        if !self.config.smart_mode {
            return Ok(RoutingDecision::Direct(request));
        }

        debug!("Smart routing request for model: {}", request.model);
        
        // Step 1: Assess complexity for potential reasoning chains
        let complexity = self.assess_complexity(&request);
        if let Some(chain) = self.select_reasoning_chain(complexity) {
            info!("Routing to reasoning chain: {}", chain.name);
            return Ok(RoutingDecision::Chain(chain, request));
        }

        // Step 2: Check for cost optimization opportunities
        // (In Smart Mode, we always analyze for optimizations)
        let optimization = self.cost_optimizer.analyze_request(&request, endpoint).await?;

        // Step 3: Check for A/B testing
        if let Some(ab_test_model) = self.check_ab_test(&request).await? {
            info!("A/B testing: routing to {}", ab_test_model);
            request.model = ab_test_model;
            return Ok(RoutingDecision::AbTest(request));
        }

        // Step 4: Apply optimization if found and confidence is high
        if let Some(opt) = optimization {
            if opt.confidence_score > 0.7 && self.should_apply_optimization(&opt) {
                info!("Applying optimization: {} -> {} ({}% savings)", 
                     opt.original_model, opt.suggested_model, opt.potential_savings_percent);
                request.model = opt.suggested_model.clone();
                return Ok(RoutingDecision::Optimized(request, opt));
            }
        }

        Ok(RoutingDecision::Direct(request))
    }

    /// Execute a routing decision
    pub async fn execute_routing(&self, decision: RoutingDecision) -> Result<SmartRoutingResult> {
        match decision {
            RoutingDecision::Direct(request) => {
                self.execute_with_fallback(request, "direct", None).await
            },
            
            RoutingDecision::Optimized(request, optimization) => {
                self.execute_with_fallback(request, "optimized", Some(optimization)).await
            },
            
            RoutingDecision::Chain(chain, original_request) => {
                self.execute_reasoning_chain(chain, original_request).await
            },
            
            RoutingDecision::AbTest(request) => {
                self.execute_with_fallback(request, "ab_test", None).await
            }
        }
    }

    /// Execute a request with potential semantic fallback
    async fn execute_with_fallback(
        &self,
        request: ChatRequest,
        routing_type: &str,
        optimization: Option<OptimizationSuggestion>
    ) -> Result<SmartRoutingResult> {
        let mut current_model = request.model.clone();
        let mut attempts = 0;
        let mut models_used = Vec::new();
        let mut quality_scores = Vec::new();
        let mut total_latency = 0u32;
        let mut final_response = None;
        let mut final_routing_type = routing_type.to_string();
        let start_routing_time = std::time::Instant::now();

        loop {
            attempts += 1;
            models_used.push(current_model.clone());

            let provider = self.get_provider(&current_model)?;
            let result = provider.chat_completion(ChatRequest {
                model: current_model.clone(),
                ..request.clone()
            }).await;

            total_latency = start_routing_time.elapsed().as_millis() as u32;

            match result {
                Ok(response) => {
                    let quality = self.score_response_quality(&response, "general");

                    // Check if quality is sufficient or we should fallback
                    if quality >= self.config.quality_threshold || attempts >= 2 || current_model.contains("claude") {
                        final_response = Some(response);
                        quality_scores.push(quality);
                        break;
                    } else {
                        info!("Low quality response from {} (score: {:.2}). Attempting quality rescue.", current_model, quality);
                        quality_scores.push(quality);
                        final_routing_type = "quality_rescue".to_string();
                    }
                },
                Err(e) => {
                    warn!("Request to {} failed: {}. Attempting fallback.", current_model, e);
                    final_routing_type = "fallback".to_string();
                    if attempts >= 2 {
                        return Err(e);
                    }
                }
            }

            // Find fallback model
            if let Some(fallback_model) = self.find_fallback_model(&current_model) {
                current_model = fallback_model;
            } else {
                // If no explicit fallback, try a known "Superior" model based on provider type
                if current_model.contains("claude") {
                    current_model = "claude-3-haiku-20240307".to_string(); // Use the same model that we know works
                } else if current_model.contains("gpt") {
                    current_model = "gpt-4o".to_string();
                } else if current_model.contains("gemini") {
                    current_model = "gemini-pro".to_string();
                } else {
                    current_model = "gpt-4o".to_string(); // Default fallback
                }
            }
        }

        Ok(SmartRoutingResult {
            response: final_response.expect("Should have a response"),
            routing_type: final_routing_type,
            models_used,
            total_latency,
            optimization_applied: optimization,
            quality_scores,
        })
    }

    fn find_fallback_model(&self, current_model: &str) -> Option<String> {
        self.config.fallback_rules.iter()
            .find(|r| r.on_model == current_model)
            .map(|r| r.fallback_to.clone())
    }

    /// Execute a multi-model reasoning chain
    async fn execute_reasoning_chain(&self, chain: ReasoningChain, original_request: ChatRequest) -> Result<SmartRoutingResult> {
        let mut current_messages = original_request.messages.clone();
        let mut models_used = Vec::new();
        let mut quality_scores = Vec::new();
        let mut total_latency = 0u32;

        for (i, step) in chain.steps.iter().enumerate() {
            info!("Executing chain step {}: {} with {}", i + 1, step.purpose, step.model);
            
            let step_request = ChatRequest {
                model: step.model.clone(),
                messages: current_messages.clone(),
                stream: false,
                temperature: step.temperature.or(original_request.temperature),
                max_tokens: step.max_tokens.or(original_request.max_tokens),
            };

            let provider = self.get_provider(&step.model)?;
            let step_start = std::time::Instant::now();
            let step_response = provider.chat_completion(step_request).await?;
            let step_latency = step_start.elapsed().as_millis() as u32;
            
            total_latency += step_latency;
            models_used.push(step.model.clone());

            // Score this step's quality
            let quality_score = self.score_response_quality(&step_response, &step.purpose);
            quality_scores.push(quality_score);

            // Add the response to context for next step
            if let Some(choice) = step_response.choices.first() {
                current_messages.push(crate::provider::ChatMessage {
                    role: "assistant".to_string(),
                    content: choice.message.content.clone(),
                });
                
                // If this is not the last step, add a context message for the next model
                if i < chain.steps.len() - 1 {
                    let next_step = &chain.steps[i + 1];
                    current_messages.push(crate::provider::ChatMessage {
                        role: "user".to_string(),
                        content: format!("Now, please {} the above response.", next_step.purpose),
                    });
                }
            }
        }

        // Create final response using the last step's output
        let final_response = self.create_chain_response(&chain, &current_messages, total_latency);

        Ok(SmartRoutingResult {
            response: final_response,
            routing_type: "chain".to_string(),
            models_used,
            total_latency,
            optimization_applied: None,
            quality_scores,
        })
    }

    fn create_chain_response(&self, chain: &ReasoningChain, messages: &[crate::provider::ChatMessage], _latency: u32) -> ChatResponse {
        // Find the last assistant message
        let final_content = messages.iter()
            .rev()
            .find(|msg| msg.role == "assistant")
            .map(|msg| msg.content.clone())
            .unwrap_or_else(|| "Chain execution completed".to_string());

        ChatResponse {
            id: Uuid::new_v4().to_string(),
            object: "chat.completion".to_string(),
            created: Utc::now().timestamp() as u64,
            model: format!("chain:{}", chain.name),
            choices: vec![crate::provider::ChatChoice {
                index: 0,
                message: crate::provider::ChatMessage {
                    role: "assistant".to_string(),
                    content: final_content,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: None, // TODO: Calculate combined usage
        }
    }

    fn score_response_quality(&self, response: &ChatResponse, purpose: &str) -> f32 {
        // Simple heuristic quality scoring
        if let Some(choice) = response.choices.first() {
            let content = &choice.message.content;
            let mut score: f32 = 0.5; // Base score
            
            // Length-based scoring (not too short, not too verbose)
            match content.len() {
                0..=10 => score *= 0.3,
                11..=50 => score *= 0.7,
                51..=500 => score *= 1.0,
                501..=2000 => score *= 0.9,
                _ => score *= 0.8,
            }
            
            // Purpose-specific scoring
            match purpose {
                "analyze" => {
                    if content.to_lowercase().contains("because") || 
                       content.to_lowercase().contains("therefore") {
                        score *= 1.2;
                    }
                },
                "simplify" => {
                    if content.len() < 300 && !content.contains("complex") {
                        score *= 1.1;
                    }
                },
                "verify" => {
                    if content.to_lowercase().contains("correct") || 
                       content.to_lowercase().contains("accurate") {
                        score *= 1.1;
                    }
                },
                _ => {}
            }
            
            score.min(1.0)
        } else {
            0.1 // Very low score for empty response
        }
    }

    fn assess_complexity(&self, request: &ChatRequest) -> f32 {
        // Use the same complexity assessment as the cost optimizer
        self.cost_optimizer.assess_complexity(request)
    }

    fn select_reasoning_chain(&self, complexity: f32) -> Option<ReasoningChain> {
        self.reasoning_chains.iter()
            .filter(|chain| complexity >= chain.complexity_threshold)
            .max_by(|a, b| a.complexity_threshold.partial_cmp(&b.complexity_threshold).unwrap())
            .cloned()
    }

    async fn check_ab_test(&self, request: &ChatRequest) -> Result<Option<String>> {
        let mut rng = rand::thread_rng();
        
        // Find active A/B test for this model
        for ab_test in self.ab_tests.values() {
            if ab_test.active && ab_test.models.contains(&request.model) {
                let test_roll: f32 = rng.gen();
                let mut cumulative = 0.0;

                for (i, &split) in ab_test.traffic_split.iter().enumerate() {
                    cumulative += split;
                    if test_roll < cumulative {
                        return Ok(Some(ab_test.models[i].clone()));
                    }
                }
            }
        }

        Ok(None)
    }

    fn should_apply_optimization(&self, optimization: &OptimizationSuggestion) -> bool {
        // Only apply if quality impact is acceptable
        if optimization.quality_impact < -0.2 {
            return false; // Too much quality loss
        }
        
        // Only apply if savings are significant
        if optimization.potential_savings_percent < 15.0 {
            return false;
        }
        
        true
    }

    fn get_provider(&self, model: &str) -> Result<Arc<dyn LlmProvider>> {
        // Map model names to provider names
        let provider_name = if model.contains("gpt") {
            "openai"
        } else if model.contains("claude") {
            "anthropic"
        } else if model.contains("gemini") {
            "gemini"
        } else if model.contains("mistral") {
            "mistral"
        } else if model.contains("command") {
            "cohere"
        } else if model.contains("pplx") {
            "perplexity"
        } else if model.contains("llama") && !model.contains("ollama") {
            "together"
        } else if model.contains("ollama") || model.contains("llama3:") {
            "ollama"
        } else {
            model // Assume provider name matches model name
        };

        self.providers.get(provider_name)
            .ok_or_else(|| anyhow::anyhow!("Provider not found for model: {}", model))
            .map(|p| p.clone())
    }

    pub fn add_ab_test(&mut self, name: String, models: Vec<String>, traffic_split: Vec<f32>) {
        let ab_test = AbTest {
            name: name.clone(),
            models,
            traffic_split,
            start_time: Utc::now(),
            end_time: None,
            active: true,
        };
        self.ab_tests.insert(name, ab_test);
    }

    pub fn stop_ab_test(&mut self, name: &str) {
        if let Some(test) = self.ab_tests.get_mut(name) {
            test.active = false;
            test.end_time = Some(Utc::now());
        }
    }

    fn default_reasoning_chains() -> Vec<ReasoningChain> {
        vec![
            ReasoningChain {
                name: "analyze_then_simplify".to_string(),
                description: "Use a powerful model to analyze, then a fast model to simplify".to_string(),
                complexity_threshold: 0.8,
                steps: vec![
                    ChainStep {
                        model: "gpt-4o".to_string(),
                        purpose: "analyze".to_string(),
                        temperature: Some(0.3),
                        max_tokens: Some(1000),
                    },
                    ChainStep {
                        model: "gpt-4o-mini".to_string(),
                        purpose: "simplify".to_string(),
                        temperature: Some(0.7),
                        max_tokens: Some(500),
                    },
                ],
            },
            ReasoningChain {
                name: "dual_verification".to_string(),
                description: "Use two different models and cross-verify results".to_string(),
                complexity_threshold: 0.9,
                steps: vec![
                    ChainStep {
                        model: "gpt-4o".to_string(),
                        purpose: "solve".to_string(),
                        temperature: Some(0.1),
                        max_tokens: Some(1000),
                    },
                    ChainStep {
                        model: "claude-3-5-sonnet-20241022".to_string(),
                        purpose: "verify and improve".to_string(),
                        temperature: Some(0.2),
                        max_tokens: Some(1000),
                    },
                ],
            },
        ]
    }
}

#[derive(Debug)]
pub enum RoutingDecision {
    Direct(ChatRequest),
    Optimized(ChatRequest, OptimizationSuggestion),
    Chain(ReasoningChain, ChatRequest),
    AbTest(ChatRequest),
}

#[derive(Debug)]
pub struct SmartRoutingResult {
    pub response: ChatResponse,
    pub routing_type: String,
    pub models_used: Vec<String>,
    pub total_latency: u32,
    pub optimization_applied: Option<OptimizationSuggestion>,
    pub quality_scores: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complexity_based_chain_selection() {
        let chains = SmartRouter::default_reasoning_chains();
        
        // Simple request shouldn't trigger any chain
        let simple_complexity = 0.3;
        let selected = chains.iter()
            .filter(|chain| simple_complexity >= chain.complexity_threshold)
            .max_by(|a, b| a.complexity_threshold.partial_cmp(&b.complexity_threshold).unwrap());
        assert!(selected.is_none());
        
        // Complex request should trigger highest applicable chain
        let complex_complexity = 0.9;
        let selected = chains.iter()
            .filter(|chain| complex_complexity >= chain.complexity_threshold)
            .max_by(|a, b| a.complexity_threshold.partial_cmp(&b.complexity_threshold).unwrap());
        assert!(selected.is_some());
        assert_eq!(selected.unwrap().name, "dual_verification");
    }
}
