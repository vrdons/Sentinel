pub mod semantic;

use crate::provider::{ChatRequest, ChatResponse};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

pub fn hash_request(request: &ChatRequest) -> u64 {
    let mut hasher = DefaultHasher::new();
    request.hash(&mut hasher);
    hasher.finish()
}

#[derive(Debug, Clone)]
pub struct CachedResponse {
    pub response: ChatResponse,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub embedding: Option<Vec<f32>>,
}