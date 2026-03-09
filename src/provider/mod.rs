use async_trait::async_trait;
use futures_util::stream::BoxStream;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

pub mod anthropic;
pub mod cohere;
pub mod gemini;
pub mod mistral;
pub mod ollama;
pub mod openai;
pub mod perplexity;
pub mod together;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
}

impl Hash for ChatRequest {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.model.hash(state);
        for msg in &self.messages {
            msg.hash(state);
        }
        self.stream.hash(state);
        if let Some(t) = self.temperature {
            t.to_bits().hash(state);
        }
        self.max_tokens.hash(state);
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Hash)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatResponseChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatChunkChoice {
    pub index: u32,
    pub delta: ChatChunkDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatChunkDelta {
    pub role: Option<String>,
    pub content: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[async_trait]
pub trait LlmProvider: Send + Sync + Debug {
    async fn chat_completion(&self, request: ChatRequest) -> anyhow::Result<ChatResponse>;
    async fn chat_completion_stream(
        &self,
        request: ChatRequest,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<ChatResponseChunk>>>;
    async fn health_check(&self) -> anyhow::Result<()>;
    fn name(&self) -> &str;
}
