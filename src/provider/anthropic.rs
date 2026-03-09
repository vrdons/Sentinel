use crate::provider::{
    ChatChoice, ChatChunkChoice, ChatChunkDelta, ChatMessage, ChatRequest, ChatResponse,
    ChatResponseChunk, LlmProvider, Usage,
};
use async_trait::async_trait;
use futures_util::stream::{BoxStream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub struct AnthropicProvider {
    pub api_key: String,
    pub client: Client,
    pub base_url: String,
}

impl AnthropicProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: Client::new(),
            base_url: "https://api.anthropic.com/v1".to_string(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicResponse {
    id: String,
    model: String,
    content: Vec<AnthropicContent>,
    role: String,
    usage: AnthropicUsage,
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicContent {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicStreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: AnthropicResponse },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: u32, delta: AnthropicDelta },
    #[serde(rename = "message_delta")]
    #[allow(dead_code)]
    MessageDelta { usage: AnthropicUsage },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(other)]
    Unknown,
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    async fn chat_completion(&self, request: ChatRequest) -> anyhow::Result<ChatResponse> {
        let anthropic_request = AnthropicRequest {
            model: request.model.clone(),
            messages: request
                .messages
                .iter()
                .map(|m| AnthropicMessage {
                    role: m.role.clone(),
                    content: m.content.clone(),
                })
                .collect(),
            max_tokens: request.max_tokens.unwrap_or(1024),
            temperature: request.temperature,
            stream: if request.stream { Some(true) } else { None },
        };

        let response = self
            .client
            .post(format!("{}/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&anthropic_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Anthropic API error: {}", error_text));
        }

        let anth_res = response.json::<AnthropicResponse>().await?;

        let chat_response = ChatResponse {
            id: anth_res.id,
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp() as u64,
            model: anth_res.model,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: anth_res.role,
                    content: anth_res
                        .content
                        .first()
                        .map(|c| c.text.clone())
                        .unwrap_or_default(),
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(Usage {
                prompt_tokens: anth_res.usage.input_tokens,
                completion_tokens: anth_res.usage.output_tokens,
                total_tokens: anth_res.usage.input_tokens + anth_res.usage.output_tokens,
            }),
        };

        Ok(chat_response)
    }

    async fn chat_completion_stream(
        &self,
        request: ChatRequest,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<ChatResponseChunk>>> {
        let anthropic_request = AnthropicRequest {
            model: request.model.clone(),
            messages: request
                .messages
                .iter()
                .map(|m| AnthropicMessage {
                    role: m.role.clone(),
                    content: m.content.clone(),
                })
                .collect(),
            max_tokens: request.max_tokens.unwrap_or(1024),
            temperature: request.temperature,
            stream: Some(true),
        };

        let response = self
            .client
            .post(format!("{}/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&anthropic_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Anthropic API error: {}", error_text));
        }

        let mut buffer = String::new();
        let mut current_id = String::new();
        let mut current_model = request.model.clone();

        let stream = response.bytes_stream().flat_map(move |item| match item {
            Ok(bytes) => {
                buffer.push_str(&String::from_utf8_lossy(&bytes));
                let mut chunks = Vec::new();
                while let Some(line_end) = buffer.find('\n') {
                    let line = buffer.drain(..line_end + 1).collect::<String>();
                    let line = line.trim();
                    if let Some(data) = line.strip_prefix("data: ")
                        && let Ok(event) = serde_json::from_str::<AnthropicStreamEvent>(data)
                    {
                        match event {
                            AnthropicStreamEvent::MessageStart { message } => {
                                current_id = message.id.clone();
                                current_model = message.model.clone();
                            }
                            AnthropicStreamEvent::ContentBlockDelta {
                                index,
                                delta: AnthropicDelta::TextDelta { text },
                            } => {
                                chunks.push(Ok(ChatResponseChunk {
                                    id: current_id.clone(),
                                    object: "chat.completion.chunk".to_string(),
                                    created: Utc::now().timestamp() as u64,
                                    model: current_model.clone(),
                                    choices: vec![ChatChunkChoice {
                                        index,
                                        delta: ChatChunkDelta {
                                            role: None,
                                            content: Some(text),
                                        },
                                        finish_reason: None,
                                    }],
                                }));
                            }
                            _ => {}
                        }
                    }
                }
                futures_util::stream::iter(chunks)
            }
            Err(e) => futures_util::stream::iter(vec![Err(anyhow::anyhow!("Stream error: {}", e))]),
        });

        Ok(Box::pin(stream))
    }

    async fn health_check(&self) -> anyhow::Result<()> {
        // Anthropic doesn't have a simple health endpoint, so we make a minimal request
        let test_request = AnthropicRequest {
            model: "claude-3-haiku-20240307".to_string(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: "Hi".to_string(),
            }],
            max_tokens: 1,
            temperature: None,
            stream: None,
        };

        let response = self
            .client
            .post(format!("{}/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&test_request)
            .send()
            .await?;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(anyhow::anyhow!(
                "Anthropic health check failed: {}",
                response.status()
            ))
        }
    }

    fn name(&self) -> &str {
        "anthropic"
    }
}

use chrono::Utc;
