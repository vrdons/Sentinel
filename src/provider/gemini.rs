use crate::provider::{ChatRequest, ChatResponse, ChatResponseChunk, ChatChoice, ChatChunkChoice, ChatMessage, ChatChunkDelta, LlmProvider, Usage};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use futures_util::stream::{BoxStream, StreamExt};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug)]
pub struct GeminiProvider {
    pub api_key: String,
    pub client: Client,
    pub base_url: String,
}

impl GeminiProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: Client::new(),
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
        }
    }

    fn openai_to_gemini_request(&self, request: &ChatRequest) -> GeminiRequest {
        let contents = request.messages.iter().map(|msg| {
            GeminiContent {
                role: match msg.role.as_str() {
                    "system" => "user".to_string(), // Gemini doesn't have system role, treat as user
                    "assistant" => "model".to_string(),
                    _ => "user".to_string(),
                },
                parts: vec![GeminiPart {
                    text: msg.content.clone(),
                }],
            }
        }).collect();

        let generation_config = GeminiGenerationConfig {
            temperature: request.temperature,
            max_output_tokens: request.max_tokens,
            ..Default::default()
        };

        GeminiRequest {
            contents,
            generation_config: Some(generation_config),
        }
    }

    fn gemini_to_openai_response(&self, gemini_response: GeminiResponse, model: &str) -> ChatResponse {
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let choices = gemini_response.candidates.into_iter().enumerate().map(|(index, candidate)| {
            let content = candidate.content.parts.into_iter()
                .map(|part| part.text)
                .collect::<Vec<_>>()
                .join("");

            ChatChoice {
                index: index as u32,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content,
                },
                finish_reason: candidate.finish_reason.map(|reason| {
                    match reason.as_str() {
                        "STOP" => "stop".to_string(),
                        "MAX_TOKENS" => "length".to_string(),
                        _ => reason.to_lowercase(),
                    }
                }),
            }
        }).collect();

        let usage = gemini_response.usage_metadata.map(|usage| Usage {
            prompt_tokens: usage.prompt_token_count,
            completion_tokens: usage.candidates_token_count,
            total_tokens: usage.total_token_count,
        });

        ChatResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4().to_string().replace("-", "")),
            object: "chat.completion".to_string(),
            created,
            model: model.to_string(),
            choices,
            usage,
        }
    }

}

#[async_trait]
impl LlmProvider for GeminiProvider {
    async fn chat_completion(&self, request: ChatRequest) -> anyhow::Result<ChatResponse> {
        let gemini_request = self.openai_to_gemini_request(&request);
        
        let url = format!("{}/models/{}:generateContent?key={}", 
                         self.base_url, request.model, self.api_key);
        
        let response = self.client
            .post(&url)
            .json(&gemini_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Gemini API error: {}", error_text));
        }

        let gemini_response = response.json::<GeminiResponse>().await?;
        let openai_response = self.gemini_to_openai_response(gemini_response, &request.model);
        
        Ok(openai_response)
    }

    async fn chat_completion_stream(&self, request: ChatRequest) -> anyhow::Result<BoxStream<'static, anyhow::Result<ChatResponseChunk>>> {
        let gemini_request = self.openai_to_gemini_request(&request);
        
        let url = format!("{}/models/{}:streamGenerateContent?key={}&alt=sse",
                         self.base_url, request.model, self.api_key);
        
        let response = self.client
            .post(&url)
            .json(&gemini_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Gemini API error: {}", error_text));
        }

        let model = request.model.clone();
        let mut buffer = String::new();
        
        let stream = response.bytes_stream().flat_map(move |item| {
            match item {
                Ok(bytes) => {
                    buffer.push_str(&String::from_utf8_lossy(&bytes));
                    let mut chunks = Vec::new();
                    
                    while let Some(line_end) = buffer.find('\n') {
                        let line = buffer.drain(..line_end + 1).collect::<String>();
                        let line = line.trim();
                        
                        if line.is_empty() || !line.starts_with("data: ") {
                            continue;
                        }
                        
                        let data = &line[6..];
                        if data == "[DONE]" {
                            continue;
                        }
                        
                        match serde_json::from_str::<GeminiStreamResponse>(data) {
                            Ok(gemini_chunk) => {
                                if let Some(candidate) = gemini_chunk.candidates.first() {
                                    let content = candidate.content.parts.iter()
                                        .map(|part| part.text.as_str())
                                        .collect::<Vec<_>>()
                                        .join("");

                                    let delta = ChatChunkDelta {
                                        role: Some("assistant".to_string()),
                                        content: if content.is_empty() { None } else { Some(content) },
                                    };

                                    let chunk = ChatResponseChunk {
                                        id: format!("chatcmpl-{}", uuid::Uuid::new_v4().to_string().replace("-", "")),
                                        object: "chat.completion.chunk".to_string(),
                                        created: std::time::SystemTime::now()
                                            .duration_since(std::time::UNIX_EPOCH)
                                            .unwrap()
                                            .as_secs(),
                                        model: model.clone(),
                                        choices: vec![ChatChunkChoice {
                                            index: 0,
                                            delta,
                                            finish_reason: candidate.finish_reason.as_ref().map(|reason| {
                                                match reason.as_str() {
                                                    "STOP" => "stop".to_string(),
                                                    "MAX_TOKENS" => "length".to_string(),
                                                    _ => reason.to_lowercase(),
                                                }
                                            }),
                                        }],
                                    };
                                    chunks.push(Ok(chunk));
                                }
                            },
                            Err(e) => chunks.push(Err(anyhow::anyhow!("Failed to parse Gemini chunk: {}", e))),
                        }
                    }
                    
                    futures_util::stream::iter(chunks)
                }
                Err(e) => futures_util::stream::iter(vec![Err(anyhow::anyhow!("Stream error: {}", e))]),
            }
        });

        Ok(Box::pin(stream))
    }

    async fn health_check(&self) -> anyhow::Result<()> {
        let url = format!("{}/models?key={}", self.base_url, self.api_key);
        
        let response = self.client
            .get(&url)
            .send()
            .await?;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(anyhow::anyhow!("Gemini health check failed: {}", response.status()))
        }
    }

    fn name(&self) -> &str {
        "gemini"
    }
}

// Gemini API request/response structures
#[derive(Debug, Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiPart {
    text: String,
}

#[derive(Debug, Serialize, Default)]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Debug, Deserialize)]
struct GeminiStreamResponse {
    candidates: Vec<GeminiCandidate>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: GeminiContent,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiUsageMetadata {
    #[serde(rename = "promptTokenCount")]
    prompt_token_count: u32,
    #[serde(rename = "candidatesTokenCount")]
    candidates_token_count: u32,
    #[serde(rename = "totalTokenCount")]
    total_token_count: u32,
}