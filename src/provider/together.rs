use crate::provider::{ChatRequest, ChatResponse, ChatResponseChunk, LlmProvider};
use async_trait::async_trait;
use futures_util::stream::{BoxStream, StreamExt};
use reqwest::Client;

#[derive(Debug)]
pub struct TogetherAIProvider {
    pub api_key: String,
    pub client: Client,
    pub base_url: String,
}

impl TogetherAIProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: Client::new(),
            base_url: "https://api.together.xyz/v1".to_string(),
        }
    }
}

#[async_trait]
impl LlmProvider for TogetherAIProvider {
    async fn chat_completion(&self, request: ChatRequest) -> anyhow::Result<ChatResponse> {
        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Together AI API error: {}", error_text));
        }

        let chat_response = response.json::<ChatResponse>().await?;
        Ok(chat_response)
    }

    async fn chat_completion_stream(
        &self,
        mut request: ChatRequest,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<ChatResponseChunk>>> {
        request.stream = true;
        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Together AI API error: {}", error_text));
        }

        let mut buffer = String::new();
        let stream = response.bytes_stream().flat_map(move |item| match item {
            Ok(bytes) => {
                buffer.push_str(&String::from_utf8_lossy(&bytes));
                let mut chunks = Vec::new();
                while let Some(line_end) = buffer.find('\n') {
                    let line = buffer.drain(..line_end + 1).collect::<String>();
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    if let Some(data) = line.strip_prefix("data: ") {
                        if data == "[DONE]" {
                            continue;
                        }
                        match serde_json::from_str::<ChatResponseChunk>(data) {
                            Ok(chunk) => chunks.push(Ok(chunk)),
                            Err(e) => {
                                chunks.push(Err(anyhow::anyhow!("Failed to parse chunk: {}", e)))
                            }
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
        let response = self
            .client
            .get(format!("{}/models", self.base_url))
            .bearer_auth(&self.api_key)
            .send()
            .await?;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(anyhow::anyhow!(
                "Together AI health check failed: {}",
                response.status()
            ))
        }
    }

    fn name(&self) -> &str {
        "together"
    }
}
