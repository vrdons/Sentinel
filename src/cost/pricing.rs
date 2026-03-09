use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct PricingTable {
    // model_name -> (input_per_1k_tokens, output_per_1k_tokens)
    prices: HashMap<String, (f64, f64)>,
}

impl PricingTable {
    pub fn new() -> Self {
        let mut prices = HashMap::new();

        // OpenAI Pricing (Feb 2025) - All prices per 1K tokens

        // GPT-5 Series (Latest)
        prices.insert("gpt-5".to_string(), (0.00125, 0.01));
        prices.insert("gpt-5-2025-08-07".to_string(), (0.00125, 0.01));
        prices.insert("gpt-5-chat-latest".to_string(), (0.00125, 0.01));
        prices.insert("gpt-5-mini".to_string(), (0.00025, 0.002));
        prices.insert("gpt-5-mini-2025-08-07".to_string(), (0.00025, 0.002));
        prices.insert("gpt-5-nano".to_string(), (0.00005, 0.0004));
        prices.insert("gpt-5-nano-2025-08-07".to_string(), (0.00005, 0.0004));
        prices.insert("gpt-5-pro".to_string(), (0.00125, 0.01)); // Same as GPT-5 for now
        prices.insert("gpt-5-pro-2025-10-06".to_string(), (0.00125, 0.01));
        prices.insert("gpt-5-codex".to_string(), (0.00125, 0.01));
        prices.insert("gpt-5.2".to_string(), (0.00125, 0.01));
        prices.insert("gpt-5.2-2025-12-11".to_string(), (0.00125, 0.01));
        prices.insert("gpt-5.2-pro".to_string(), (0.00125, 0.01));
        prices.insert("gpt-5.2-pro-2025-12-11".to_string(), (0.00125, 0.01));
        prices.insert("gpt-5.2-chat-latest".to_string(), (0.00125, 0.01));
        prices.insert("gpt-5.2-codex".to_string(), (0.00125, 0.01));
        prices.insert("gpt-5.1".to_string(), (0.00125, 0.01));
        prices.insert("gpt-5.1-2025-11-13".to_string(), (0.00125, 0.01));
        prices.insert("gpt-5.1-chat-latest".to_string(), (0.00125, 0.01));
        prices.insert("gpt-5.1-codex".to_string(), (0.00125, 0.01));
        prices.insert("gpt-5.1-codex-mini".to_string(), (0.00025, 0.002));
        prices.insert("gpt-5.1-codex-max".to_string(), (0.00125, 0.01));

        // GPT-4.1 Series
        prices.insert("gpt-4.1".to_string(), (0.001, 0.004)); // Batch pricing
        prices.insert("gpt-4.1-2025-04-14".to_string(), (0.001, 0.004));
        prices.insert("gpt-4.1-mini".to_string(), (0.00025, 0.002));
        prices.insert("gpt-4.1-mini-2025-04-14".to_string(), (0.00025, 0.002));
        prices.insert("gpt-4.1-nano".to_string(), (0.00005, 0.0004));
        prices.insert("gpt-4.1-nano-2025-04-14".to_string(), (0.00005, 0.0004));

        // GPT-4o Series
        prices.insert("gpt-4o".to_string(), (0.0025, 0.01));
        prices.insert("gpt-4o-2024-05-13".to_string(), (0.0025, 0.01));
        prices.insert("gpt-4o-2024-08-06".to_string(), (0.0025, 0.01));
        prices.insert("gpt-4o-2024-11-20".to_string(), (0.0025, 0.01));
        prices.insert("gpt-4o-mini".to_string(), (0.00015, 0.0006));
        prices.insert("gpt-4o-mini-2024-07-18".to_string(), (0.00015, 0.0006));
        prices.insert("gpt-4o-search-preview".to_string(), (0.0025, 0.01));
        prices.insert(
            "gpt-4o-search-preview-2025-03-11".to_string(),
            (0.0025, 0.01),
        );
        prices.insert("gpt-4o-mini-search-preview".to_string(), (0.00015, 0.0006));
        prices.insert(
            "gpt-4o-mini-search-preview-2025-03-11".to_string(),
            (0.00015, 0.0006),
        );

        // GPT-4 Legacy
        prices.insert("gpt-4".to_string(), (0.03, 0.06));
        prices.insert("gpt-4-0613".to_string(), (0.03, 0.06));
        prices.insert("gpt-4-turbo".to_string(), (0.01, 0.03));
        prices.insert("gpt-4-turbo-2024-04-09".to_string(), (0.01, 0.03));
        prices.insert("gpt-4-turbo-preview".to_string(), (0.01, 0.03));
        prices.insert("gpt-4-0125-preview".to_string(), (0.01, 0.03));
        prices.insert("gpt-4-1106-preview".to_string(), (0.01, 0.03));

        // GPT-3.5 Series
        prices.insert("gpt-3.5-turbo".to_string(), (0.0005, 0.0015));
        prices.insert("gpt-3.5-turbo-0125".to_string(), (0.0005, 0.0015));
        prices.insert("gpt-3.5-turbo-1106".to_string(), (0.001, 0.002));
        prices.insert("gpt-3.5-turbo-16k".to_string(), (0.003, 0.004));
        prices.insert("gpt-3.5-turbo-instruct".to_string(), (0.0015, 0.002));
        prices.insert("gpt-3.5-turbo-instruct-0914".to_string(), (0.0015, 0.002));

        // O-Series (Reasoning Models)
        prices.insert("o1".to_string(), (0.015, 0.06));
        prices.insert("o1-2024-12-17".to_string(), (0.015, 0.06));
        prices.insert("o1-mini".to_string(), (0.003, 0.012));
        prices.insert("o1-preview".to_string(), (0.015, 0.06));
        prices.insert("o1-pro".to_string(), (0.015, 0.06));
        prices.insert("o1-pro-2025-03-19".to_string(), (0.015, 0.06));

        // O3 Series
        prices.insert("o3".to_string(), (0.002, 0.008));
        prices.insert("o3-2025-04-16".to_string(), (0.002, 0.008));
        prices.insert("o3-mini".to_string(), (0.00055, 0.0022)); // Batch pricing
        prices.insert("o3-mini-2025-01-31".to_string(), (0.00055, 0.0022));

        // O4 Series
        prices.insert("o4-mini".to_string(), (0.0011, 0.0044));
        prices.insert("o4-mini-2025-04-16".to_string(), (0.0011, 0.0044));
        prices.insert("o4-mini-deep-research".to_string(), (0.01, 0.04)); // Deep research variant
        prices.insert("o4-mini-deep-research-2025-06-26".to_string(), (0.01, 0.04));

        // Audio/Realtime Models
        prices.insert("gpt-audio".to_string(), (0.00015, 0.0006)); // Same as mini
        prices.insert("gpt-audio-2025-08-28".to_string(), (0.00015, 0.0006));
        prices.insert("gpt-audio-mini".to_string(), (0.00015, 0.0006));
        prices.insert("gpt-audio-mini-2025-10-06".to_string(), (0.00015, 0.0006));
        prices.insert("gpt-audio-mini-2025-12-15".to_string(), (0.00015, 0.0006));
        prices.insert("gpt-realtime".to_string(), (0.00015, 0.0006));
        prices.insert("gpt-realtime-2025-08-28".to_string(), (0.00015, 0.0006));
        prices.insert("gpt-realtime-mini".to_string(), (0.00015, 0.0006));
        prices.insert(
            "gpt-realtime-mini-2025-10-06".to_string(),
            (0.00015, 0.0006),
        );
        prices.insert(
            "gpt-realtime-mini-2025-12-15".to_string(),
            (0.00015, 0.0006),
        );
        prices.insert("gpt-4o-realtime-preview".to_string(), (0.005, 0.02));
        prices.insert(
            "gpt-4o-realtime-preview-2024-12-17".to_string(),
            (0.005, 0.02),
        );
        prices.insert(
            "gpt-4o-realtime-preview-2025-06-03".to_string(),
            (0.005, 0.02),
        );
        prices.insert(
            "gpt-4o-mini-realtime-preview".to_string(),
            (0.00015, 0.0006),
        );
        prices.insert(
            "gpt-4o-mini-realtime-preview-2024-12-17".to_string(),
            (0.00015, 0.0006),
        );
        prices.insert("gpt-4o-audio-preview".to_string(), (0.005, 0.02));
        prices.insert("gpt-4o-audio-preview-2024-12-17".to_string(), (0.005, 0.02));
        prices.insert("gpt-4o-audio-preview-2025-06-03".to_string(), (0.005, 0.02));
        prices.insert("gpt-4o-mini-audio-preview".to_string(), (0.00015, 0.0006));
        prices.insert(
            "gpt-4o-mini-audio-preview-2024-12-17".to_string(),
            (0.00015, 0.0006),
        );

        // Image Generation Models
        prices.insert("gpt-image-1".to_string(), (0.04, 0.04)); // Per image, not per token
        prices.insert("gpt-image-1-mini".to_string(), (0.02, 0.02));
        prices.insert("gpt-image-1.5".to_string(), (0.04, 0.04));
        prices.insert("chatgpt-image-latest".to_string(), (0.04, 0.04));

        // TTS and Transcription Models
        prices.insert("gpt-4o-transcribe".to_string(), (0.006, 0.006)); // Per minute
        prices.insert("gpt-4o-mini-transcribe".to_string(), (0.006, 0.006));
        prices.insert("gpt-4o-transcribe-diarize".to_string(), (0.006, 0.006));
        prices.insert(
            "gpt-4o-mini-transcribe-2025-12-15".to_string(),
            (0.006, 0.006),
        );
        prices.insert(
            "gpt-4o-mini-transcribe-2025-03-20".to_string(),
            (0.006, 0.006),
        );
        prices.insert("gpt-4o-mini-tts".to_string(), (0.015, 0.015)); // Per 1K characters
        prices.insert("gpt-4o-mini-tts-2025-12-15".to_string(), (0.015, 0.015));
        prices.insert("gpt-4o-mini-tts-2025-03-20".to_string(), (0.015, 0.015));

        // Legacy/Base Models
        prices.insert("davinci-002".to_string(), (0.002, 0.002));
        prices.insert("babbage-002".to_string(), (0.0004, 0.0004));

        // Whisper (Audio transcription) - $0.006 per minute
        prices.insert("whisper-1".to_string(), (0.006, 0.006)); // Special pricing per minute

        // TTS (Text-to-Speech) - $15 per 1M characters
        prices.insert("tts-1".to_string(), (0.015, 0.015)); // Per 1K characters
        prices.insert("tts-1-hd".to_string(), (0.015, 0.015));
        prices.insert("tts-1-1106".to_string(), (0.015, 0.015));
        prices.insert("tts-1-hd-1106".to_string(), (0.015, 0.015));

        // DALL-E (Image generation) - Per image pricing
        prices.insert("dall-e-2".to_string(), (0.02, 0.02)); // $0.016-0.020 per image
        prices.insert("dall-e-3".to_string(), (0.08, 0.08)); // $0.040-0.120 per image

        // Embedding Models - $0.13-3.00 per 1M tokens
        prices.insert("text-embedding-ada-002".to_string(), (0.0001, 0.0001));
        prices.insert("text-embedding-3-small".to_string(), (0.00002, 0.00002));
        prices.insert("text-embedding-3-large".to_string(), (0.00013, 0.00013));

        // Moderation Models - Free
        prices.insert("omni-moderation-latest".to_string(), (0.0, 0.0));
        prices.insert("omni-moderation-2024-09-26".to_string(), (0.0, 0.0));

        // Video Generation Models (Sora)
        prices.insert("sora-2".to_string(), (0.04, 0.04)); // Estimated pricing
        prices.insert("sora-2-pro".to_string(), (0.08, 0.08));

        // Anthropic Pricing (Feb 2026) - All Claude Series

        // Claude 4.6 Series (Latest)
        prices.insert("claude-opus-4.6".to_string(), (0.005, 0.025));
        prices.insert("claude-sonnet-4.6".to_string(), (0.003, 0.015));

        // Claude 4.5 Series
        prices.insert("claude-opus-4.5".to_string(), (0.005, 0.025));
        prices.insert("claude-sonnet-4.5".to_string(), (0.003, 0.015));
        prices.insert("claude-haiku-4.5".to_string(), (0.001, 0.005));

        // Claude 4 Series
        prices.insert("claude-opus-4".to_string(), (0.015, 0.075));
        prices.insert("claude-opus-4.1".to_string(), (0.015, 0.075));
        prices.insert("claude-sonnet-4".to_string(), (0.003, 0.015));
        prices.insert("claude-sonnet-3.7".to_string(), (0.003, 0.015)); // Deprecated

        // Claude 3.5 Series
        prices.insert("claude-3-5-sonnet-20241022".to_string(), (0.003, 0.015));
        prices.insert("claude-3-5-haiku-20241022".to_string(), (0.001, 0.005));
        prices.insert("claude-haiku-3.5".to_string(), (0.0008, 0.004));

        // Claude 3 Series (Legacy)
        prices.insert("claude-3-opus-20240229".to_string(), (0.015, 0.075));
        prices.insert("claude-opus-3".to_string(), (0.015, 0.075)); // Deprecated
        prices.insert("claude-3-sonnet-20240229".to_string(), (0.003, 0.015));
        prices.insert("claude-3-haiku-20240307".to_string(), (0.00025, 0.00125));
        prices.insert("claude-haiku-3".to_string(), (0.00025, 0.00125));

        // Alternative naming patterns
        prices.insert("claude-opus-46".to_string(), (0.005, 0.025));
        prices.insert("claude-sonnet-46".to_string(), (0.003, 0.015));
        prices.insert("claude-opus-45".to_string(), (0.005, 0.025));
        prices.insert("claude-sonnet-45".to_string(), (0.003, 0.015));
        prices.insert("claude-haiku-45".to_string(), (0.001, 0.005));

        // Google Gemini Pricing (Feb 2025)
        prices.insert("gemini-2.0-flash-exp".to_string(), (0.00075, 0.003));
        prices.insert("gemini-1.5-pro".to_string(), (0.00125, 0.005));
        prices.insert("gemini-1.5-flash".to_string(), (0.00075, 0.003));

        // Mistral Pricing
        prices.insert("mistral-large-latest".to_string(), (0.002, 0.006));
        prices.insert("mistral-small-latest".to_string(), (0.0001, 0.0003));

        // Cohere Pricing
        prices.insert("command-r-plus".to_string(), (0.003, 0.015));
        prices.insert("command-r".to_string(), (0.0005, 0.0015));

        // Perplexity Pricing
        prices.insert("pplx-70b-online".to_string(), (0.001, 0.001));

        // Together AI / Open Source models
        prices.insert("llama3-70b-8192".to_string(), (0.0009, 0.0009));

        // Ollama (Local - Free)
        prices.insert("llama3:70b".to_string(), (0.0, 0.0));
        prices.insert("mixtral:8x7b".to_string(), (0.0, 0.0));

        Self { prices }
    }

    pub fn calculate_cost(&self, model: &str, input_tokens: u32, output_tokens: u32) -> f64 {
        if let Some((input_price, output_price)) = self.prices.get(model) {
            let input_cost = (input_tokens as f64 / 1000.0) * input_price;
            let output_cost = (output_tokens as f64 / 1000.0) * output_price;
            input_cost + output_cost
        } else {
            0.0 // Unknown model
        }
    }
}

impl Default for PricingTable {
    fn default() -> Self {
        Self::new()
    }
}
