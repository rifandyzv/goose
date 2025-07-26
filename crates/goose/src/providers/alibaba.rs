use anyhow::{Error, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};
use std::time::Duration;

use super::base::{ConfigKey, Provider, ProviderMetadata, ProviderUsage, Usage};
use super::errors::ProviderError;
use super::utils::{
    emit_debug_trace, get_model, handle_response_google_compat, handle_response_openai_compat,
    is_google_model,
};
use crate::message::Message;
use crate::model::ModelConfig;
use crate::providers::formats::openai::{create_request, get_usage, response_to_message};
use rmcp::model::Tool;
use url::Url;

pub const ALIBABA_DEFAULT_MODEL: &str = "qwen3-coder-480b-a35b-instruct";
// pub const ALIBABA_MODEL_PREFIX_ANTHROPIC: &str = "anthropic";

pub const ALIBABA_KNOWN_MODELS: &[&str] = &[
    "qwen3-coder-480b-a35b-instruct",
    "qwen3-coder-plus", // Proprietary of Qwen3 Coder
    "qwen3-235b-a22b-instruct-2507",
];
pub const ALIBABA_DOC_URL: &str = "https://www.alibabacloud.com/help/en/model-studio/what-is-model-studio";

#[derive(serde::Serialize)]
pub struct AlibabaProvider {
    #[serde(skip)]
    client: Client,
    host: String,
    api_key: String,
    model: ModelConfig,
}

impl Default for AlibabaProvider {
    fn default() -> Self {
        let model = ModelConfig::new(AlibabaProvider::metadata().default_model);
        AlibabaProvider::from_env(model).expect("Failed to initialize OpenRouter provider")
    }
}

impl AlibabaProvider {
    pub fn from_env(model: ModelConfig) -> Result<Self> {
        let config = crate::config::Config::global();
        let api_key: String = config.get_secret("MODEL_STUDIO_API_KEY")?;
        let host: String = config
            .get_param("ALIBABA_HOST")
            .unwrap_or_else(|_| "https://dashscope-intl.aliyuncs.com/compatible-mode/v1".to_string());

        let client = Client::builder()
            .timeout(Duration::from_secs(600))
            .build()?;

        Ok(Self {
            client,
            host,
            api_key,
            model,
        })
    }

    async fn post(&self, payload: &Value) -> Result<Value, ProviderError> {
        let base_url = Url::parse(&self.host)
            .map_err(|e| ProviderError::RequestFailed(format!("Invalid base URL: {e}")))?;
        let url = base_url.join("api/v1/chat/completions").map_err(|e| {
            ProviderError::RequestFailed(format!("Failed to construct endpoint URL: {e}"))
        })?;

        let response = self
            .client
            .post(url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("HTTP-Referer", "https://block.github.io/goose")
            .header("X-Title", "Goose")
            .json(payload)
            .send()
            .await?;

        // Handle Google-compatible model responses differently
        if is_google_model(payload) {
            return handle_response_google_compat(response).await;
        }

        // For OpenAI-compatible models, parse the response body to JSON
        let response_body = handle_response_openai_compat(response)
            .await
            .map_err(|e| ProviderError::RequestFailed(format!("Failed to parse response: {e}")))?;

       
        if let Some(error_obj) = response_body.get("error") {
            // If there's an error object, extract the error message and code
            let error_message = error_obj
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown Model Studio error");

            let error_code = error_obj.get("code").and_then(|c| c.as_u64()).unwrap_or(0);

            // Check for context length errors in the error message
            if error_code == 400 && error_message.contains("maximum context length") {
                return Err(ProviderError::ContextLengthExceeded(
                    error_message.to_string(),
                ));
            }

            // Return appropriate error based on the Model Studio error code
            match error_code {
                401 | 403 => return Err(ProviderError::Authentication(error_message.to_string())),
                429 => return Err(ProviderError::RateLimitExceeded(error_message.to_string())),
                500 | 503 => return Err(ProviderError::ServerError(error_message.to_string())),
                _ => return Err(ProviderError::RequestFailed(error_message.to_string())),
            }
        }

        // No error detected, return the response body
        Ok(response_body)
    }
}


fn create_request_based_on_model(
    provider: &AlibabaProvider,
    system: &str,
    messages: &[Message],
    tools: &[Tool],
) -> anyhow::Result<Value, Error> {
    let mut payload = create_request(
        &provider.model,
        system,
        messages,
        tools,
        &super::utils::ImageFormat::OpenAi,
    )?;

    if provider.supports_cache_control() {
        payload = update_request_for_anthropic(&payload);
    }

    Ok(payload)
}

#[async_trait]
impl Provider for AlibabaProvider {
    fn metadata() -> ProviderMetadata {
        ProviderMetadata::new(
            "alibaba",
            "Alibaba",
            "Router for many model providers",
            ALIBABA_DEFAULT_MODEL,
            ALIBABA_KNOWN_MODELS.to_vec(),
            ALIBABA_DOC_URL,
            vec![
                ConfigKey::new("MODEL_STUDIO_API_KEY", true, true, None),
                ConfigKey::new(
                    "ALIBABA_HOST",
                    false,
                    false,
                    Some("https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
                ),
            ],
        )
    }

    fn get_model_config(&self) -> ModelConfig {
        self.model.clone()
    }

    #[tracing::instrument(
        skip(self, system, messages, tools),
        fields(model_config, input, output, input_tokens, output_tokens, total_tokens)
    )]
    async fn complete(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<(Message, ProviderUsage), ProviderError> {
        // Create the base payload
        let payload = create_request_based_on_model(self, system, messages, tools)?;

        // Make request
        let response = self.post(&payload).await?;

        // Parse response
        let message = response_to_message(&response)?;
        let usage = response.get("usage").map(get_usage).unwrap_or_else(|| {
            tracing::debug!("Failed to get usage data");
            Usage::default()
        });
        let model = get_model(&response);
        emit_debug_trace(&self.model, &payload, &response, &usage);
        Ok((message, ProviderUsage::new(model, usage)))
    }

    /// Fetch supported models from Model Studio API (only models with tool support)
    async fn fetch_supported_models_async(&self) -> Result<Option<Vec<String>>, ProviderError> {
        let base_url = Url::parse(&self.host)
            .map_err(|e| ProviderError::RequestFailed(format!("Invalid base URL: {e}")))?;
        let url = base_url.join("api/v1/models").map_err(|e| {
            ProviderError::RequestFailed(format!("Failed to construct models URL: {e}"))
        })?;

        // Handle request failures gracefully
        // If the request fails, fall back to manual entry
        let response = match self
            .client
            .get(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("HTTP-Referer", "https://block.github.io/goose")
            .header("X-Title", "Goose")
            .send()
            .await
        {
            Ok(response) => response,
            Err(e) => {
                tracing::warn!("Failed to fetch models from Model Studio API: {}, falling back to manual model entry", e);
                return Ok(None);
            }
        };

        // Handle JSON parsing failures gracefully
        let json: serde_json::Value = match response.json().await {
            Ok(json) => json,
            Err(e) => {
                tracing::warn!("Failed to parse Model Studio API response as JSON: {}, falling back to manual model entry", e);
                return Ok(None);
            }
        };

        // Check for error in response
        if let Some(err_obj) = json.get("error") {
            let msg = err_obj
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown error");
            tracing::warn!("Model Studio API returned an error: {}", msg);
            return Ok(None);
        }

        let data = json.get("data").and_then(|v| v.as_array()).ok_or_else(|| {
            ProviderError::UsageError("Missing data field in JSON response".into())
        })?;

        let mut models: Vec<String> = data
            .iter()
            .filter_map(|model| {
                // Get the model ID
                let id = model.get("id").and_then(|v| v.as_str())?;

                // Check if the model supports tools
                let supported_params =
                    match model.get("supported_parameters").and_then(|v| v.as_array()) {
                        Some(params) => params,
                        None => {
                            // If supported_parameters is missing, skip this model (assume no tool support)
                            tracing::debug!(
                                "Model '{}' missing supported_parameters field, skipping",
                                id
                            );
                            return None;
                        }
                    };

                let has_tool_support = supported_params
                    .iter()
                    .any(|param| param.as_str() == Some("tools"));

                if has_tool_support {
                    Some(id.to_string())
                } else {
                    None
                }
            })
            .collect();

        // If no models with tool support were found, fall back to manual entry
        if models.is_empty() {
            tracing::warn!("No models with tool support found in Alibaba Cloud Model Studio API response, falling back to manual model entry");
            return Ok(None);
        }

        models.sort();
        Ok(Some(models))
    }

    fn supports_cache_control(&self) -> bool {
        self.model
            .model_name
            .starts_with(ALIBABA_MODEL_PREFIX_ANTHROPIC)
    }
}
