//! Provider construction and custom-endpoint validation.

use crate::{
    CohereEmbeddingModel, EmbeddingModel, Error, NoopEmbeddingModel, OllamaEmbeddingModel,
    OpenAiEmbeddingModel, Result, VOYAGE_API_BASE, VoyageEmbeddingModel, model_supports_dimensions,
};

/// Build an embedding model for a named provider.
///
/// Managed products inject their own authenticated cloud model; all standard,
/// local, and OpenAI-compatible providers are constructed here.
pub fn create_embedding_model(
    provider: &str,
    model: &str,
    dimensions: usize,
    api_key: &str,
    custom_endpoint: Option<&str>,
    ollama_base_url: &str,
) -> Result<Box<dyn EmbeddingModel>> {
    let model: Box<dyn EmbeddingModel> = match provider {
        "voyage" => Box::new(VoyageEmbeddingModel::with_options(
            api_key,
            model,
            dimensions,
            VOYAGE_API_BASE,
        )),
        "ollama" => Box::new(OllamaEmbeddingModel::try_new(
            ollama_base_url,
            model,
            dimensions,
        )?),
        "openai" => Box::new(openai_embedding_model(
            "https://api.openai.com",
            api_key,
            model,
            dimensions,
            true,
        )),
        "cohere" => Box::new(
            CohereEmbeddingModel::new(api_key)
                .with_model(model)
                .with_dimensions(dimensions),
        ),
        "custom" => {
            let endpoint =
                validate_custom_endpoint(custom_endpoint.unwrap_or_default(), !api_key.is_empty())?;
            Box::new(openai_embedding_model(
                &endpoint, api_key, model, dimensions, false,
            ))
        }
        name if name.starts_with("custom:") => {
            let endpoint = custom_endpoint
                .or_else(|| name.strip_prefix("custom:"))
                .unwrap_or_default();
            let endpoint = validate_custom_endpoint(endpoint, !api_key.is_empty())?;
            Box::new(openai_embedding_model(
                &endpoint, api_key, model, dimensions, false,
            ))
        }
        "none" => Box::new(NoopEmbeddingModel),
        "cloud" | "managed" => {
            return Err(Error::Validation(
                "managed embeddings require a host-provided bearer resolver".into(),
            ));
        }
        unknown => {
            return Err(Error::Validation(format!(
                "unknown embedding provider: \"{unknown}\". Supported: \"voyage\", \
                 \"openai\", \"cohere\", \"ollama\", \"custom\", \"none\""
            )));
        }
    };
    Ok(model)
}

/// Build an OpenAI-compatible model with consistent dimension semantics.
pub fn openai_embedding_model(
    base_url: &str,
    api_key: &str,
    model: &str,
    dimensions: usize,
    required_api_key: bool,
) -> OpenAiEmbeddingModel {
    OpenAiEmbeddingModel::new(api_key)
        .with_base_url(base_url)
        .with_model(model)
        .with_dimensions(dimensions)
        .with_send_dimensions(model_supports_dimensions(model))
        .with_required_api_key(required_api_key)
}

/// Validate an OpenAI-compatible endpoint before credentials may be sent.
pub fn validate_custom_endpoint(endpoint: &str, has_credentials: bool) -> Result<String> {
    let endpoint = endpoint.trim().trim_end_matches('/');
    if endpoint.is_empty() {
        return Err(Error::Validation(
            "custom embedding provider endpoint must not be empty".into(),
        ));
    }

    let parsed = reqwest::Url::parse(endpoint)
        .map_err(|_| Error::Validation("custom embedding provider endpoint is invalid".into()))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(Error::Validation(
            "custom embedding provider endpoint must use HTTP or HTTPS".into(),
        ));
    }

    if has_credentials {
        let loopback = parsed
            .host()
            .map(|host| match host {
                url::Host::Domain(domain) => domain.eq_ignore_ascii_case("localhost"),
                url::Host::Ipv4(address) => address.is_loopback(),
                url::Host::Ipv6(address) => address.is_loopback(),
            })
            .unwrap_or(false);
        if parsed.scheme() != "https" && !(parsed.scheme() == "http" && loopback) {
            return Err(Error::Validation(
                "credentialed custom embedding provider endpoints must use HTTPS or loopback HTTP"
                    .into(),
            ));
        }
    }

    Ok(endpoint.to_owned())
}
