//! Uniform construction and listing discovery for resolved provider specs.
//!
//! Hosts own saved configuration, routing policy and access gates. This factory
//! consumes their resolved spec and obtains credentials through a callback;
//! it neither loads host configuration nor implements managed gateway policy.

use std::sync::Arc;

use crate::model::ChatModel;
use crate::model::discover::{DiscoveryRequest, pinned_openrouter_providers};
use crate::{Error, Result};

use super::anthropic::{AnthropicConfig, endpoint_is_anthropic_messages};
use super::openai::codex::OpenAiCodexRouting;
use super::openai::{AuthStyle, OpenAiConfig, endpoint_is_openrouter};
use super::{ProviderKind, ProviderSpec};

/// Resolves a credential for an already selected provider.
///
/// Implementations can consult a keyring, OAuth profile, or environment. Errors
/// are returned unchanged. Empty credentials are permitted only when the spec
/// does not require an API key.
pub trait CredentialResolver: Send + Sync {
    /// Reads the credential without making an inference request.
    ///
    /// # Errors
    /// Returns the credential source's failure when lookup cannot complete.
    fn resolve(&self, spec: &ProviderSpec) -> Result<String>;
}

impl<F> CredentialResolver for F
where
    F: Fn(&ProviderSpec) -> Result<String> + Send + Sync,
{
    fn resolve(&self, spec: &ProviderSpec) -> Result<String> {
        self(spec)
    }
}

/// Local servers that use only the OpenAI-compatible inference transport.
///
/// The resolved spec retains its provider slug, endpoint and model identity.
/// These families have no automatic model listing or native context setup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalCompatibleKind {
    /// MLX's compatible model server.
    Mlx,
    /// OMLX's compatible model server.
    Omlx,
    /// An explicitly local, otherwise unclassified compatible server.
    OpenAi,
}

/// How an Ollama context option reaches the server.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OllamaContextMode {
    /// Apply num_ctx through a native warm-up before inference, preserving the
    /// registry's default behavior. Failed native setup prevents inference.
    #[default]
    Native,
    /// Keep the compatible request options without issuing a native warm-up.
    /// Ollama may ignore num_ctx on this wire; no confirmed limit is advertised.
    CompatibleOnly,
}

/// Optional adapter settings applied after host policy and route resolution.
///
/// Header/query values can contain secrets; Debug omits their values. The registry never adds host identity or telemetry.
#[derive(Clone, Default)]
pub struct ProviderBuildOptions<'a> {
    /// Authentication override. Defaults to Anthropic for that family, no
    /// authentication for local families or empty optional credentials, and Bearer otherwise.
    pub auth_style: Option<AuthStyle>,
    /// Fixed per-route temperature.
    pub temperature_override: Option<f64>,
    /// Model glob patterns that reject temperature.
    pub temperature_unsupported_models: &'a [String],
    /// Native tool capability override; local adapters default to false.
    pub native_tool_calling: Option<bool>,
    /// Vision capability override; local adapters default to false.
    pub vision: Option<bool>,
    /// Whether to fold system messages into the first user message.
    pub merge_system_into_user: bool,
    /// Additional request headers.
    pub extra_headers: &'a [(String, String)],
    /// Additional request query parameters.
    pub extra_query_params: &'a [(String, String)],
    /// Host client identity, when required by an endpoint.
    pub user_agent: Option<&'a str>,
    /// Options merged beneath request-specific options.
    pub default_provider_options: Option<serde_json::Value>,
    /// Optional Ollama context setting. Native mode applies it before the first
    /// inference request; a failed setup prevents inference. CompatibleOnly
    /// retains the option on the compatible wire without native setup.
    pub num_ctx: Option<u32>,
    /// Explicit local-compatible family for a Compatible spec. Preserves its
    /// provider/profile/cache identity, conservative tools/vision, and disables
    /// automatic listings and Anthropic cache markers.
    pub local_compatible: Option<LocalCompatibleKind>,
    /// Ollama context setup policy. Defaults to native setup; CompatibleOnly
    /// preserves hosts that historically used the compatible wire exclusively.
    pub ollama_context_mode: OllamaContextMode,
    /// Resolved Codex routing. OAuth endpoints have no discoverable listing.
    /// The host supplies any account/originator headers and client identity.
    pub codex: Option<&'a OpenAiCodexRouting>,
    /// HTTP client carrying host TLS/proxy policy.
    pub http: Option<reqwest::Client>,
}

impl std::fmt::Debug for ProviderBuildOptions<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProviderBuildOptions")
            .field("temperature_override", &self.temperature_override)
            .field("native_tool_calling", &self.native_tool_calling)
            .field("vision", &self.vision)
            .field("local_compatible", &self.local_compatible)
            .field("ollama_context_mode", &self.ollama_context_mode)
            .field(
                "header_names",
                &self
                    .extra_headers
                    .iter()
                    .map(|(name, _)| name)
                    .collect::<Vec<_>>(),
            )
            .field(
                "query_names",
                &self
                    .extra_query_params
                    .iter()
                    .map(|(name, _)| name)
                    .collect::<Vec<_>>(),
            )
            .field("codex_oauth", &self.codex.map(|route| route.using_oauth))
            .finish_non_exhaustive()
    }
}

/// Constructs third-party and local adapters from a host-resolved spec.
///
/// ProviderKind::TinyHumans is treated as an ordinary compatible endpoint:
/// managed credentials, billing and endpoint guards belong to the host's
/// managed gateway extension, which should bypass this factory.
pub struct ProviderRegistry<R> {
    credentials: R,
}

impl<R> std::fmt::Debug for ProviderRegistry<R> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProviderRegistry")
            .finish_non_exhaustive()
    }
}

impl<R: CredentialResolver> ProviderRegistry<R> {
    /// Creates a factory using the supplied credential source.
    pub fn new(credentials: R) -> Self {
        Self { credentials }
    }

    /// Builds a model using default adapter settings.
    ///
    /// # Errors
    /// Returns validation failures for empty models/endpoints or missing required
    /// credentials, and propagates credential resolver errors.
    pub fn build(&self, spec: &ProviderSpec) -> Result<Arc<dyn ChatModel<()>>> {
        self.build_with(spec, ProviderBuildOptions::default())
    }

    /// Builds a model with host-resolved adapter settings.
    ///
    /// First-party Anthropic endpoints use Messages when native tools are not
    /// explicitly disabled. Other endpoints use Chat Completions; Codex OAuth
    /// routing selects Responses and omits max_output_tokens.
    ///
    /// # Errors
    /// Returns the same configuration and credential errors as [Self::build],
    /// and rejects local_compatible settings for non-Compatible specs.
    pub fn build_with(
        &self,
        spec: &ProviderSpec,
        options: ProviderBuildOptions<'_>,
    ) -> Result<Arc<dyn ChatModel<()>>> {
        validate(spec)?;
        if options.local_compatible.is_some() && spec.kind != ProviderKind::Compatible {
            return Err(Error::Validation(
                "local_compatible requires a Compatible provider spec".into(),
            ));
        }
        let key = self.resolve_key(spec)?;
        let endpoint = effective_endpoint(spec, &options).trim();
        let model = spec.model.trim();
        let auth = effective_auth(spec, &options, &key);
        if options.local_compatible.is_none()
            && matches!(auth, AuthStyle::Anthropic)
            && options.native_tool_calling != Some(false)
            && endpoint_is_anthropic_messages(endpoint)
        {
            return Ok(
                super::anthropic::config::build_anthropic_model_with_overrides(
                    AnthropicConfig {
                        endpoint,
                        api_key: &key,
                        model,
                        temperature_override: options.temperature_override,
                        temperature_unsupported_models: options.temperature_unsupported_models,
                        extra_headers: options.extra_headers,
                    },
                    options.http.clone(),
                    &spec.provider,
                    options.vision,
                ),
            );
        }
        let compatible_local = options.local_compatible.is_some()
            || (spec.kind == ProviderKind::Ollama
                && options.ollama_context_mode == OllamaContextMode::CompatibleOnly);
        let local = is_local(&spec.kind) || compatible_local;
        let oauth = options.codex.is_some_and(|routing| routing.using_oauth);
        let route_options = if local {
            options
                .num_ctx
                .map(|num_ctx| serde_json::json!({"options": {"num_ctx": num_ctx}}))
        } else {
            openrouter_default_provider_options(endpoint)
        };
        let provider_options =
            merge_route_options(route_options, options.default_provider_options.clone());
        let native_num_ctx = if spec.kind == ProviderKind::Ollama && !compatible_local {
            provider_options
                .as_ref()
                .and_then(|value| value.get("options"))
                .and_then(|value| value.get("num_ctx"))
                .map(|value| {
                    value
                        .as_u64()
                        .and_then(|value| u32::try_from(value).ok())
                        .filter(|value| *value > 0)
                        .ok_or_else(|| {
                            Error::Validation(
                                "Ollama num_ctx must be a positive 32-bit integer".into(),
                            )
                        })
                })
                .transpose()?
        } else {
            None
        };
        let native_options = provider_options.clone();
        let config = OpenAiConfig {
            provider_name: &spec.provider,
            endpoint,
            api_key: &key,
            auth_style: auth.clone(),
            model,
            temperature_unsupported_models: options.temperature_unsupported_models,
            temperature_override: options.temperature_override,
            merge_system_into_user: options.merge_system_into_user,
            extra_headers: options.extra_headers,
            native_tool_calling: options.native_tool_calling.or(local.then_some(false)),
            vision: options.vision.or(local.then_some(false)),
            default_provider_options: provider_options,
            responses_api_primary: oauth,
            responses_omit_max_output_tokens: oauth,
            extra_query_params: options.extra_query_params,
            user_agent: options.user_agent,
            explicit_cache_control: !local && endpoint_is_openrouter(endpoint),
        };
        if compatible_local {
            return Ok(super::openai::config::build_openai_model_with(
                config,
                false,
                options.http.clone(),
            ));
        }
        if local {
            let mut resolved = spec.clone();
            resolved.base_url = endpoint.to_owned();
            resolved.model = model.to_owned();
            let model =
                super::openai::OpenAiModel::from_spec(resolved, &key)?.with_auth_style(auth);
            let mut model =
                super::openai::config::configure_openai_model(model, config, options.http.clone());
            let num_ctx = native_num_ctx;
            if let Some(num_ctx) = num_ctx {
                model = model.with_local_num_ctx(num_ctx);
                if let Some(native_options) = native_options {
                    model = model.with_default_provider_options(native_options);
                }
            }
            return Ok(Arc::new(LocalRegistryModel {
                model,
                apply_context: num_ctx.is_some(),
                ready: tokio::sync::OnceCell::new(),
            }));
        }
        Ok(super::openai::config::build_openai_model_with(
            config,
            true,
            options.http.clone(),
        ))
    }

    /// Builds a listing request with the same endpoint, model and credentials
    /// as construction. Ollama enables native /api/show limit discovery. Other
    /// local families and Codex OAuth return None without reading credentials.
    ///
    /// Extra headers are retained; query parameters and user-agent settings
    /// describe inference transport and are not copied into listing requests.
    ///
    /// # Errors
    /// Returns configuration/credential errors for discoverable routes. Local
    /// families other than Ollama and OAuth routes do not consult credentials.
    pub fn discovery_request(
        &self,
        spec: &ProviderSpec,
        options: &ProviderBuildOptions<'_>,
    ) -> Result<Option<DiscoveryRequest>> {
        if (is_local(&spec.kind) && spec.kind != ProviderKind::Ollama)
            || options.local_compatible.is_some()
            || options.codex.is_some_and(|route| route.using_oauth)
        {
            return Ok(None);
        }
        validate(spec)?;
        let key = self.resolve_key(spec)?;
        let endpoint = effective_endpoint(spec, options).trim();
        let model = spec.model.trim();
        let mut request = DiscoveryRequest::new(endpoint, model);
        if spec.kind == ProviderKind::Ollama {
            request = request.with_ollama_native(true);
        }
        if !key.trim().is_empty() {
            request = match effective_auth(spec, options, &key) {
                AuthStyle::None => request,
                AuthStyle::Bearer => {
                    request.with_header("Authorization", format!("Bearer {}", key.trim()))
                }
                AuthStyle::XApiKey => request.with_header("x-api-key", key.trim()),
                AuthStyle::Anthropic => request
                    .with_header("x-api-key", key.trim())
                    .with_header("anthropic-version", "2023-06-01"),
                AuthStyle::Custom(name) => request.with_header(name, key.trim()),
            };
        }
        for (name, value) in options.extra_headers {
            request = with_replaced_header(request, name, value);
        }
        if let Some(provider_options) = options
            .default_provider_options
            .clone()
            .or_else(|| openrouter_default_provider_options(endpoint))
        {
            request = request.with_pinned_providers(pinned_openrouter_providers(&provider_options));
        }
        Ok(Some(request))
    }

    fn resolve_key(&self, spec: &ProviderSpec) -> Result<String> {
        let key = self.credentials.resolve(spec)?;
        if spec.requires_api_key && key.trim().is_empty() {
            return Err(Error::Validation(format!(
                "provider '{}' requires an API key",
                spec.provider
            )));
        }
        Ok(key.trim().to_string())
    }
}

fn with_replaced_header(
    mut request: DiscoveryRequest,
    name: &str,
    value: &str,
) -> DiscoveryRequest {
    request
        .headers
        .retain(|(existing, _)| !existing.eq_ignore_ascii_case(name));
    request.with_header(name, value)
}

/// Stable price ordering for direct OpenRouter routes; request-specific
/// options can override it. Other endpoints receive no default routing option.
pub fn openrouter_default_provider_options(endpoint: &str) -> Option<serde_json::Value> {
    endpoint_is_openrouter(endpoint).then(|| serde_json::json!({"provider": {"sort": "price"}}))
}

fn validate(spec: &ProviderSpec) -> Result<()> {
    if spec.model.trim().is_empty() {
        return Err(Error::Validation("provider model must not be empty".into()));
    }
    if spec.base_url.trim().is_empty() {
        return Err(Error::Validation(
            "provider endpoint must not be empty".into(),
        ));
    }
    Ok(())
}

fn is_local(kind: &ProviderKind) -> bool {
    matches!(
        kind,
        ProviderKind::Ollama | ProviderKind::LmStudio | ProviderKind::LlamaCpp | ProviderKind::Vllm
    )
}

fn effective_endpoint<'a>(
    spec: &'a ProviderSpec,
    options: &'a ProviderBuildOptions<'_>,
) -> &'a str {
    options
        .codex
        .map_or(&spec.base_url, |route| &route.endpoint)
}

// Configuring Ollama requires its native endpoint; the compatible wire ignores
// num_ctx. One successful setup is shared by all concurrent calls on this model.
struct LocalRegistryModel {
    model: super::openai::OpenAiModel,
    apply_context: bool,
    ready: tokio::sync::OnceCell<()>,
}

impl LocalRegistryModel {
    async fn prepare(&self) -> Result<()> {
        if self.apply_context {
            self.ready
                .get_or_try_init(|| async { self.model.warm_up().await })
                .await?;
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl ChatModel<()> for LocalRegistryModel {
    fn profile(&self) -> Option<&crate::model::ModelProfile> {
        <super::openai::OpenAiModel as ChatModel<()>>::profile(&self.model)
    }
    fn supports_input(
        &self,
        modality: crate::model::InputModality,
        mime: &str,
        source: crate::model::InputSource,
    ) -> bool {
        <super::openai::OpenAiModel as ChatModel<()>>::supports_input(
            &self.model,
            modality,
            mime,
            source,
        )
    }
    fn cache_identity(&self) -> Option<String> {
        <super::openai::OpenAiModel as ChatModel<()>>::cache_identity(&self.model)
    }
    fn tracks_physical_attempts(&self) -> bool {
        <super::openai::OpenAiModel as ChatModel<()>>::tracks_physical_attempts(&self.model)
    }
    async fn invoke(
        &self,
        state: &(),
        request: crate::model::ModelRequest,
    ) -> Result<crate::model::ModelResponse> {
        self.prepare().await?;
        self.model.invoke(state, request).await
    }
    async fn stream(
        &self,
        state: &(),
        request: crate::model::ModelRequest,
    ) -> Result<crate::model::ModelStream> {
        self.prepare().await?;
        self.model.stream(state, request).await
    }
    async fn fetch_deferred(
        &self,
        handle: &crate::model::DeferredHandle,
    ) -> Result<crate::model::DeferredStatus> {
        <super::openai::OpenAiModel as ChatModel<()>>::fetch_deferred(&self.model, handle).await
    }
}

// Route defaults form a base; host settings override only the fields supplied.
fn merge_route_options(
    defaults: Option<serde_json::Value>,
    overrides: Option<serde_json::Value>,
) -> Option<serde_json::Value> {
    match (defaults, overrides) {
        (Some(serde_json::Value::Object(mut base)), Some(serde_json::Value::Object(over))) => {
            for (key, value) in over {
                let previous = base.remove(&key);
                base.insert(
                    key,
                    merge_route_options(previous, Some(value)).expect("override is present"),
                );
            }
            Some(serde_json::Value::Object(base))
        }
        (base, None | Some(serde_json::Value::Null)) => base,
        (_, over) => over,
    }
}

fn effective_auth(spec: &ProviderSpec, options: &ProviderBuildOptions<'_>, key: &str) -> AuthStyle {
    options.auth_style.clone().unwrap_or_else(|| {
        if is_local(&spec.kind)
            || options.local_compatible.is_some()
            || (!spec.requires_api_key && key.is_empty())
        {
            AuthStyle::None
        } else if spec.kind == ProviderKind::Anthropic {
            AuthStyle::Anthropic
        } else {
            AuthStyle::Bearer
        }
    })
}

#[cfg(test)]
#[path = "registry_tests.rs"]
mod tests;
