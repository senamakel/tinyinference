//! Generated media artifacts and their persistence through the blob port.

use bytes::Bytes;
use tinystoragedrivers_api::BlobStore;
use tinystoragedrivers_types::BlobMeta;

use crate::reference::extension_for_media_type;
use crate::{Error, Result};

/// One generated artifact, held in memory.
///
/// Providers return media either inline (OpenRouter images arrive as base64)
/// or behind an authenticated content endpoint (OpenRouter videos); generators
/// always download before returning, so a `GeneratedMedia` is self-contained
/// and never carries a URL that expires or needs a credential.
#[derive(Clone, PartialEq, Eq)]
pub struct GeneratedMedia {
    /// MIME type, for example `image/png` or `video/mp4`.
    pub media_type: String,
    /// The artifact bytes.
    pub data: Bytes,
}

impl GeneratedMedia {
    /// Creates an artifact from its media type and bytes.
    #[must_use]
    pub fn new(media_type: impl Into<String>, data: impl Into<Bytes>) -> Self {
        Self {
            media_type: media_type.into(),
            data: data.into(),
        }
    }

    /// File extension for this artifact's media type, or `fallback`.
    #[must_use]
    pub fn extension<'a>(&self, fallback: &'a str) -> &'a str {
        extension_for_media_type(&self.media_type, fallback)
    }

    /// Stores the artifact in `blobs` under `<prefix>/<stem>.<ext>` (or
    /// `<stem>.<ext>` for an empty `prefix`) and returns what was written.
    /// `fallback_extension` is used when the media type is unknown.
    ///
    /// The stem and the extension are sanitized to `[A-Za-z0-9_-]` so a
    /// provider id can never add a path segment (`..`) to the key. `prefix`
    /// is the caller's own namespace and is validated by the store.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an empty artifact, [`Error::Storage`] when the
    /// store refuses the key or the write.
    pub async fn persist(
        &self,
        blobs: &dyn BlobStore,
        prefix: &str,
        stem: &str,
        fallback_extension: &str,
    ) -> Result<BlobMeta> {
        if self.data.is_empty() {
            return Err(Error::Validation(
                "refusing to persist an empty artifact".into(),
            ));
        }
        let stem = sanitize_component(stem, "media");
        let safe_fallback = sanitize_component(fallback_extension, "bin");
        let extension = extension_for_media_type(&self.media_type, &safe_fallback);
        let prefix = prefix.trim_matches('/');
        let key = if prefix.is_empty() {
            format!("{stem}.{extension}")
        } else {
            format!("{prefix}/{stem}.{extension}")
        };
        let meta = blobs
            .put(&key, self.data.to_vec(), Some(&self.media_type))
            .await?;
        tracing::debug!(
            key = %meta.key,
            bytes = self.data.len(),
            media_type = %self.media_type,
            "[tinyinference-image] persisted generated media"
        );
        Ok(meta)
    }
}

fn sanitize_component(raw: &str, empty: &str) -> String {
    let cleaned: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if cleaned.is_empty() {
        empty.to_owned()
    } else {
        cleaned
    }
}

impl std::fmt::Debug for GeneratedMedia {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GeneratedMedia")
            .field("media_type", &self.media_type)
            .field("len", &self.data.len())
            .finish()
    }
}
