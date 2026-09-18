//! Harness embeddings + retrieval module.
//!
//! In the recursive (RLM-style) architecture this module is how a model reaches
//! *outside* its context window: instead of stuffing a whole corpus into one
//! prompt, an agent (or a sub-agent / REPL step) embeds documents once and then
//! recursively retrieves only the snippets relevant to the current sub-question,
//! mitigating context rot the same way the Recursive Language Models work treats
//! a long prompt as an external, searchable environment.
//!
//! Embeddings are provider-neutral dense vector representations used for
//! retrieval, semantic search, and retrieval-augmented prompt context. This
//! module owns:
//!
//! - [`EmbeddingModel`] — the trait every embedding provider implements.
//! - [`MockEmbeddingModel`] — a deterministic, offline implementation for tests.
//! - [`VectorStore`] — a trait for storing and searching dense vectors.
//! - [`InMemoryVectorStore`] — an in-process cosine-similarity vector store.
//! - [`Retriever`] — ties an embedding model and vector store together for
//!   indexing documents and answering text queries.
//! - [`cosine_similarity`] — the distance metric used by the in-memory store.
//!
//! [`OpenAiEmbeddingModel`] adds a hosted provider backed by the OpenAI
//! embeddings endpoint (always compiled).
//!
//! The design mirrors LangChain's separation of concerns: chat models generate
//! messages, embedding models generate vectors, vector stores search vectors,
//! and retrievers return documents. Prompt and middleware code decides what
//! retrieved context enters a model request.
//!
//! # Example
//! ```
//! use std::sync::Arc;
//! use tinyinference_embeddings::{InMemoryVectorStore, MockEmbeddingModel, Retriever};
//! use serde_json::json;
//!
//! # tokio::runtime::Runtime::new().unwrap().block_on(async {
//! let retriever = Retriever::new(
//!     Arc::new(MockEmbeddingModel::new(64)),
//!     Arc::new(InMemoryVectorStore::new()),
//! );
//! retriever
//!     .index(vec![
//!         ("cats".into(), "cats are great pets".into(), json!({"topic": "animals"})),
//!         ("finance".into(), "the stock market crashed".into(), json!({"topic": "finance"})),
//!     ])
//!     .await
//!     .unwrap();
//!
//! let hits = retriever.retrieve("cats are great pets", 1).await.unwrap();
//! assert_eq!(hits[0].id, "cats");
//! # });
//! ```

pub mod catalog;
mod error;
pub mod factory;
pub mod probe;
pub mod served_models;
mod types;

pub use error::{Error, Result};
pub use types::*;

use async_trait::async_trait;
use serde_json::Value;

// ── Vector math ───────────────────────────────────────────────────────────────

/// Computes the cosine similarity between two vectors.
///
/// Cosine similarity is the dot product of the vectors divided by the product
/// of their Euclidean norms, yielding a value in `[-1.0, 1.0]` where `1.0`
/// means identical direction, `0.0` means orthogonal, and `-1.0` means
/// opposite direction. It is invariant to vector magnitude.
///
/// Returns `0.0` when the vectors have different lengths or when either vector
/// has zero magnitude, so callers never observe `NaN` from a degenerate input.
///
/// # Example
/// ```
/// use tinyinference_embeddings::cosine_similarity;
///
/// assert_eq!(cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]), 1.0);
/// assert_eq!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
/// assert_eq!(cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]), -1.0);
/// ```
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

/// Computes cosine similarity with `f64` accumulation for long vectors.
///
/// Returns zero for empty, mismatched, or zero-magnitude vectors and clamps
/// floating-point drift to the mathematical `[-1.0, 1.0]` range.
pub fn cosine_similarity_f64(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let x = f64::from(*x);
        let y = f64::from(*y);
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom <= f64::EPSILON {
        return 0.0;
    }
    (dot / denom).clamp(-1.0, 1.0)
}

/// Fold a vector into an existing running centroid.
///
/// A missing or dimensionally incompatible centroid is replaced with the new
/// vector, keeping vectors from different embedding spaces out of one mean.
#[must_use]
pub fn incremental_mean_embedding(
    current_centroid: &[f32],
    new_embedding: &[f32],
    count: usize,
) -> Vec<f32> {
    if current_centroid.is_empty() || current_centroid.len() != new_embedding.len() {
        return new_embedding.to_vec();
    }
    current_centroid
        .iter()
        .zip(new_embedding.iter())
        .map(|(current, new)| current + (new - current) / (count as f32 + 1.0))
        .collect()
}

/// Estimate embedding input tokens from Unicode scalar count.
///
/// This intentionally uses the common four-characters-per-token heuristic;
/// callers requiring billable usage should use provider-reported
/// [`EmbeddingUsage`] instead.
#[must_use]
pub fn estimate_embedding_input_tokens(texts: &[String]) -> u64 {
    let characters = texts.iter().map(|text| text.chars().count()).sum::<usize>();
    (characters as u64).div_ceil(4)
}

/// Candidate for maximal marginal relevance selection.
#[derive(Clone, Copy, Debug)]
pub struct MmrCandidate<'a> {
    /// Caller-defined index returned with the result.
    pub index: usize,
    /// Candidate vector.
    pub embedding: &'a [f32],
    /// Precomputed relevance to the query.
    pub relevance: f64,
}

/// One maximal marginal relevance selection result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MmrResult {
    /// Caller-defined candidate index.
    pub index: usize,
    /// MMR score at the selection step.
    pub score: f64,
}

/// Select candidates using maximal marginal relevance.
///
/// `lambda` is clamped to `[0.0, 1.0]`; higher values favor relevance and
/// lower values favor diversity.
#[must_use]
pub fn mmr_select(candidates: &[MmrCandidate<'_>], limit: usize, lambda: f64) -> Vec<MmrResult> {
    if candidates.is_empty() || limit == 0 {
        return Vec::new();
    }

    let lambda = lambda.clamp(0.0, 1.0);
    let limit = limit.min(candidates.len());
    let mut selected_embeddings: Vec<&[f32]> = Vec::with_capacity(limit);
    let mut results = Vec::with_capacity(limit);
    let mut available = vec![true; candidates.len()];

    for _ in 0..limit {
        let mut best_idx = None;
        let mut best_mmr = f64::NEG_INFINITY;
        for (index, candidate) in candidates.iter().enumerate() {
            if !available[index] {
                continue;
            }
            let max_similarity = if selected_embeddings.is_empty() {
                0.0
            } else {
                selected_embeddings
                    .iter()
                    .map(|selected| cosine_similarity_f64(candidate.embedding, selected))
                    .fold(f64::NEG_INFINITY, f64::max)
            };
            let score = lambda * candidate.relevance - (1.0 - lambda) * max_similarity;
            if score > best_mmr {
                best_mmr = score;
                best_idx = Some(index);
            }
        }
        let Some(index) = best_idx else { break };
        available[index] = false;
        selected_embeddings.push(candidates[index].embedding);
        results.push(MmrResult {
            index: candidates[index].index,
            score: best_mmr,
        });
    }
    results
}

// ── InMemoryVectorStore ───────────────────────────────────────────────────────

impl InMemoryVectorStore {
    /// Creates a new, empty in-memory vector store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of vectors currently stored.
    pub fn len(&self) -> usize {
        self.inner
            .lock()
            .map(|inner| inner.entries.len())
            .unwrap_or(0)
    }

    /// Returns `true` when the store holds no vectors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn store_lock_err(e: impl std::fmt::Display) -> crate::Error {
    crate::Error::Embedding(format!("vector store lock poisoned: {e}"))
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn add(&self, id: String, vector: Vec<f32>, metadata: Value) -> Result<()> {
        if vector.is_empty() {
            return Err(crate::Error::Validation(
                "cannot add a zero-dimensional vector to the vector store".to_string(),
            ));
        }
        let mut inner = self.inner.lock().map_err(store_lock_err)?;
        // The store's dimensionality is fixed by its first vector; every later
        // insert must match so queries always compare like with like (a
        // mismatched stored vector would silently score 0.0 forever).
        if let Some(first) = inner.entries.first()
            && first.vector.len() != vector.len()
        {
            return Err(crate::Error::Validation(format!(
                "vector for id `{id}` has {} dimensions but the store holds {}-dimensional vectors",
                vector.len(),
                first.vector.len()
            )));
        }
        // Replace an existing entry with the same id so re-indexing updates it.
        // The id → index map makes the upsert O(1) instead of a linear scan.
        match inner.index.get(&id) {
            Some(&at) => {
                let existing = &mut inner.entries[at];
                existing.vector = vector;
                existing.metadata = metadata;
            }
            None => {
                let at = inner.entries.len();
                inner.index.insert(id.clone(), at);
                inner.entries.push(VectorEntry {
                    id,
                    vector,
                    metadata,
                });
            }
        }
        Ok(())
    }

    async fn query(&self, vector: &[f32], top_k: usize) -> Result<Vec<ScoredDoc>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let inner = self.inner.lock().map_err(store_lock_err)?;
        // An empty store has no dimensionality to validate against; it answers
        // every query with no hits.
        let Some(first) = inner.entries.first() else {
            return Ok(Vec::new());
        };
        // A wrong-length query vector would cosine-score 0.0 against every
        // entry and return arbitrary "matches"; fail loudly instead.
        if vector.len() != first.vector.len() {
            return Err(crate::Error::Validation(format!(
                "query vector has {} dimensions but the store holds {}-dimensional vectors",
                vector.len(),
                first.vector.len()
            )));
        }
        // Score first (index + score only), pick the top-k, and clone ids and
        // metadata only for the winners — entries outside the top-k are never
        // cloned.
        let mut scored: Vec<(usize, f32)> = inner
            .entries
            .iter()
            .enumerate()
            .map(|(at, e)| (at, cosine_similarity(vector, &e.vector)))
            .collect();
        // Sort by descending score; `total_cmp` keeps ordering total even with
        // NaN-free f32 scores (cosine_similarity never returns NaN). The sort
        // is stable, so equal scores keep insertion order, exactly as before.
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(top_k);
        Ok(scored
            .into_iter()
            .map(|(at, score)| {
                let entry = &inner.entries[at];
                ScoredDoc {
                    id: entry.id.clone(),
                    score,
                    metadata: entry.metadata.clone(),
                }
            })
            .collect())
    }
}

// ── Retriever ─────────────────────────────────────────────────────────────────

impl Retriever {
    /// Creates a retriever from an embedding model and a vector store.
    pub fn new(
        model: std::sync::Arc<dyn EmbeddingModel>,
        store: std::sync::Arc<dyn VectorStore>,
    ) -> Self {
        Self { model, store }
    }

    /// Returns the embedding model backing this retriever.
    pub fn model(&self) -> &std::sync::Arc<dyn EmbeddingModel> {
        &self.model
    }

    /// Returns the vector store backing this retriever.
    pub fn store(&self) -> &std::sync::Arc<dyn VectorStore> {
        &self.store
    }

    /// Embeds and indexes a batch of documents.
    ///
    /// Each document is a `(id, text, metadata)` tuple: `text` is embedded with
    /// the configured model and stored under `id` along with `metadata`. The
    /// texts are embedded in a single batched [`EmbeddingModel::embed`] call.
    ///
    /// Re-indexing a document with an existing `id` replaces its vector and
    /// metadata in stores that support in-place update (such as
    /// [`InMemoryVectorStore`]).
    pub async fn index(&self, docs: Vec<(String, String, Value)>) -> Result<()> {
        if docs.is_empty() || !self.model.is_enabled() {
            return Ok(());
        }
        let texts: Vec<String> = docs.iter().map(|(_, text, _)| text.clone()).collect();
        let vectors = self.model.embed(&texts).await?;
        if vectors.len() != docs.len() {
            return Err(crate::Error::Embedding(format!(
                "embedding model returned {} vectors for {} documents",
                vectors.len(),
                docs.len()
            )));
        }
        for ((id, _text, metadata), vector) in docs.into_iter().zip(vectors) {
            self.store.add(id, vector, metadata).await?;
        }
        Ok(())
    }

    /// Embeds `query` and returns up to `top_k` most-similar documents.
    ///
    /// Results are sorted by descending cosine similarity (most similar first).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](crate::Error::Validation)
    /// (from the backing store) when the query embedding's dimensionality does
    /// not match the indexed vectors — for example when the store was indexed
    /// with a different embedding model. An empty store never errors: it
    /// answers every query with no hits.
    pub async fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<ScoredDoc>> {
        if !self.model.is_enabled() {
            return Ok(Vec::new());
        }
        let query_vector = self.model.embed_query(query).await?;
        self.store.query(&query_vector, top_k).await
    }
}

mod cloud;
mod cohere;
mod noop;
mod ollama;
mod openai;
mod rate_limit;
mod voyage;

pub use noop::NoopEmbeddingModel;
pub use ollama::{
    DEFAULT_OLLAMA_DIMENSIONS, DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_URL, OllamaEmbeddingModel,
    RECOMMENDED_OLLAMA_CONTEXT_TOKENS, known_ollama_embedding_dimensions,
};
pub use openai::{MODELS_SUPPORTING_DIMENSIONS, OpenAiEmbeddingModel, model_supports_dimensions};
pub use rate_limit::{DEFAULT_REQUESTS_PER_MINUTE, acquire, rate_limit, set_rate_limit};
pub use tinyinference_core::{
    BASE_BACKOFF_MS, MAX_BACKOFF_MS, MAX_RETRIES, backoff_ms_for_attempt, parse_retry_after_ms,
};
pub use types::format_embedding_signature;
pub use voyage::{
    VOYAGE_API_BASE, VOYAGE_DEFAULT_DIMENSIONS, VOYAGE_DEFAULT_MODEL, VoyageEmbeddingModel,
};

#[cfg(test)]
mod test;
pub use cloud::{
    BearerResolver, CloudEmbeddingModel, DEFAULT_CLOUD_DIMENSIONS, DEFAULT_CLOUD_MODEL,
};
pub use cohere::{
    COHERE_API_BASE, COHERE_DEFAULT_DIMENSIONS, COHERE_DEFAULT_MODEL, CohereEmbeddingModel,
};
