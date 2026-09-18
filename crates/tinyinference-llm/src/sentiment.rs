//! Provider-neutral sentiment response parsing.

/// Result of sentiment / emotion analysis on a user message.
#[derive(Debug, serde::Serialize)]
pub struct SentimentResult {
    /// Primary emotion label.
    /// One of: joy, sadness, anger, surprise, fear, disgust, neutral.
    pub emotion: String,
    /// Overall valence: positive, negative, or neutral.
    pub valence: String,
    /// Model's self-reported confidence (0.0–1.0).
    pub confidence: f32,
}

impl SentimentResult {
    /// Safe default when analysis is skipped or parsing fails.
    pub fn neutral() -> Self {
        Self {
            emotion: "neutral".to_string(),
            valence: "neutral".to_string(),
            confidence: 1.0,
        }
    }
}

/// Known emotion labels the model is expected to produce.
const VALID_EMOTIONS: &[&str] = &[
    "joy", "sadness", "anger", "surprise", "fear", "disgust", "neutral",
];

/// Known valence labels.
const VALID_VALENCES: &[&str] = &["positive", "negative", "neutral"];

/// Parse the model's 3-word response into a `SentimentResult`.
/// Falls back to neutral on any parsing error.
pub fn parse_sentiment_response(text: &str) -> SentimentResult {
    let parts: Vec<&str> = text.split_whitespace().collect();
    if parts.len() != 3 {
        tracing::debug!(
            parts = parts.len(),
            "[local_ai:sentiment] unexpected token count, falling back to neutral"
        );
        return SentimentResult::neutral();
    }

    let emotion = parts[0].to_string();
    let valence = parts[1].to_string();
    let confidence: f32 = parts[2].parse().unwrap_or(0.5);

    // Validate labels, fall back to neutral for garbage
    let emotion = if VALID_EMOTIONS.contains(&emotion.as_str()) {
        emotion
    } else {
        tracing::debug!(raw = %emotion, "[local_ai:sentiment] unknown emotion label, defaulting to neutral");
        "neutral".to_string()
    };

    let valence = if VALID_VALENCES.contains(&valence.as_str()) {
        valence
    } else {
        tracing::debug!(raw = %valence, "[local_ai:sentiment] unknown valence label, defaulting to neutral");
        "neutral".to_string()
    };

    let confidence = if confidence.is_finite() {
        confidence.clamp(0.0, 1.0)
    } else {
        0.5
    };

    SentimentResult {
        emotion,
        valence,
        confidence,
    }
}

#[cfg(test)]
#[path = "sentiment_test.rs"]
mod tests;
