//! Types for Cactus API

use serde::{Deserialize, Serialize};

/// Chat message role
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System message (instructions)
    System,
    /// User message
    User,
    /// Assistant response
    Assistant,
    /// Tool/function result
    Tool,
}

/// A chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role of the message sender
    pub role: Role,
    /// Message content
    pub content: String,
    /// Optional image paths for multi-modal input
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
}

impl Message {
    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
            images: None,
        }
    }

    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            images: None,
        }
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            images: None,
        }
    }

    /// Add images to the message
    pub fn with_images(mut self, images: Vec<String>) -> Self {
        self.images = Some(images);
        self
    }
}

/// Options for text generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateOptions {
    /// Sampling temperature (0.0 - 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Top-p (nucleus) sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Top-k sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,

    /// Frequency penalty
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    /// Presence penalty
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    /// Random seed for reproducibility
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(40),
            max_tokens: Some(2048),
            stop: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
        }
    }
}

impl GenerateOptions {
    /// Create options optimized for creative generation
    pub fn creative() -> Self {
        Self {
            temperature: Some(1.0),
            top_p: Some(0.95),
            top_k: Some(50),
            ..Default::default()
        }
    }

    /// Create options optimized for deterministic output
    pub fn deterministic() -> Self {
        Self {
            temperature: Some(0.0),
            top_p: Some(1.0),
            top_k: Some(1),
            ..Default::default()
        }
    }

    /// Create options optimized for balanced output
    pub fn balanced() -> Self {
        Self::default()
    }
}

/// Response from text generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResponse {
    /// Generated text content
    pub content: String,

    /// Number of tokens in the prompt
    pub prompt_tokens: u32,

    /// Number of tokens generated
    pub completion_tokens: u32,

    /// Time to first token (milliseconds)
    pub time_to_first_token_ms: f64,

    /// Tokens per second
    pub tokens_per_second: f64,

    /// Whether generation was stopped early
    pub stopped: bool,
}

/// Embedding result
#[derive(Debug, Clone)]
pub struct Embedding {
    /// The embedding vector
    pub vector: Vec<f32>,
    /// Dimensionality of the embedding
    pub dimension: usize,
}

/// A document stored in the vector index
#[derive(Debug, Clone)]
pub struct Document {
    /// Unique identifier
    pub id: i32,
    /// Document content
    pub content: String,
    /// Optional metadata (JSON string)
    pub metadata: Option<String>,
    /// Optional pre-computed embedding
    pub embedding: Option<Vec<f32>>,
}

impl Document {
    /// Create a new document with content
    pub fn new(id: i32, content: impl Into<String>) -> Self {
        Self {
            id,
            content: content.into(),
            metadata: None,
            embedding: None,
        }
    }

    /// Add metadata to the document
    pub fn with_metadata(mut self, metadata: impl Into<String>) -> Self {
        self.metadata = Some(metadata.into());
        self
    }

    /// Add pre-computed embedding
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }
}

/// Query options for vector search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryOptions {
    /// Number of results to return
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,

    /// Minimum similarity threshold
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_score: Option<f32>,

    /// Include embeddings in results
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_embeddings: Option<bool>,

    /// Include metadata in results
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_metadata: Option<bool>,
}

impl Default for QueryOptions {
    fn default() -> Self {
        Self {
            top_k: Some(10),
            min_score: None,
            include_embeddings: Some(false),
            include_metadata: Some(true),
        }
    }
}

/// A search result from vector query
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Document ID
    pub id: i32,
    /// Similarity score (higher is better)
    pub score: f32,
}

/// Model configuration
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Path to the model file
    pub path: String,
    /// Context window size
    pub context_size: u32,
    /// Whether to use GPU acceleration
    pub use_gpu: bool,
    /// Number of threads for CPU inference
    pub threads: Option<u32>,
}

impl ModelConfig {
    /// Create a new model configuration
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            context_size: 2048,
            use_gpu: true,
            threads: None,
        }
    }

    /// Set the context window size
    pub fn with_context_size(mut self, size: u32) -> Self {
        self.context_size = size;
        self
    }

    /// Enable or disable GPU acceleration
    pub fn with_gpu(mut self, enabled: bool) -> Self {
        self.use_gpu = enabled;
        self
    }

    /// Set the number of CPU threads
    pub fn with_threads(mut self, threads: u32) -> Self {
        self.threads = Some(threads);
        self
    }
}
