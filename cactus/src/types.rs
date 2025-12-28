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

/// Options for audio transcription
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscribeOptions {
    /// Language code (e.g., "en", "de", "tr")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,

    /// Whether to include timestamps in output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamps: Option<bool>,

    /// Maximum duration to transcribe (seconds)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_duration: Option<f32>,

    /// Temperature for sampling (0.0 = deterministic)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Initial prompt to guide transcription style
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initial_prompt: Option<String>,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            language: None, // auto-detect
            timestamps: Some(false),
            max_duration: None,
            temperature: Some(0.0),
            initial_prompt: None,
        }
    }
}

impl TranscribeOptions {
    /// Create options for a specific language
    pub fn with_language(language: impl Into<String>) -> Self {
        Self {
            language: Some(language.into()),
            ..Default::default()
        }
    }

    /// Enable timestamp output
    pub fn with_timestamps(mut self) -> Self {
        self.timestamps = Some(true);
        self
    }

    /// Set initial prompt for context
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.initial_prompt = Some(prompt.into());
        self
    }
}

/// Response from audio transcription
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscribeResponse {
    /// Transcribed text
    pub text: String,

    /// Detected or specified language
    #[serde(default)]
    pub language: String,

    /// Duration of audio processed (seconds)
    #[serde(default)]
    pub duration: f32,

    /// Segments with timestamps (if requested)
    #[serde(default)]
    pub segments: Vec<TranscribeSegment>,
}

/// A segment of transcribed audio with timing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscribeSegment {
    /// Start time (seconds)
    pub start: f32,
    /// End time (seconds)
    pub end: f32,
    /// Transcribed text for this segment
    pub text: String,
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

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Message Tests ====================

    #[test]
    fn test_message_system() {
        let msg = Message::system("You are helpful");
        assert_eq!(msg.role, Role::System);
        assert_eq!(msg.content, "You are helpful");
        assert!(msg.images.is_none());
    }

    #[test]
    fn test_message_user() {
        let msg = Message::user("Hello!");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content, "Hello!");
    }

    #[test]
    fn test_message_assistant() {
        let msg = Message::assistant("Hi there!");
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.content, "Hi there!");
    }

    #[test]
    fn test_message_with_images() {
        let msg = Message::user("Look at this")
            .with_images(vec!["image1.jpg".into(), "image2.png".into()]);
        assert_eq!(msg.images.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_message_serialization() {
        let msg = Message::user("Test");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"Test\""));
    }

    // ==================== GenerateOptions Tests ====================

    #[test]
    fn test_generate_options_default() {
        let opts = GenerateOptions::default();
        assert_eq!(opts.temperature, Some(0.7));
        assert_eq!(opts.top_p, Some(0.9));
        assert_eq!(opts.top_k, Some(40));
        assert_eq!(opts.max_tokens, Some(2048));
    }

    #[test]
    fn test_generate_options_creative() {
        let opts = GenerateOptions::creative();
        assert_eq!(opts.temperature, Some(1.0));
        assert_eq!(opts.top_p, Some(0.95));
        assert_eq!(opts.top_k, Some(50));
    }

    #[test]
    fn test_generate_options_deterministic() {
        let opts = GenerateOptions::deterministic();
        assert_eq!(opts.temperature, Some(0.0));
        assert_eq!(opts.top_p, Some(1.0));
        assert_eq!(opts.top_k, Some(1));
    }

    #[test]
    fn test_generate_options_serialization() {
        let opts = GenerateOptions {
            temperature: Some(0.5),
            max_tokens: Some(100),
            ..Default::default()
        };
        let json = serde_json::to_string(&opts).unwrap();
        assert!(json.contains("\"temperature\":0.5"));
        assert!(json.contains("\"max_tokens\":100"));
    }

    // ==================== Document Tests ====================

    #[test]
    fn test_document_new() {
        let doc = Document::new(1, "Hello world");
        assert_eq!(doc.id, 1);
        assert_eq!(doc.content, "Hello world");
        assert!(doc.metadata.is_none());
        assert!(doc.embedding.is_none());
    }

    #[test]
    fn test_document_with_metadata() {
        let doc = Document::new(1, "Test")
            .with_metadata(r#"{"category": "test"}"#);
        assert_eq!(doc.metadata.as_ref().unwrap(), r#"{"category": "test"}"#);
    }

    #[test]
    fn test_document_with_embedding() {
        let embedding = vec![0.1, 0.2, 0.3];
        let doc = Document::new(1, "Test")
            .with_embedding(embedding.clone());
        assert_eq!(doc.embedding.as_ref().unwrap(), &embedding);
    }

    #[test]
    fn test_document_builder_chain() {
        let doc = Document::new(42, "Content")
            .with_metadata(r#"{"key": "value"}"#)
            .with_embedding(vec![1.0, 2.0, 3.0]);

        assert_eq!(doc.id, 42);
        assert_eq!(doc.content, "Content");
        assert!(doc.metadata.is_some());
        assert!(doc.embedding.is_some());
    }

    // ==================== QueryOptions Tests ====================

    #[test]
    fn test_query_options_default() {
        let opts = QueryOptions::default();
        assert_eq!(opts.top_k, Some(10));
        assert!(opts.min_score.is_none());
        assert_eq!(opts.include_embeddings, Some(false));
        assert_eq!(opts.include_metadata, Some(true));
    }

    #[test]
    fn test_query_options_serialization() {
        let opts = QueryOptions {
            top_k: Some(5),
            min_score: Some(0.5),
            ..Default::default()
        };
        let json = serde_json::to_string(&opts).unwrap();
        assert!(json.contains("\"top_k\":5"));
        assert!(json.contains("\"min_score\":0.5"));
    }

    // ==================== TranscribeOptions Tests ====================

    #[test]
    fn test_transcribe_options_default() {
        let opts = TranscribeOptions::default();
        assert!(opts.language.is_none());
        assert_eq!(opts.timestamps, Some(false));
        assert_eq!(opts.temperature, Some(0.0));
    }

    #[test]
    fn test_transcribe_options_with_language() {
        let opts = TranscribeOptions::with_language("en");
        assert_eq!(opts.language, Some("en".to_string()));
    }

    #[test]
    fn test_transcribe_options_with_timestamps() {
        let opts = TranscribeOptions::default().with_timestamps();
        assert_eq!(opts.timestamps, Some(true));
    }

    #[test]
    fn test_transcribe_options_with_prompt() {
        let opts = TranscribeOptions::default()
            .with_prompt("Meeting notes:");
        assert_eq!(opts.initial_prompt, Some("Meeting notes:".to_string()));
    }

    #[test]
    fn test_transcribe_options_builder_chain() {
        let opts = TranscribeOptions::with_language("de")
            .with_timestamps()
            .with_prompt("Transcript:");

        assert_eq!(opts.language, Some("de".to_string()));
        assert_eq!(opts.timestamps, Some(true));
        assert_eq!(opts.initial_prompt, Some("Transcript:".to_string()));
    }

    // ==================== ModelConfig Tests ====================

    #[test]
    fn test_model_config_new() {
        let config = ModelConfig::new("model.gguf");
        assert_eq!(config.path, "model.gguf");
        assert_eq!(config.context_size, 2048);
        assert!(config.use_gpu);
        assert!(config.threads.is_none());
    }

    #[test]
    fn test_model_config_with_context_size() {
        let config = ModelConfig::new("model")
            .with_context_size(4096);
        assert_eq!(config.context_size, 4096);
    }

    #[test]
    fn test_model_config_with_gpu() {
        let config = ModelConfig::new("model")
            .with_gpu(false);
        assert!(!config.use_gpu);
    }

    #[test]
    fn test_model_config_with_threads() {
        let config = ModelConfig::new("model")
            .with_threads(8);
        assert_eq!(config.threads, Some(8));
    }

    #[test]
    fn test_model_config_builder_chain() {
        let config = ModelConfig::new("/path/to/model")
            .with_context_size(8192)
            .with_gpu(true)
            .with_threads(4);

        assert_eq!(config.path, "/path/to/model");
        assert_eq!(config.context_size, 8192);
        assert!(config.use_gpu);
        assert_eq!(config.threads, Some(4));
    }

    // ==================== Embedding Tests ====================

    #[test]
    fn test_embedding_struct() {
        let emb = Embedding {
            vector: vec![0.1, 0.2, 0.3, 0.4],
            dimension: 4,
        };
        assert_eq!(emb.dimension, 4);
        assert_eq!(emb.vector.len(), 4);
    }

    // ==================== SearchResult Tests ====================

    #[test]
    fn test_search_result() {
        let result = SearchResult {
            id: 42,
            score: 0.95,
        };
        assert_eq!(result.id, 42);
        assert!((result.score - 0.95).abs() < f32::EPSILON);
    }

    // ==================== GenerateResponse Tests ====================

    #[test]
    fn test_generate_response_deserialization() {
        let json = r#"{
            "content": "Hello!",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "time_to_first_token_ms": 150.5,
            "tokens_per_second": 45.2,
            "stopped": false
        }"#;

        let response: GenerateResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.content, "Hello!");
        assert_eq!(response.prompt_tokens, 10);
        assert_eq!(response.completion_tokens, 5);
        assert!(!response.stopped);
    }

    // ==================== TranscribeResponse Tests ====================

    #[test]
    fn test_transcribe_response_deserialization() {
        let json = r#"{
            "text": "Hello world",
            "language": "en",
            "duration": 5.5,
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "Hello"},
                {"start": 2.5, "end": 5.5, "text": "world"}
            ]
        }"#;

        let response: TranscribeResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.text, "Hello world");
        assert_eq!(response.language, "en");
        assert_eq!(response.segments.len(), 2);
        assert_eq!(response.segments[0].text, "Hello");
    }

    // ==================== Role Tests ====================

    #[test]
    fn test_role_serialization() {
        assert_eq!(serde_json::to_string(&Role::System).unwrap(), "\"system\"");
        assert_eq!(serde_json::to_string(&Role::User).unwrap(), "\"user\"");
        assert_eq!(serde_json::to_string(&Role::Assistant).unwrap(), "\"assistant\"");
        assert_eq!(serde_json::to_string(&Role::Tool).unwrap(), "\"tool\"");
    }

    #[test]
    fn test_role_deserialization() {
        let system: Role = serde_json::from_str("\"system\"").unwrap();
        let user: Role = serde_json::from_str("\"user\"").unwrap();
        assert_eq!(system, Role::System);
        assert_eq!(user, Role::User);
    }
}
