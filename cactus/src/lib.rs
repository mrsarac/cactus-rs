//! # Cactus
//!
//! Safe, idiomatic Rust bindings for [Cactus](https://github.com/cactus-compute/cactus),
//! a cross-platform, energy-efficient AI inference engine for mobile devices.
//!
//! ## Features
//!
//! - **On-Device AI** - Run LLMs locally without internet
//! - **Multi-Modal** - Text, image, and audio support
//! - **Streaming** - Token-by-token generation
//! - **Vector Index** - Built-in semantic search
//! - **GPU Accelerated** - Metal (macOS/iOS), Vulkan (Android)
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use cactus::{Model, Message, Role, GenerateOptions};
//!
//! fn main() -> Result<(), cactus::Error> {
//!     // Load model
//!     let model = Model::from_gguf("gemma-2b-it-q4.gguf")?;
//!
//!     // Create conversation
//!     let messages = vec![
//!         Message::system("You are a helpful assistant."),
//!         Message::user("Hello, how are you?"),
//!     ];
//!
//!     // Generate response
//!     let response = model.complete(&messages, GenerateOptions::default())?;
//!     println!("{}", response.content);
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Streaming
//!
//! ```rust,ignore
//! use cactus::{Model, Message, GenerateOptions};
//!
//! let model = Model::from_gguf("model.gguf")?;
//! let messages = vec![Message::user("Tell me a story")];
//!
//! model.complete_streaming(&messages, GenerateOptions::default(), |token| {
//!     print!("{}", token);
//!     true // continue generation
//! })?;
//! ```

mod error;
mod index;
mod model;
mod types;

pub use error::{Error, Result};
pub use index::VectorIndex;
pub use model::Model;
pub use types::*;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
