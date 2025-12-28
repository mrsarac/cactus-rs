# 🌵 cactus-rs

Safe, idiomatic Rust bindings for [Cactus](https://github.com/cactus-compute/cactus) — a cross-platform, energy-efficient AI inference engine optimized for mobile and edge devices.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)

## Features

| Feature | Description |
|---------|-------------|
| 🚀 **On-Device AI** | Run LLMs locally without internet connectivity |
| 💬 **Chat Inference** | OpenAI-compatible chat completions API |
| 📡 **Streaming** | Token-by-token generation with callbacks |
| 🧮 **Embeddings** | Text, image, and audio embeddings |
| 🔍 **Vector Search** | Built-in semantic search index |
| 🎤 **Transcription** | Whisper-compatible speech-to-text |
| ⚡ **GPU Accelerated** | Metal (macOS/iOS), Vulkan (Android) |

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
cactus = { git = "https://github.com/mrsarac/cactus-rs" }
```

### Prerequisites

- Rust 1.70+
- CMake 3.14+
- Clang (for bindgen)

**macOS:**
```bash
xcode-select --install
brew install cmake
```

**Linux:**
```bash
sudo apt install cmake clang libclang-dev
```

## Quick Start

### Chat Completion

```rust
use cactus::{Model, Message, GenerateOptions};

fn main() -> Result<(), cactus::Error> {
    // Load model
    let model = Model::from_gguf("path/to/model")?;

    // Create conversation
    let messages = vec![
        Message::system("You are a helpful assistant."),
        Message::user("What is the meaning of life?"),
    ];

    // Generate response
    let response = model.complete(&messages, GenerateOptions::default())?;
    println!("{}", response.content);

    Ok(())
}
```

### Streaming Output

```rust
use cactus::{Model, Message, GenerateOptions};
use std::io::{self, Write};

fn main() -> Result<(), cactus::Error> {
    let model = Model::from_gguf("path/to/model")?;
    let messages = vec![Message::user("Tell me a story")];

    model.complete_streaming(&messages, GenerateOptions::default(), |token, _id| {
        print!("{}", token);
        io::stdout().flush().ok();
        true // return false to stop generation
    })?;

    Ok(())
}
```

## API Reference

### Model Configuration

```rust
use cactus::{Model, ModelConfig};

// Simple loading
let model = Model::from_gguf("model/path")?;

// With custom configuration
let config = ModelConfig::new("model/path")
    .with_context_size(4096)  // Context window size
    .with_gpu(true)           // Enable GPU acceleration
    .with_threads(4);         // CPU threads

let model = Model::from_config(config)?;
```

### Generation Options

```rust
use cactus::GenerateOptions;

// Default options
let opts = GenerateOptions::default();

// Creative writing
let opts = GenerateOptions::creative();

// Deterministic output
let opts = GenerateOptions::deterministic();

// Custom options
let opts = GenerateOptions {
    temperature: Some(0.7),
    top_p: Some(0.9),
    top_k: Some(40),
    max_tokens: Some(2048),
    stop: Some(vec!["User:".into()]),
    frequency_penalty: Some(0.5),
    presence_penalty: Some(0.5),
    seed: Some(42),
};
```

### Embeddings

Generate vector embeddings for text, images, or audio:

```rust
use cactus::Model;

let model = Model::from_gguf("embedding-model")?;

// Text embedding (with L2 normalization)
let embedding = model.embed("Hello, world!", true)?;
println!("Dimension: {}", embedding.dimension);  // e.g., 2048
println!("Vector: {:?}", &embedding.vector[..5]);

// Image embedding
let embedding = model.embed_image("photo.jpg")?;

// Audio embedding
let embedding = model.embed_audio("speech.wav")?;
```

### Vector Search

Semantic search using embedding similarity:

```rust
use cactus::{Model, VectorIndex, Document, QueryOptions};

let model = Model::from_gguf("model")?;

// Create index
let index = VectorIndex::new("./my_index", 2048)?;  // 2048 = embedding dimension

// Add documents
let docs = vec![
    Document::new(1, "Rust is a systems programming language")
        .with_embedding(model.embed("Rust is a systems programming language", true)?.vector)
        .with_metadata(r#"{"category": "programming"}"#),
    Document::new(2, "Python is great for data science")
        .with_embedding(model.embed("Python is great for data science", true)?.vector)
        .with_metadata(r#"{"category": "programming"}"#),
];
index.add(&docs)?;

// Search
let query_embedding = model.embed("What language is good for systems?", true)?;
let results = index.query(&query_embedding.vector, QueryOptions {
    top_k: Some(5),
    min_score: Some(0.5),
    ..Default::default()
})?;

for result in results {
    println!("ID: {}, Score: {:.3}", result.id, result.score);
}

// Other operations
index.delete(&[1, 2])?;  // Delete by IDs
index.compact()?;        // Optimize storage
```

### Audio Transcription

Speech-to-text using Whisper-compatible models:

```rust
use cactus::{Model, TranscribeOptions};
use std::io::{self, Write};

let model = Model::from_gguf("whisper-model")?;

// Basic transcription
let result = model.transcribe("speech.wav", TranscribeOptions::default())?;
println!("Text: {}", result.text);
println!("Language: {}", result.language);
println!("Duration: {:.2}s", result.duration);

// With options
let opts = TranscribeOptions::with_language("en")
    .with_timestamps()
    .with_prompt("Meeting transcript:");

let result = model.transcribe("meeting.wav", opts)?;

// Show segments with timestamps
for seg in &result.segments {
    println!("[{:.2}s - {:.2}s] {}", seg.start, seg.end, seg.text);
}

// Streaming transcription
model.transcribe_streaming("speech.wav", opts, |text, _| {
    print!("{}", text);
    io::stdout().flush().ok();
    true
})?;

// From raw PCM data (16-bit, mono, 16kHz)
let pcm_data: Vec<u8> = capture_microphone();
let result = model.transcribe_pcm(&pcm_data, opts)?;
```

### Model Control

```rust
// Stop ongoing generation (from another thread)
model.stop();

// Reset KV cache
model.reset();

// Get configuration
let config = model.config();
println!("Model path: {}", config.path);
```

## Examples

Run the included examples:

```bash
# Chat completion
cargo run --release --example simple_chat -- path/to/model

# Embeddings with similarity matrix
cargo run --release --example embeddings -- path/to/model

# Vector search demo
cargo run --release --example vector_search -- path/to/model

# Audio transcription
cargo run --release --example transcribe -- path/to/whisper-model audio.wav
```

## Crate Structure

```
cactus-rs/
├── cactus-sys/          # Low-level FFI bindings (unsafe)
│   ├── build.rs         # CMake + bindgen integration
│   └── src/lib.rs       # Raw C bindings
│
├── cactus/              # Safe, idiomatic Rust API
│   ├── src/
│   │   ├── lib.rs       # Public exports
│   │   ├── model.rs     # Model + inference
│   │   ├── index.rs     # VectorIndex
│   │   ├── types.rs     # All types
│   │   └── error.rs     # Error handling
│   └── examples/
│       ├── simple_chat.rs
│       ├── embeddings.rs
│       ├── vector_search.rs
│       └── transcribe.rs
│
└── vendor/cactus/       # Cactus C++ engine (git submodule)
```

## Building from Source

```bash
# Clone with submodules
git clone --recursive https://github.com/mrsarac/cactus-rs
cd cactus-rs

# Build
cargo build --release

# Run tests
cargo test

# Build documentation
cargo doc --open
```

## Supported Models

### LLM (Chat/Completion)

| Model | Tested | Notes |
|-------|--------|-------|
| LFM2-1.2B | ✅ | Default test model |
| Gemma 2B/7B | ✅ | Recommended for edge |
| Llama 2/3 | ⚠️ | Requires conversion |
| Qwen | ⚠️ | Requires conversion |

### Whisper (Transcription)

| Model | Parameters | Speed |
|-------|------------|-------|
| whisper-tiny | 39M | Fastest |
| whisper-base | 74M | Fast |
| whisper-small | 244M | Balanced |
| whisper-medium | 769M | Accurate |
| whisper-large | 1.5B | Most accurate |

### Model Download

```bash
# Using Cactus CLI (if available)
cactus download lfm2-1.2b

# Or download manually from HuggingFace
# Models should be in Cactus weight folder format
```

## Platform Support

| Platform | Backend | Status |
|----------|---------|--------|
| macOS (Apple Silicon) | Metal | ✅ Tested |
| macOS (Intel) | CPU | ✅ Tested |
| Linux (x86_64) | CPU/OpenBLAS | ✅ Supported |
| iOS | Metal | 🚧 Planned |
| Android | Vulkan | 🚧 Planned |
| Windows | DirectX/CPU | 🚧 Planned |

## Performance

Benchmarks on Apple M1 MacBook Pro with LFM2-1.2B:

| Operation | Performance |
|-----------|-------------|
| Model load | ~2s |
| First token | ~150ms |
| Token generation | ~45 tokens/sec |
| Embedding (2048-dim) | ~50ms |
| Vector query (1K docs) | ~5ms |

## Error Handling

```rust
use cactus::{Model, Error};

fn main() {
    match Model::from_gguf("nonexistent.gguf") {
        Ok(model) => println!("Model loaded!"),
        Err(Error::ModelLoad(msg)) => eprintln!("Failed to load: {}", msg),
        Err(Error::ModelInit(msg)) => eprintln!("Failed to init: {}", msg),
        Err(Error::NullPointer) => eprintln!("FFI returned null"),
        Err(e) => eprintln!("Other error: {}", e),
    }
}
```

## Thread Safety

`Model` implements `Send` and `Sync`, making it safe to share across threads:

```rust
use std::sync::Arc;
use std::thread;

let model = Arc::new(Model::from_gguf("model")?);

let handles: Vec<_> = (0..4).map(|i| {
    let model = Arc::clone(&model);
    thread::spawn(move || {
        let messages = vec![Message::user(format!("Thread {} says hi!", i))];
        model.complete(&messages, GenerateOptions::default())
    })
}).collect();

for handle in handles {
    let result = handle.join().unwrap()?;
    println!("{}", result.content);
}
```

## Related Projects

- [Cactus](https://github.com/cactus-compute/cactus) - The original C++ inference engine
- [llama-cpp-rs](https://github.com/mdrokz/rust-llama.cpp) - Alternative Rust LLM bindings

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Cactus Compute](https://github.com/cactus-compute) for the amazing inference engine
- The Rust community for excellent FFI tooling (bindgen, cmake-rs)

---

<p align="center">
  Built with 🌵 by <a href="https://neurabytelabs.com">NeuraByte Labs</a>
</p>
