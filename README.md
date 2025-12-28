# 🌵 cactus-rs

Rust bindings for [Cactus](https://github.com/cactus-compute/cactus), a cross-platform, energy-efficient AI inference engine for mobile devices.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)

## Features

- 🚀 **On-Device AI** - Run LLMs locally without internet
- 🎨 **Multi-Modal** - Text, image, and audio support
- 📡 **Streaming** - Token-by-token generation
- 🔍 **Vector Index** - Built-in semantic search
- ⚡ **GPU Accelerated** - Metal (macOS/iOS), Vulkan (Android)

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

```rust
use cactus::{Model, Message, GenerateOptions};

fn main() -> Result<(), cactus::Error> {
    // Load a GGUF model
    let model = Model::from_gguf("gemma-2b-it-q4.gguf")?;

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

## Streaming

```rust
use cactus::{Model, Message, GenerateOptions};
use std::io::Write;

fn main() -> Result<(), cactus::Error> {
    let model = Model::from_gguf("model.gguf")?;
    let messages = vec![Message::user("Tell me a story about a robot")];

    model.complete_streaming(&messages, GenerateOptions::creative(), |token| {
        print!("{}", token);
        std::io::stdout().flush().ok();
        true // continue generation
    })?;

    println!();
    Ok(())
}
```

## Configuration

```rust
use cactus::{Model, ModelConfig, GenerateOptions};

// Custom model configuration
let config = ModelConfig::new("model.gguf")
    .with_context_size(4096)  // Larger context window
    .with_gpu(true)           // Enable GPU acceleration
    .with_threads(4);         // CPU threads for fallback

let model = Model::from_config(config)?;

// Generation options
let options = GenerateOptions {
    temperature: Some(0.8),
    top_p: Some(0.95),
    max_tokens: Some(1024),
    ..Default::default()
};
```

## Crate Structure

| Crate | Description |
|-------|-------------|
| `cactus-sys` | Low-level FFI bindings (unsafe) |
| `cactus` | Safe, idiomatic Rust API |

## Building from Source

```bash
# Clone with submodules
git clone --recursive https://github.com/mrsarac/cactus-rs
cd cactus-rs

# Build
cargo build --release

# Run tests
cargo test
```

## Supported Models

Any GGUF-format model compatible with Cactus:

- Gemma 2B / 7B
- Llama 2/3
- Qwen
- And more...

## Platform Support

| Platform | Backend | Status |
|----------|---------|--------|
| macOS | Metal | ✅ Supported |
| Linux | CPU/OpenBLAS | ✅ Supported |
| iOS | Metal | 🚧 Planned |
| Android | Vulkan | 🚧 Planned |

## Related Projects

- [Cactus](https://github.com/cactus-compute/cactus) - The original C++ engine
- [AYAS](https://github.com/mrsarac/ayas) - Terminal knowledge visualizer (uses cactus-rs)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

Built with 🌵 by [NeuraByte Labs](https://neurabytelabs.com)
