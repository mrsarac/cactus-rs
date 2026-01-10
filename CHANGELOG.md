# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-01-10

### Added
- Speech-to-Text streaming API support (Cactus v1.5)
- `StreamTranscriber` struct for real-time audio transcription
- `StreamTranscriberBuilder` for configuration (language, VAD, realtime factor)
- `feed()` method for incremental audio chunk processing
- `take_segments()` for retrieving finalized transcription segments
- `finalize()` for flushing remaining audio buffer

### Changed
- Updated Cactus submodule to v1.5
- INT4 quantization support (upstream)
- KV cache quantization support (upstream)
- Improved documentation with streaming transcription examples

### Upstream (Cactus v1.5)
- INT4 quantization for reduced model size
- KV cache quantization for memory efficiency
- Streaming transcription API
- Performance improvements

## [0.1.0] - 2025-12-29

### Added
- Initial release
- Safe Rust bindings for Cactus inference engine
- LLM chat completions with streaming support
- Text, image, and audio embeddings
- Vector search index with semantic similarity
- Whisper-compatible speech-to-text transcription
- Metal GPU acceleration (macOS/iOS)
- Vulkan GPU acceleration (Android) - feature flag
- `Model` struct with `Send + Sync` for thread safety
- `VectorIndex` for document storage and search
- `GenerateOptions` presets (creative, deterministic)
- `TranscribeOptions` with language and timestamp support
- Example programs: simple_chat, embeddings, vector_search, transcribe
- Comprehensive error handling with `thiserror`

[0.2.0]: https://github.com/mrsarac/cactus-rs/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/mrsarac/cactus-rs/releases/tag/v0.1.0
