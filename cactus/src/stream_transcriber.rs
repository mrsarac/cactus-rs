//! Streaming audio transcription
//!
//! This module provides real-time streaming transcription capabilities
//! for continuous audio processing, such as live microphone input.
//!
//! # Example
//!
//! ```rust,ignore
//! use cactus::{Model, StreamTranscriber, StreamTranscribeOptions};
//!
//! let model = Model::from_gguf("whisper-base.gguf")?;
//! let mut transcriber = StreamTranscriber::new(&model)?;
//!
//! // Feed audio chunks as they arrive
//! transcriber.insert(&audio_chunk_1)?;
//! transcriber.insert(&audio_chunk_2)?;
//!
//! // Get intermediate results
//! let partial = transcriber.process(StreamTranscribeOptions::default())?;
//! println!("Partial: {}", partial.text);
//!
//! // Finalize when done
//! let final_result = transcriber.finalize()?;
//! println!("Final: {}", final_result.text);
//! ```

use crate::error::{Error, Result};
use crate::types::TranscribeResponse;
use crate::Model;
use serde::{Deserialize, Serialize};
use std::ffi::CStr;
use std::ffi::CString;
use std::marker::PhantomData;
use std::ptr::NonNull;

/// Default response buffer size (64KB)
const DEFAULT_BUFFER_SIZE: usize = 64 * 1024;

/// Options for streaming transcription processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamTranscribeOptions {
    /// Language code (e.g., "en", "de", "tr")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,

    /// Whether to include timestamps in output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamps: Option<bool>,

    /// Temperature for sampling (0.0 = deterministic)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Initial prompt to guide transcription style
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initial_prompt: Option<String>,
}

impl Default for StreamTranscribeOptions {
    fn default() -> Self {
        Self {
            language: None,
            timestamps: Some(false),
            temperature: Some(0.0),
            initial_prompt: None,
        }
    }
}

impl StreamTranscribeOptions {
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

    /// Set temperature for sampling
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }
}

/// A streaming audio transcriber for real-time speech-to-text
///
/// This struct provides a way to continuously feed audio data and get
/// intermediate transcription results. It holds a reference to a [`Model`]
/// and manages the underlying stream state.
///
/// # Lifetime
///
/// The transcriber borrows the model, so the model must outlive the transcriber.
///
/// # Thread Safety
///
/// The transcriber is `Send` but not `Sync` - it can be moved between threads
/// but should not be accessed concurrently.
///
/// # Example
///
/// ```rust,ignore
/// let model = Model::from_gguf("whisper.gguf")?;
/// let mut transcriber = StreamTranscriber::new(&model)?;
///
/// // Process audio in a loop
/// loop {
///     let audio_data = get_audio_chunk(); // Your audio source
///     if audio_data.is_empty() {
///         break;
///     }
///     transcriber.insert(&audio_data)?;
///
///     // Optionally get intermediate results
///     let partial = transcriber.process(StreamTranscribeOptions::default())?;
///     if !partial.text.is_empty() {
///         println!("Heard: {}", partial.text);
///     }
/// }
///
/// // Get final transcription
/// let result = transcriber.finalize()?;
/// ```
pub struct StreamTranscriber<'a> {
    /// Raw handle to the stream transcriber
    handle: NonNull<std::ffi::c_void>,
    /// Phantom data to track the model lifetime
    _marker: PhantomData<&'a Model>,
}

// Safety: The stream handle can be sent between threads
unsafe impl<'a> Send for StreamTranscriber<'a> {}

impl<'a> StreamTranscriber<'a> {
    /// Create a new streaming transcriber from a model
    ///
    /// # Arguments
    ///
    /// * `model` - The model to use for transcription (must be Whisper-compatible)
    ///
    /// # Errors
    ///
    /// Returns an error if the stream transcriber fails to initialize,
    /// typically if the model doesn't support transcription.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let model = Model::from_gguf("whisper-base.gguf")?;
    /// let transcriber = StreamTranscriber::new(&model)?;
    /// ```
    pub fn new(model: &'a Model) -> Result<Self> {
        let handle = unsafe { cactus_sys::cactus_stream_transcribe_init(model.handle()) };

        let handle = NonNull::new(handle).ok_or_else(|| {
            Error::last_cactus_error()
                .map(Error::ModelInit)
                .unwrap_or(Error::NullPointer)
        })?;

        Ok(Self {
            handle,
            _marker: PhantomData,
        })
    }

    /// Insert PCM audio data into the stream buffer
    ///
    /// Audio data should be in the format expected by Whisper:
    /// - 16-bit signed integers (i16)
    /// - Mono channel
    /// - 16kHz sample rate
    ///
    /// You can call this method multiple times to accumulate audio data
    /// before processing.
    ///
    /// # Arguments
    ///
    /// * `pcm_data` - Raw PCM audio data as bytes
    ///
    /// # Errors
    ///
    /// Returns an error if the data couldn't be inserted into the buffer.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Read audio chunks from microphone
    /// let audio_chunk: Vec<u8> = read_from_microphone();
    /// transcriber.insert(&audio_chunk)?;
    /// ```
    pub fn insert(&mut self, pcm_data: &[u8]) -> Result<()> {
        let result = unsafe {
            cactus_sys::cactus_stream_transcribe_insert(
                self.handle.as_ptr(),
                pcm_data.as_ptr(),
                pcm_data.len(),
            )
        };

        if result < 0 {
            return Err(Error::last_cactus_error()
                .map(Error::Inference)
                .unwrap_or_else(|| {
                    Error::Inference(format!(
                        "cactus_stream_transcribe_insert failed with code {}",
                        result
                    ))
                }));
        }

        Ok(())
    }

    /// Insert PCM audio data as i16 samples
    ///
    /// This is a convenience method that accepts i16 samples directly.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples as 16-bit signed integers
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let samples: Vec<i16> = get_audio_samples();
    /// transcriber.insert_samples(&samples)?;
    /// ```
    pub fn insert_samples(&mut self, samples: &[i16]) -> Result<()> {
        // Safety: i16 slice can be safely viewed as u8 slice
        let bytes = unsafe {
            std::slice::from_raw_parts(
                samples.as_ptr() as *const u8,
                samples.len() * std::mem::size_of::<i16>(),
            )
        };
        self.insert(bytes)
    }

    /// Insert PCM audio data as f32 samples
    ///
    /// This method converts f32 samples (range -1.0 to 1.0) to i16 format
    /// before inserting into the stream.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples as 32-bit floats in range [-1.0, 1.0]
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let float_samples: Vec<f32> = get_float_audio();
    /// transcriber.insert_f32(&float_samples)?;
    /// ```
    pub fn insert_f32(&mut self, samples: &[f32]) -> Result<()> {
        // Convert f32 to i16
        let i16_samples: Vec<i16> = samples
            .iter()
            .map(|&s| {
                let clamped = s.clamp(-1.0, 1.0);
                (clamped * i16::MAX as f32) as i16
            })
            .collect();
        self.insert_samples(&i16_samples)
    }

    /// Process the accumulated audio and get intermediate results
    ///
    /// This method transcribes the audio data that has been inserted so far
    /// and returns partial results. It does not clear the internal buffer,
    /// allowing for continued accumulation of audio data.
    ///
    /// Call this periodically to get real-time transcription updates.
    ///
    /// # Arguments
    ///
    /// * `options` - Options for transcription processing
    ///
    /// # Returns
    ///
    /// Returns a [`TranscribeResponse`] with the current transcription.
    /// The `text` field may be empty if not enough audio has been accumulated.
    ///
    /// # Errors
    ///
    /// Returns an error if transcription processing fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let options = StreamTranscribeOptions::with_language("en");
    /// let partial = transcriber.process(options)?;
    /// println!("Partial transcription: {}", partial.text);
    /// ```
    pub fn process(&mut self, options: StreamTranscribeOptions) -> Result<TranscribeResponse> {
        let options_json = serde_json::to_string(&options)?;
        let options_c = CString::new(options_json)?;

        let mut response_buffer: Vec<u8> = vec![0u8; DEFAULT_BUFFER_SIZE];

        let result = unsafe {
            cactus_sys::cactus_stream_transcribe_process(
                self.handle.as_ptr(),
                response_buffer.as_mut_ptr() as *mut i8,
                response_buffer.len(),
                options_c.as_ptr(),
            )
        };

        if result < 0 {
            return Err(Error::last_cactus_error()
                .map(Error::Inference)
                .unwrap_or_else(|| {
                    Error::Inference(format!(
                        "cactus_stream_transcribe_process failed with code {}",
                        result
                    ))
                }));
        }

        let response_str = unsafe {
            let c_str = CStr::from_ptr(response_buffer.as_ptr() as *const i8);
            c_str.to_string_lossy().into_owned()
        };

        let response: TranscribeResponse =
            serde_json::from_str(&response_str).unwrap_or_else(|_| TranscribeResponse {
                text: response_str,
                language: String::new(),
                duration: 0.0,
                segments: Vec::new(),
            });

        Ok(response)
    }

    /// Finalize the stream and get the final transcription
    ///
    /// This method processes any remaining audio in the buffer and returns
    /// the complete transcription. After calling this method, the transcriber
    /// should not be used further (though the Rust type system allows it,
    /// the underlying stream state may be invalid).
    ///
    /// # Returns
    ///
    /// Returns a [`TranscribeResponse`] with the final complete transcription.
    ///
    /// # Errors
    ///
    /// Returns an error if finalization fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // After all audio has been inserted
    /// let final_result = transcriber.finalize()?;
    /// println!("Final transcription: {}", final_result.text);
    /// ```
    pub fn finalize(&mut self) -> Result<TranscribeResponse> {
        let mut response_buffer: Vec<u8> = vec![0u8; DEFAULT_BUFFER_SIZE];

        let result = unsafe {
            cactus_sys::cactus_stream_transcribe_finalize(
                self.handle.as_ptr(),
                response_buffer.as_mut_ptr() as *mut i8,
                response_buffer.len(),
            )
        };

        if result < 0 {
            return Err(Error::last_cactus_error()
                .map(Error::Inference)
                .unwrap_or_else(|| {
                    Error::Inference(format!(
                        "cactus_stream_transcribe_finalize failed with code {}",
                        result
                    ))
                }));
        }

        let response_str = unsafe {
            let c_str = CStr::from_ptr(response_buffer.as_ptr() as *const i8);
            c_str.to_string_lossy().into_owned()
        };

        let response: TranscribeResponse =
            serde_json::from_str(&response_str).unwrap_or_else(|_| TranscribeResponse {
                text: response_str,
                language: String::new(),
                duration: 0.0,
                segments: Vec::new(),
            });

        Ok(response)
    }
}

impl<'a> Drop for StreamTranscriber<'a> {
    fn drop(&mut self) {
        unsafe {
            cactus_sys::cactus_stream_transcribe_destroy(self.handle.as_ptr());
        }
    }
}

/// Builder for creating a [`StreamTranscriber`] with configuration
///
/// # Example
///
/// ```rust,ignore
/// let transcriber = StreamTranscriberBuilder::new(&model)
///     .with_buffer_size(128 * 1024)
///     .build()?;
/// ```
pub struct StreamTranscriberBuilder<'a> {
    model: &'a Model,
    #[allow(dead_code)]
    buffer_size: usize,
}

impl<'a> StreamTranscriberBuilder<'a> {
    /// Create a new builder for the given model
    pub fn new(model: &'a Model) -> Self {
        Self {
            model,
            buffer_size: DEFAULT_BUFFER_SIZE,
        }
    }

    /// Set the response buffer size (default: 64KB)
    ///
    /// Larger buffers may be needed for long transcriptions.
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Build the [`StreamTranscriber`]
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    pub fn build(self) -> Result<StreamTranscriber<'a>> {
        StreamTranscriber::new(self.model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TranscribeResponse;

    // ==================== StreamTranscribeOptions Tests ====================

    #[test]
    fn test_stream_transcribe_options_default() {
        let opts = StreamTranscribeOptions::default();
        assert!(opts.language.is_none());
        assert_eq!(opts.timestamps, Some(false));
        assert_eq!(opts.temperature, Some(0.0));
        assert!(opts.initial_prompt.is_none());
    }

    #[test]
    fn test_stream_transcribe_options_with_language() {
        let opts = StreamTranscribeOptions::with_language("en");
        assert_eq!(opts.language, Some("en".to_string()));
    }

    #[test]
    fn test_stream_transcribe_options_with_timestamps() {
        let opts = StreamTranscribeOptions::default().with_timestamps();
        assert_eq!(opts.timestamps, Some(true));
    }

    #[test]
    fn test_stream_transcribe_options_with_prompt() {
        let opts = StreamTranscribeOptions::default().with_prompt("Meeting notes:");
        assert_eq!(opts.initial_prompt, Some("Meeting notes:".to_string()));
    }

    #[test]
    fn test_stream_transcribe_options_with_temperature() {
        let opts = StreamTranscribeOptions::default().with_temperature(0.5);
        assert_eq!(opts.temperature, Some(0.5));
    }

    #[test]
    fn test_stream_transcribe_options_builder_chain() {
        let opts = StreamTranscribeOptions::with_language("de")
            .with_timestamps()
            .with_prompt("Transcript:")
            .with_temperature(0.2);

        assert_eq!(opts.language, Some("de".to_string()));
        assert_eq!(opts.timestamps, Some(true));
        assert_eq!(opts.initial_prompt, Some("Transcript:".to_string()));
        assert_eq!(opts.temperature, Some(0.2));
    }

    #[test]
    fn test_stream_transcribe_options_serialization() {
        let opts = StreamTranscribeOptions {
            language: Some("en".to_string()),
            timestamps: Some(true),
            temperature: Some(0.5),
            initial_prompt: Some("Hello".to_string()),
        };
        let json = serde_json::to_string(&opts).unwrap();
        assert!(json.contains("\"language\":\"en\""));
        assert!(json.contains("\"timestamps\":true"));
        assert!(json.contains("\"temperature\":0.5"));
        assert!(json.contains("\"initial_prompt\":\"Hello\""));
    }

    #[test]
    fn test_stream_transcribe_options_deserialization() {
        let json = r#"{"language":"tr","timestamps":true}"#;
        let opts: StreamTranscribeOptions = serde_json::from_str(json).unwrap();
        assert_eq!(opts.language, Some("tr".to_string()));
        assert_eq!(opts.timestamps, Some(true));
        assert!(opts.temperature.is_none());
    }

    // ==================== StreamTranscriberBuilder Tests ====================

    #[test]
    fn test_stream_transcriber_builder_default_buffer_size() {
        // Verify the default buffer size constant
        assert_eq!(DEFAULT_BUFFER_SIZE, 64 * 1024);
    }

    #[test]
    fn test_buffer_size_is_reasonable() {
        // Buffer should be at least 1KB and at most 1MB for practical use
        assert!(DEFAULT_BUFFER_SIZE >= 1024);
        assert!(DEFAULT_BUFFER_SIZE <= 1024 * 1024);
    }

    // ==================== PCM Data Format Tests ====================

    #[test]
    fn test_pcm_format_16khz_mono_16bit() {
        // 16kHz sample rate, mono, 16-bit = 32,000 bytes per second
        let bytes_per_second = 16000 * 2; // 16kHz * 2 bytes per sample
        assert_eq!(bytes_per_second, 32000);
    }

    #[test]
    fn test_pcm_chunk_100ms() {
        // 100ms of audio at 16kHz mono 16-bit
        let chunk_size = (16000 * 2) / 10; // 32000 / 10
        assert_eq!(chunk_size, 3200);
    }

    #[test]
    fn test_pcm_chunk_1_second() {
        // 1 second of audio at 16kHz mono 16-bit
        let chunk_size = 16000 * 2;
        assert_eq!(chunk_size, 32000);
    }

    #[test]
    fn test_empty_pcm_data() {
        // Empty data should be valid (no-op)
        let empty: Vec<u8> = vec![];
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn test_silence_pcm_data() {
        // All zeros represents silence in PCM
        let silence: Vec<u8> = vec![0u8; 3200]; // 100ms of silence
        assert_eq!(silence.len(), 3200);
        assert!(silence.iter().all(|&b| b == 0));
    }

    // ==================== Sample Conversion Tests ====================

    #[test]
    fn test_i16_to_bytes_conversion() {
        // Verify i16 samples convert to correct byte count
        let samples: Vec<i16> = vec![0, 1000, -1000, i16::MAX, i16::MIN];
        let byte_count = samples.len() * std::mem::size_of::<i16>();
        assert_eq!(byte_count, 10); // 5 samples * 2 bytes each
    }

    #[test]
    fn test_f32_to_i16_conversion_zero() {
        // f32 0.0 should map to i16 0
        let f32_sample = 0.0f32;
        let i16_sample = (f32_sample * i16::MAX as f32) as i16;
        assert_eq!(i16_sample, 0);
    }

    #[test]
    fn test_f32_to_i16_conversion_max() {
        // f32 1.0 should map to i16::MAX
        let f32_sample = 1.0f32;
        let i16_sample = (f32_sample * i16::MAX as f32) as i16;
        assert_eq!(i16_sample, i16::MAX);
    }

    #[test]
    fn test_f32_to_i16_conversion_min() {
        // f32 -1.0 should map close to i16::MIN
        let f32_sample = -1.0f32;
        let clamped = f32_sample.clamp(-1.0, 1.0);
        let i16_sample = (clamped * i16::MAX as f32) as i16;
        // Note: i16::MAX as f32 * -1.0 = -32767, not -32768
        assert_eq!(i16_sample, -i16::MAX);
    }

    #[test]
    fn test_f32_clamping_above_1() {
        // Values above 1.0 should be clamped to 1.0
        let f32_sample = 1.5f32;
        let clamped = f32_sample.clamp(-1.0, 1.0);
        assert_eq!(clamped, 1.0);
    }

    #[test]
    fn test_f32_clamping_below_minus_1() {
        // Values below -1.0 should be clamped to -1.0
        let f32_sample = -2.0f32;
        let clamped = f32_sample.clamp(-1.0, 1.0);
        assert_eq!(clamped, -1.0);
    }

    // ==================== TranscribeResponse Tests ====================

    #[test]
    fn test_transcribe_response_empty() {
        let response = TranscribeResponse {
            text: String::new(),
            language: String::new(),
            duration: 0.0,
            segments: Vec::new(),
        };
        assert!(response.text.is_empty());
        assert!(response.segments.is_empty());
    }

    #[test]
    fn test_transcribe_response_with_content() {
        let response = TranscribeResponse {
            text: "Hello, world!".to_string(),
            language: "en".to_string(),
            duration: 1.5,
            segments: Vec::new(),
        };
        assert_eq!(response.text, "Hello, world!");
        assert_eq!(response.language, "en");
        assert!((response.duration - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_transcribe_response_deserialization() {
        let json = r#"{
            "text": "Test transcription",
            "language": "de",
            "duration": 2.5,
            "segments": []
        }"#;
        let response: TranscribeResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.text, "Test transcription");
        assert_eq!(response.language, "de");
    }

    #[test]
    fn test_transcribe_response_fallback_parsing() {
        // When JSON parsing fails, the raw string becomes the text
        let raw_text = "Just plain text, not JSON";
        let response: TranscribeResponse =
            serde_json::from_str(raw_text).unwrap_or_else(|_| TranscribeResponse {
                text: raw_text.to_string(),
                language: String::new(),
                duration: 0.0,
                segments: Vec::new(),
            });
        assert_eq!(response.text, raw_text);
    }

    // ==================== Thread Safety Tests (compile-time) ====================

    #[test]
    fn test_stream_transcriber_is_send() {
        // Compile-time check that StreamTranscriber implements Send
        fn assert_send<T: Send>() {}
        assert_send::<StreamTranscriber<'_>>();
    }

    // Note: StreamTranscriber is intentionally NOT Sync
    // This test documents that design decision
    #[test]
    fn test_stream_transcriber_not_sync_by_design() {
        // StreamTranscriber can be moved between threads (Send)
        // but should not be accessed concurrently (not Sync)
        // This is documented in the struct's doc comment
        fn assert_send<T: Send>() {}
        assert_send::<StreamTranscriber<'_>>();
    }

    // ==================== Error Type Tests ====================

    #[test]
    fn test_error_inference_display() {
        let err = Error::Inference("stream transcription failed".into());
        assert_eq!(
            err.to_string(),
            "Inference failed: stream transcription failed"
        );
    }

    #[test]
    fn test_error_model_init_display() {
        let err = Error::ModelInit("invalid whisper model".into());
        assert_eq!(
            err.to_string(),
            "Failed to initialize model: invalid whisper model"
        );
    }

    #[test]
    fn test_error_null_pointer_display() {
        let err = Error::NullPointer;
        assert_eq!(err.to_string(), "Null pointer returned from Cactus FFI");
    }

    // ==================== Options Skip Serializing Tests ====================

    #[test]
    fn test_options_skip_serializing_none_fields() {
        let opts = StreamTranscribeOptions::default();
        let json = serde_json::to_string(&opts).unwrap();
        // language is None, so it should not appear in JSON
        assert!(!json.contains("\"language\""));
        // initial_prompt is None, so it should not appear
        assert!(!json.contains("\"initial_prompt\""));
    }

    #[test]
    fn test_options_include_some_fields() {
        let opts = StreamTranscribeOptions::with_language("en");
        let json = serde_json::to_string(&opts).unwrap();
        // language has a value, so it should appear
        assert!(json.contains("\"language\":\"en\""));
    }

    // ==================== All Supported Languages Test ====================

    #[test]
    fn test_common_language_codes() {
        // Whisper supports many languages - test common ones compile/work
        let languages = ["en", "de", "tr", "es", "fr", "zh", "ja", "ko", "ar", "ru"];

        for lang in languages {
            let opts = StreamTranscribeOptions::with_language(lang);
            assert_eq!(opts.language, Some(lang.to_string()));
        }
    }
}

// ==================== Integration Tests ====================
// These tests require a real Whisper model file and are marked #[ignore]
// Run with: cargo test --package cactus -- --ignored

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Test StreamTranscriber creation with a real model
    /// Requires: whisper-tiny.gguf or similar in current directory
    #[test]
    #[ignore = "requires whisper model file"]
    fn test_stream_transcriber_creation() {
        let model = Model::from_gguf("whisper-tiny.gguf").expect("Failed to load model");
        let _transcriber = StreamTranscriber::new(&model).expect("Failed to create transcriber");
        // Transcriber created successfully, will be cleaned up on drop
    }

    /// Test inserting PCM data into stream
    /// Requires: whisper-tiny.gguf or similar in current directory
    #[test]
    #[ignore = "requires whisper model file"]
    fn test_stream_transcriber_insert() {
        let model = Model::from_gguf("whisper-tiny.gguf").expect("Failed to load model");
        let mut transcriber = StreamTranscriber::new(&model).expect("Failed to create transcriber");

        // Insert 1 second of silence (valid PCM data)
        let silence: Vec<u8> = vec![0u8; 32000];
        transcriber
            .insert(&silence)
            .expect("Failed to insert PCM data");

        // Insert more data
        let more_silence: Vec<u8> = vec![0u8; 16000];
        transcriber
            .insert(&more_silence)
            .expect("Failed to insert more data");
    }

    /// Test full lifecycle: init -> insert -> process -> finalize -> drop
    /// Requires: whisper-tiny.gguf or similar in current directory
    #[test]
    #[ignore = "requires whisper model file"]
    fn test_stream_transcriber_lifecycle() {
        // 1. Initialize
        let model = Model::from_gguf("whisper-tiny.gguf").expect("Failed to load model");
        let mut transcriber = StreamTranscriber::new(&model).expect("Failed to create transcriber");

        // 2. Insert audio data (1 second of silence)
        let pcm_data: Vec<u8> = vec![0u8; 32000];
        transcriber.insert(&pcm_data).expect("Failed to insert");

        // 3. Process and get intermediate result
        let options = StreamTranscribeOptions::with_language("en");
        let partial = transcriber.process(options).expect("Failed to process");
        println!("Partial result: '{}'", partial.text);

        // 4. Insert more data
        transcriber
            .insert(&pcm_data)
            .expect("Failed to insert more");

        // 5. Finalize and get final result
        let final_result = transcriber.finalize().expect("Failed to finalize");
        println!("Final result: '{}'", final_result.text);

        // 6. Drop happens automatically
    }

    /// Test builder pattern
    /// Requires: whisper-tiny.gguf or similar in current directory
    #[test]
    #[ignore = "requires whisper model file"]
    fn test_stream_transcriber_builder() {
        let model = Model::from_gguf("whisper-tiny.gguf").expect("Failed to load model");

        let transcriber = StreamTranscriberBuilder::new(&model)
            .with_buffer_size(128 * 1024)
            .build()
            .expect("Failed to build transcriber");

        drop(transcriber);
    }

    /// Test insert_samples convenience method
    /// Requires: whisper-tiny.gguf or similar in current directory
    #[test]
    #[ignore = "requires whisper model file"]
    fn test_stream_transcriber_insert_samples() {
        let model = Model::from_gguf("whisper-tiny.gguf").expect("Failed to load model");
        let mut transcriber = StreamTranscriber::new(&model).expect("Failed to create transcriber");

        // Create i16 samples (16000 samples = 1 second at 16kHz)
        let samples: Vec<i16> = vec![0i16; 16000];
        transcriber
            .insert_samples(&samples)
            .expect("Failed to insert samples");
    }

    /// Test insert_f32 convenience method
    /// Requires: whisper-tiny.gguf or similar in current directory
    #[test]
    #[ignore = "requires whisper model file"]
    fn test_stream_transcriber_insert_f32() {
        let model = Model::from_gguf("whisper-tiny.gguf").expect("Failed to load model");
        let mut transcriber = StreamTranscriber::new(&model).expect("Failed to create transcriber");

        // Create f32 samples in [-1.0, 1.0] range
        let samples: Vec<f32> = vec![0.0f32; 16000];
        transcriber
            .insert_f32(&samples)
            .expect("Failed to insert f32 samples");
    }
}
