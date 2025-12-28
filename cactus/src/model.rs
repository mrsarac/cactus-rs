//! Model management and inference

use crate::error::{Error, Result};
use crate::types::{Embedding, GenerateOptions, GenerateResponse, Message, ModelConfig, TranscribeOptions, TranscribeResponse};
use std::ffi::{CStr, CString};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Default response buffer size (64KB)
const DEFAULT_BUFFER_SIZE: usize = 64 * 1024;

/// Default embedding buffer size (16K floats = 64KB, supports up to 16K dimensions)
const DEFAULT_EMBEDDING_SIZE: usize = 16 * 1024;

/// A Cactus model for on-device AI inference
///
/// This is the main entry point for using Cactus. It wraps the underlying
/// C model handle and provides safe Rust methods for inference.
///
/// # Example
///
/// ```rust,ignore
/// use cactus::{Model, Message, GenerateOptions};
///
/// let model = Model::from_gguf("model.gguf")?;
/// let response = model.complete(
///     &[Message::user("Hello!")],
///     GenerateOptions::default()
/// )?;
/// println!("{}", response.content);
/// ```
pub struct Model {
    /// Raw handle to the Cactus model
    handle: NonNull<std::ffi::c_void>,
    /// Configuration used to create this model
    config: ModelConfig,
    /// Flag to stop generation
    stop_flag: Arc<AtomicBool>,
}

// Safety: The Cactus model handle is internally synchronized with mutexes
unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    /// Load a model from a GGUF file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the GGUF model file
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let model = Model::from_gguf("gemma-2b-it-q4.gguf")?;
    /// ```
    pub fn from_gguf(path: impl Into<String>) -> Result<Self> {
        let config = ModelConfig::new(path);
        Self::from_config(config)
    }

    /// Load a model with custom configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = ModelConfig::new("model.gguf")
    ///     .with_context_size(4096)
    ///     .with_gpu(true);
    /// let model = Model::from_config(config)?;
    /// ```
    pub fn from_config(config: ModelConfig) -> Result<Self> {
        let path = CString::new(config.path.as_str())?;

        let handle = unsafe {
            cactus_sys::cactus_init(
                path.as_ptr(),
                config.context_size as usize,
                std::ptr::null(), // corpus_dir (optional)
            )
        };

        let handle = NonNull::new(handle).ok_or_else(|| {
            Error::last_cactus_error()
                .map(Error::ModelInit)
                .unwrap_or(Error::NullPointer)
        })?;

        Ok(Self {
            handle,
            config,
            stop_flag: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Generate a response for the given messages
    ///
    /// # Arguments
    ///
    /// * `messages` - Conversation history
    /// * `options` - Generation options
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let messages = vec![
    ///     Message::system("You are helpful."),
    ///     Message::user("What is 2+2?"),
    /// ];
    /// let response = model.complete(&messages, GenerateOptions::default())?;
    /// ```
    pub fn complete(&self, messages: &[Message], options: GenerateOptions) -> Result<GenerateResponse> {
        let messages_json = serde_json::to_string(messages)?;
        let options_json = serde_json::to_string(&options)?;

        let messages_c = CString::new(messages_json)?;
        let options_c = CString::new(options_json)?;

        self.stop_flag.store(false, Ordering::SeqCst);

        // Allocate response buffer
        let mut response_buffer: Vec<u8> = vec![0u8; DEFAULT_BUFFER_SIZE];

        let result = unsafe {
            cactus_sys::cactus_complete(
                self.handle.as_ptr(),
                messages_c.as_ptr(),
                response_buffer.as_mut_ptr() as *mut i8,
                response_buffer.len(),
                options_c.as_ptr(),
                std::ptr::null(), // tools_json
                None,             // callback
                std::ptr::null_mut(), // user_data
            )
        };

        // result > 0 means bytes written (success), result < 0 means error
        if result < 0 {
            return Err(Error::last_cactus_error()
                .map(Error::Inference)
                .unwrap_or_else(|| Error::Inference(format!("cactus_complete failed with code {}", result))));
        }

        // Parse response from buffer
        let response_str = unsafe {
            let c_str = CStr::from_ptr(response_buffer.as_ptr() as *const i8);
            c_str.to_string_lossy().into_owned()
        };

        // Try to parse as JSON, otherwise return raw content
        let response: GenerateResponse = serde_json::from_str(&response_str)
            .unwrap_or_else(|_| GenerateResponse {
                content: response_str,
                prompt_tokens: 0,
                completion_tokens: 0,
                time_to_first_token_ms: 0.0,
                tokens_per_second: 0.0,
                stopped: false,
            });

        Ok(response)
    }

    /// Generate a response with streaming callback
    ///
    /// The callback is called for each generated token. Return `true` to
    /// continue generation, or `false` to stop early.
    ///
    /// # Arguments
    ///
    /// * `messages` - Conversation history
    /// * `options` - Generation options
    /// * `callback` - Called for each token
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// model.complete_streaming(&messages, options, |token| {
    ///     print!("{}", token);
    ///     std::io::stdout().flush().ok();
    ///     true // continue
    /// })?;
    /// ```
    pub fn complete_streaming<F>(
        &self,
        messages: &[Message],
        options: GenerateOptions,
        mut callback: F,
    ) -> Result<GenerateResponse>
    where
        F: FnMut(&str, u32) -> bool,
    {
        let messages_json = serde_json::to_string(messages)?;
        let options_json = serde_json::to_string(&options)?;

        let messages_c = CString::new(messages_json)?;
        let options_c = CString::new(options_json)?;

        self.stop_flag.store(false, Ordering::SeqCst);

        // Create callback wrapper
        struct CallbackData<'a, F: FnMut(&str, u32) -> bool> {
            callback: &'a mut F,
            should_stop: bool,
        }

        let mut callback_data = CallbackData {
            callback: &mut callback,
            should_stop: false,
        };

        // Trampoline function
        unsafe extern "C" fn trampoline<F: FnMut(&str, u32) -> bool>(
            token: *const std::ffi::c_char,
            token_id: u32,
            user_data: *mut std::ffi::c_void,
        ) {
            if token.is_null() || user_data.is_null() {
                return;
            }
            let data = &mut *(user_data as *mut CallbackData<F>);
            let token_str = CStr::from_ptr(token).to_string_lossy();
            let should_continue = (data.callback)(&token_str, token_id);
            if !should_continue {
                data.should_stop = true;
            }
        }

        // Allocate response buffer
        let mut response_buffer: Vec<u8> = vec![0u8; DEFAULT_BUFFER_SIZE];

        let result = unsafe {
            cactus_sys::cactus_complete(
                self.handle.as_ptr(),
                messages_c.as_ptr(),
                response_buffer.as_mut_ptr() as *mut i8,
                response_buffer.len(),
                options_c.as_ptr(),
                std::ptr::null(),
                Some(trampoline::<F>),
                &mut callback_data as *mut CallbackData<F> as *mut std::ffi::c_void,
            )
        };

        // result > 0 means bytes written (success), result < 0 means error
        if result < 0 && !callback_data.should_stop {
            return Err(Error::last_cactus_error()
                .map(Error::Inference)
                .unwrap_or_else(|| Error::Inference(format!("cactus_complete failed with code {}", result))));
        }

        let response_str = unsafe {
            let c_str = CStr::from_ptr(response_buffer.as_ptr() as *const i8);
            c_str.to_string_lossy().into_owned()
        };

        let response: GenerateResponse = serde_json::from_str(&response_str)
            .unwrap_or_else(|_| GenerateResponse {
                content: response_str,
                prompt_tokens: 0,
                completion_tokens: 0,
                time_to_first_token_ms: 0.0,
                tokens_per_second: 0.0,
                stopped: callback_data.should_stop,
            });

        Ok(response)
    }

    /// Generate embeddings for text
    ///
    /// # Arguments
    ///
    /// * `text` - Text to embed
    /// * `normalize` - Whether to L2-normalize the embedding vector
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let embedding = model.embed("Hello, world!", true)?;
    /// println!("Dimension: {}", embedding.dimension);
    /// println!("Vector: {:?}", &embedding.vector[..5]);
    /// ```
    pub fn embed(&self, text: &str, normalize: bool) -> Result<Embedding> {
        let text_c = CString::new(text)?;

        let mut embeddings_buffer: Vec<f32> = vec![0.0; DEFAULT_EMBEDDING_SIZE];
        let mut embedding_dim: usize = 0;

        let result = unsafe {
            cactus_sys::cactus_embed(
                self.handle.as_ptr(),
                text_c.as_ptr(),
                embeddings_buffer.as_mut_ptr(),
                embeddings_buffer.len(),
                &mut embedding_dim,
                normalize,
            )
        };

        if result < 0 {
            return Err(Error::last_cactus_error()
                .map(Error::Inference)
                .unwrap_or_else(|| Error::Inference(format!("cactus_embed failed with code {}", result))));
        }

        // Truncate to actual dimension
        embeddings_buffer.truncate(embedding_dim);

        Ok(Embedding {
            vector: embeddings_buffer,
            dimension: embedding_dim,
        })
    }

    /// Generate embeddings for an image file
    ///
    /// # Arguments
    ///
    /// * `image_path` - Path to the image file
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let embedding = model.embed_image("photo.jpg")?;
    /// ```
    pub fn embed_image(&self, image_path: &str) -> Result<Embedding> {
        let path_c = CString::new(image_path)?;

        let mut embeddings_buffer: Vec<f32> = vec![0.0; DEFAULT_EMBEDDING_SIZE];
        let mut embedding_dim: usize = 0;

        let result = unsafe {
            cactus_sys::cactus_image_embed(
                self.handle.as_ptr(),
                path_c.as_ptr(),
                embeddings_buffer.as_mut_ptr(),
                embeddings_buffer.len(),
                &mut embedding_dim,
            )
        };

        if result < 0 {
            return Err(Error::last_cactus_error()
                .map(Error::Inference)
                .unwrap_or_else(|| Error::Inference(format!("cactus_image_embed failed with code {}", result))));
        }

        embeddings_buffer.truncate(embedding_dim);

        Ok(Embedding {
            vector: embeddings_buffer,
            dimension: embedding_dim,
        })
    }

    /// Generate embeddings for an audio file
    ///
    /// # Arguments
    ///
    /// * `audio_path` - Path to the audio file
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let embedding = model.embed_audio("speech.wav")?;
    /// ```
    pub fn embed_audio(&self, audio_path: &str) -> Result<Embedding> {
        let path_c = CString::new(audio_path)?;

        let mut embeddings_buffer: Vec<f32> = vec![0.0; DEFAULT_EMBEDDING_SIZE];
        let mut embedding_dim: usize = 0;

        let result = unsafe {
            cactus_sys::cactus_audio_embed(
                self.handle.as_ptr(),
                path_c.as_ptr(),
                embeddings_buffer.as_mut_ptr(),
                embeddings_buffer.len(),
                &mut embedding_dim,
            )
        };

        if result < 0 {
            return Err(Error::last_cactus_error()
                .map(Error::Inference)
                .unwrap_or_else(|| Error::Inference(format!("cactus_audio_embed failed with code {}", result))));
        }

        embeddings_buffer.truncate(embedding_dim);

        Ok(Embedding {
            vector: embeddings_buffer,
            dimension: embedding_dim,
        })
    }

    /// Transcribe audio from a file
    ///
    /// Converts speech to text using the loaded model.
    /// Requires a Whisper-compatible model.
    ///
    /// # Arguments
    ///
    /// * `audio_path` - Path to audio file (WAV, MP3, etc.)
    /// * `options` - Transcription options
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let result = model.transcribe("speech.wav", TranscribeOptions::default())?;
    /// println!("Transcription: {}", result.text);
    /// ```
    pub fn transcribe(
        &self,
        audio_path: &str,
        options: TranscribeOptions,
    ) -> Result<TranscribeResponse> {
        let audio_path_c = CString::new(audio_path)?;
        let prompt_c = CString::new(options.initial_prompt.as_deref().unwrap_or(""))?;
        let options_json = serde_json::to_string(&options)?;
        let options_c = CString::new(options_json)?;

        self.stop_flag.store(false, Ordering::SeqCst);

        let mut response_buffer: Vec<u8> = vec![0u8; DEFAULT_BUFFER_SIZE];

        let result = unsafe {
            cactus_sys::cactus_transcribe(
                self.handle.as_ptr(),
                audio_path_c.as_ptr(),
                prompt_c.as_ptr(),
                response_buffer.as_mut_ptr() as *mut i8,
                response_buffer.len(),
                options_c.as_ptr(),
                None,             // callback
                std::ptr::null_mut(), // user_data
                std::ptr::null(),  // pcm_buffer (not using raw PCM)
                0,                 // pcm_buffer_size
            )
        };

        if result < 0 {
            return Err(Error::last_cactus_error()
                .map(Error::Inference)
                .unwrap_or_else(|| Error::Inference(format!("cactus_transcribe failed with code {}", result))));
        }

        let response_str = unsafe {
            let c_str = CStr::from_ptr(response_buffer.as_ptr() as *const i8);
            c_str.to_string_lossy().into_owned()
        };

        // Try to parse as JSON, otherwise return raw text
        let response: TranscribeResponse = serde_json::from_str(&response_str)
            .unwrap_or_else(|_| TranscribeResponse {
                text: response_str,
                language: String::new(),
                duration: 0.0,
                segments: Vec::new(),
            });

        Ok(response)
    }

    /// Transcribe audio with streaming callback
    ///
    /// The callback is called for each transcribed segment.
    /// Return `true` to continue, `false` to stop.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// model.transcribe_streaming("speech.wav", options, |text, _| {
    ///     print!("{}", text);
    ///     std::io::stdout().flush().ok();
    ///     true
    /// })?;
    /// ```
    pub fn transcribe_streaming<F>(
        &self,
        audio_path: &str,
        options: TranscribeOptions,
        mut callback: F,
    ) -> Result<TranscribeResponse>
    where
        F: FnMut(&str, u32) -> bool,
    {
        let audio_path_c = CString::new(audio_path)?;
        let prompt_c = CString::new(options.initial_prompt.as_deref().unwrap_or(""))?;
        let options_json = serde_json::to_string(&options)?;
        let options_c = CString::new(options_json)?;

        self.stop_flag.store(false, Ordering::SeqCst);

        // Create callback wrapper
        struct CallbackData<'a, F: FnMut(&str, u32) -> bool> {
            callback: &'a mut F,
            should_stop: bool,
        }

        let mut callback_data = CallbackData {
            callback: &mut callback,
            should_stop: false,
        };

        unsafe extern "C" fn trampoline<F: FnMut(&str, u32) -> bool>(
            token: *const std::ffi::c_char,
            token_id: u32,
            user_data: *mut std::ffi::c_void,
        ) {
            if token.is_null() || user_data.is_null() {
                return;
            }
            let data = &mut *(user_data as *mut CallbackData<F>);
            let token_str = CStr::from_ptr(token).to_string_lossy();
            let should_continue = (data.callback)(&token_str, token_id);
            if !should_continue {
                data.should_stop = true;
            }
        }

        let mut response_buffer: Vec<u8> = vec![0u8; DEFAULT_BUFFER_SIZE];

        let result = unsafe {
            cactus_sys::cactus_transcribe(
                self.handle.as_ptr(),
                audio_path_c.as_ptr(),
                prompt_c.as_ptr(),
                response_buffer.as_mut_ptr() as *mut i8,
                response_buffer.len(),
                options_c.as_ptr(),
                Some(trampoline::<F>),
                &mut callback_data as *mut CallbackData<F> as *mut std::ffi::c_void,
                std::ptr::null(),
                0,
            )
        };

        if result < 0 && !callback_data.should_stop {
            return Err(Error::last_cactus_error()
                .map(Error::Inference)
                .unwrap_or_else(|| Error::Inference(format!("cactus_transcribe failed with code {}", result))));
        }

        let response_str = unsafe {
            let c_str = CStr::from_ptr(response_buffer.as_ptr() as *const i8);
            c_str.to_string_lossy().into_owned()
        };

        let response: TranscribeResponse = serde_json::from_str(&response_str)
            .unwrap_or_else(|_| TranscribeResponse {
                text: response_str,
                language: String::new(),
                duration: 0.0,
                segments: Vec::new(),
            });

        Ok(response)
    }

    /// Transcribe raw PCM audio data
    ///
    /// # Arguments
    ///
    /// * `pcm_data` - Raw PCM audio data (16-bit, mono, 16kHz)
    /// * `options` - Transcription options
    pub fn transcribe_pcm(
        &self,
        pcm_data: &[u8],
        options: TranscribeOptions,
    ) -> Result<TranscribeResponse> {
        let prompt_c = CString::new(options.initial_prompt.as_deref().unwrap_or(""))?;
        let options_json = serde_json::to_string(&options)?;
        let options_c = CString::new(options_json)?;

        self.stop_flag.store(false, Ordering::SeqCst);

        let mut response_buffer: Vec<u8> = vec![0u8; DEFAULT_BUFFER_SIZE];

        let result = unsafe {
            cactus_sys::cactus_transcribe(
                self.handle.as_ptr(),
                std::ptr::null(),  // no file path
                prompt_c.as_ptr(),
                response_buffer.as_mut_ptr() as *mut i8,
                response_buffer.len(),
                options_c.as_ptr(),
                None,
                std::ptr::null_mut(),
                pcm_data.as_ptr(),
                pcm_data.len(),
            )
        };

        if result < 0 {
            return Err(Error::last_cactus_error()
                .map(Error::Inference)
                .unwrap_or_else(|| Error::Inference(format!("cactus_transcribe failed with code {}", result))));
        }

        let response_str = unsafe {
            let c_str = CStr::from_ptr(response_buffer.as_ptr() as *const i8);
            c_str.to_string_lossy().into_owned()
        };

        let response: TranscribeResponse = serde_json::from_str(&response_str)
            .unwrap_or_else(|_| TranscribeResponse {
                text: response_str,
                language: String::new(),
                duration: 0.0,
                segments: Vec::new(),
            });

        Ok(response)
    }

    /// Stop any ongoing generation
    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::SeqCst);
        unsafe {
            cactus_sys::cactus_stop(self.handle.as_ptr());
        }
    }

    /// Reset the model state (clear KV cache)
    pub fn reset(&self) {
        unsafe {
            cactus_sys::cactus_reset(self.handle.as_ptr());
        }
    }

    /// Get the model configuration
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe {
            cactus_sys::cactus_destroy(self.handle.as_ptr());
        }
    }
}
