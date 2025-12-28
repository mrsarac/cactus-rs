//! Model management and inference

use crate::error::{Error, Result};
use crate::types::{GenerateOptions, GenerateResponse, Message, ModelConfig};
use std::ffi::{CStr, CString};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Default response buffer size (64KB)
const DEFAULT_BUFFER_SIZE: usize = 64 * 1024;

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
