//! Error types for the Cactus library

use std::ffi::NulError;
use thiserror::Error;

/// Result type alias for Cactus operations
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur when using Cactus
#[derive(Error, Debug)]
pub enum Error {
    /// Model file not found or invalid
    #[error("Failed to load model: {0}")]
    ModelLoad(String),

    /// Model initialization failed
    #[error("Failed to initialize model: {0}")]
    ModelInit(String),

    /// Inference error
    #[error("Inference failed: {0}")]
    Inference(String),

    /// Invalid input provided
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Null pointer returned from FFI
    #[error("Null pointer returned from Cactus FFI")]
    NullPointer,

    /// String contains null byte
    #[error("String contains null byte: {0}")]
    NulError(#[from] NulError),

    /// Internal Cactus error
    #[error("Cactus error: {0}")]
    Cactus(String),

    /// Model was stopped
    #[error("Generation was stopped")]
    Stopped,

    /// Feature not supported
    #[error("Feature not supported: {0}")]
    Unsupported(String),

    /// Vector index error
    #[error("Index error: {0}")]
    Index(String),
}

impl Error {
    /// Get the last error from Cactus FFI
    pub(crate) fn last_cactus_error() -> Option<String> {
        unsafe {
            let error_ptr = cactus_sys::cactus_get_last_error();
            if error_ptr.is_null() {
                None
            } else {
                let c_str = std::ffi::CStr::from_ptr(error_ptr);
                let s = c_str.to_string_lossy().into_owned();
                if s.is_empty() {
                    None
                } else {
                    Some(s)
                }
            }
        }
    }
}
