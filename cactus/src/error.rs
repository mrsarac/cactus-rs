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

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_error_display_model_load() {
        let err = Error::ModelLoad("file not found".into());
        assert_eq!(err.to_string(), "Failed to load model: file not found");
    }

    #[test]
    fn test_error_display_model_init() {
        let err = Error::ModelInit("invalid format".into());
        assert_eq!(
            err.to_string(),
            "Failed to initialize model: invalid format"
        );
    }

    #[test]
    fn test_error_display_inference() {
        let err = Error::Inference("out of memory".into());
        assert_eq!(err.to_string(), "Inference failed: out of memory");
    }

    #[test]
    fn test_error_display_invalid_input() {
        let err = Error::InvalidInput("empty message".into());
        assert_eq!(err.to_string(), "Invalid input: empty message");
    }

    #[test]
    fn test_error_display_null_pointer() {
        let err = Error::NullPointer;
        assert_eq!(err.to_string(), "Null pointer returned from Cactus FFI");
    }

    #[test]
    fn test_error_display_stopped() {
        let err = Error::Stopped;
        assert_eq!(err.to_string(), "Generation was stopped");
    }

    #[test]
    fn test_error_display_unsupported() {
        let err = Error::Unsupported("GPU not available".into());
        assert_eq!(err.to_string(), "Feature not supported: GPU not available");
    }

    #[test]
    fn test_error_display_index() {
        let err = Error::Index("index corrupted".into());
        assert_eq!(err.to_string(), "Index error: index corrupted");
    }

    #[test]
    fn test_error_display_cactus() {
        let err = Error::Cactus("internal error".into());
        assert_eq!(err.to_string(), "Cactus error: internal error");
    }

    #[test]
    fn test_error_from_json() {
        let json_err = serde_json::from_str::<i32>("not a number").unwrap_err();
        let err: Error = json_err.into();
        assert!(matches!(err, Error::Json(_)));
    }

    #[test]
    fn test_error_from_nul() {
        let nul_err = CString::new("hello\0world").unwrap_err();
        let err: Error = nul_err.into();
        assert!(matches!(err, Error::NulError(_)));
    }

    #[test]
    fn test_result_type_ok() {
        let result: Result<i32> = Ok(42);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_result_type_err() {
        let result: Result<i32> = Err(Error::NullPointer);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_debug_format() {
        let err = Error::ModelLoad("test".into());
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("ModelLoad"));
    }
}
