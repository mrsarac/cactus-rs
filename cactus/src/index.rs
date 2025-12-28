//! Vector index for semantic search

use crate::error::{Error, Result};
use crate::types::{Document, QueryOptions, SearchResult};
use std::ffi::CString;
use std::ptr::NonNull;

/// A vector index for semantic search
///
/// Stores document embeddings and enables fast similarity search.
///
/// # Example
///
/// ```rust,ignore
/// use cactus::{VectorIndex, Document, QueryOptions};
///
/// // Create index
/// let index = VectorIndex::new("./my_index", 2048)?;
///
/// // Add documents with embeddings
/// let docs = vec![
///     Document::new(1, "Hello world").with_embedding(embedding1),
///     Document::new(2, "Goodbye world").with_embedding(embedding2),
/// ];
/// index.add(&docs)?;
///
/// // Search
/// let results = index.query(&query_embedding, QueryOptions::default())?;
/// ```
pub struct VectorIndex {
    handle: NonNull<std::ffi::c_void>,
    embedding_dim: usize,
}

// Safety: The index handle is internally synchronized
unsafe impl Send for VectorIndex {}
unsafe impl Sync for VectorIndex {}

impl VectorIndex {
    /// Create or open a vector index
    ///
    /// # Arguments
    ///
    /// * `index_dir` - Directory to store the index
    /// * `embedding_dim` - Dimensionality of embeddings (e.g., 2048)
    pub fn new(index_dir: impl Into<String>, embedding_dim: usize) -> Result<Self> {
        let dir = index_dir.into();
        let dir_c = CString::new(dir.as_str())?;

        let handle = unsafe {
            cactus_sys::cactus_index_init(dir_c.as_ptr(), embedding_dim)
        };

        let handle = NonNull::new(handle).ok_or_else(|| {
            Error::last_cactus_error()
                .map(Error::Index)
                .unwrap_or(Error::NullPointer)
        })?;

        Ok(Self {
            handle,
            embedding_dim,
        })
    }

    /// Add documents to the index
    ///
    /// Documents must have embeddings set via `with_embedding()`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let docs = vec![
    ///     Document::new(1, "First doc").with_embedding(emb1),
    ///     Document::new(2, "Second doc").with_embedding(emb2),
    /// ];
    /// index.add(&docs)?;
    /// ```
    pub fn add(&self, documents: &[Document]) -> Result<()> {
        if documents.is_empty() {
            return Ok(());
        }

        // Validate all documents have embeddings
        for doc in documents {
            if doc.embedding.is_none() {
                return Err(Error::Index(format!(
                    "Document {} has no embedding",
                    doc.id
                )));
            }
        }

        let count = documents.len();

        // Prepare IDs
        let ids: Vec<i32> = documents.iter().map(|d| d.id).collect();

        // Prepare document strings
        let doc_cstrings: Vec<CString> = documents
            .iter()
            .map(|d| CString::new(d.content.as_str()))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut doc_ptrs: Vec<*const i8> = doc_cstrings.iter().map(|s| s.as_ptr()).collect();

        // Prepare metadata strings
        let meta_cstrings: Vec<CString> = documents
            .iter()
            .map(|d| CString::new(d.metadata.as_deref().unwrap_or("{}")))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut meta_ptrs: Vec<*const i8> = meta_cstrings.iter().map(|s| s.as_ptr()).collect();

        // Prepare embeddings
        let embeddings: Vec<&Vec<f32>> = documents
            .iter()
            .map(|d| d.embedding.as_ref().unwrap())
            .collect();
        let mut emb_ptrs: Vec<*const f32> = embeddings.iter().map(|e| e.as_ptr()).collect();

        let result = unsafe {
            cactus_sys::cactus_index_add(
                self.handle.as_ptr(),
                ids.as_ptr(),
                doc_ptrs.as_mut_ptr(),
                meta_ptrs.as_mut_ptr(),
                emb_ptrs.as_mut_ptr(),
                count,
                self.embedding_dim,
            )
        };

        if result < 0 {
            return Err(Error::last_cactus_error()
                .map(Error::Index)
                .unwrap_or_else(|| Error::Index(format!("cactus_index_add failed: {}", result))));
        }

        Ok(())
    }

    /// Delete documents by ID
    pub fn delete(&self, ids: &[i32]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }

        let result = unsafe {
            cactus_sys::cactus_index_delete(
                self.handle.as_ptr(),
                ids.as_ptr(),
                ids.len(),
            )
        };

        if result < 0 {
            return Err(Error::last_cactus_error()
                .map(Error::Index)
                .unwrap_or_else(|| Error::Index(format!("cactus_index_delete failed: {}", result))));
        }

        Ok(())
    }

    /// Query the index with an embedding vector
    ///
    /// Returns documents sorted by similarity (highest first).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let results = index.query(&query_embedding, QueryOptions {
    ///     top_k: Some(5),
    ///     ..Default::default()
    /// })?;
    ///
    /// for result in results {
    ///     println!("ID: {}, Score: {:.3}", result.id, result.score);
    /// }
    /// ```
    pub fn query(&self, embedding: &[f32], options: QueryOptions) -> Result<Vec<SearchResult>> {
        let options_json = serde_json::to_string(&options)?;
        let options_c = CString::new(options_json)?;

        let top_k = options.top_k.unwrap_or(10);

        // Prepare embedding pointer
        let emb_ptr = embedding.as_ptr();
        let mut emb_ptrs: Vec<*const f32> = vec![emb_ptr];

        // Allocate result buffers
        let mut id_buffer: Vec<i32> = vec![0; top_k];
        let mut score_buffer: Vec<f32> = vec![0.0; top_k];

        let mut id_ptr = id_buffer.as_mut_ptr();
        let mut score_ptr = score_buffer.as_mut_ptr();

        let mut id_size = top_k;
        let mut score_size = top_k;

        let result = unsafe {
            cactus_sys::cactus_index_query(
                self.handle.as_ptr(),
                emb_ptrs.as_mut_ptr(),
                1, // single query
                self.embedding_dim,
                options_c.as_ptr(),
                &mut id_ptr,
                &mut id_size,
                &mut score_ptr,
                &mut score_size,
            )
        };

        if result < 0 {
            return Err(Error::last_cactus_error()
                .map(Error::Index)
                .unwrap_or_else(|| Error::Index(format!("cactus_index_query failed: {}", result))));
        }

        // Build results
        let num_results = id_size.min(score_size);
        let results: Vec<SearchResult> = (0..num_results)
            .map(|i| SearchResult {
                id: id_buffer[i],
                score: score_buffer[i],
            })
            .filter(|r| r.id != 0 || r.score != 0.0) // filter empty slots
            .collect();

        Ok(results)
    }

    /// Compact the index to optimize storage and search performance
    pub fn compact(&self) -> Result<()> {
        let result = unsafe {
            cactus_sys::cactus_index_compact(self.handle.as_ptr())
        };

        if result < 0 {
            return Err(Error::last_cactus_error()
                .map(Error::Index)
                .unwrap_or_else(|| Error::Index(format!("cactus_index_compact failed: {}", result))));
        }

        Ok(())
    }

    /// Get the embedding dimension of this index
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

impl Drop for VectorIndex {
    fn drop(&mut self) {
        unsafe {
            cactus_sys::cactus_index_destroy(self.handle.as_ptr());
        }
    }
}
