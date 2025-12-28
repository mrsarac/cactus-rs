//! Vector search example using Cactus Rust bindings
//!
//! Usage: cargo run --example vector_search -- path/to/model
//!
//! This example:
//! 1. Loads a model for generating embeddings
//! 2. Creates a vector index
//! 3. Adds sample documents with their embeddings
//! 4. Performs semantic search

use cactus::{Document, Model, QueryOptions, VectorIndex};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let model_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("vendor/cactus/weights/lfm2-1.2b");

    println!("🌵 Cactus Rust Bindings - Vector Search Example");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Load model for embeddings
    println!("📦 Loading model: {}", model_path);
    let model = Model::from_gguf(model_path)?;
    println!("✅ Model loaded!");
    println!();

    // Sample knowledge base
    let knowledge_base = [
        (1, "Rust is a systems programming language focused on safety and performance."),
        (2, "Python is great for data science and machine learning applications."),
        (3, "JavaScript runs in web browsers and powers interactive websites."),
        (4, "Go is designed for building scalable network services and cloud infrastructure."),
        (5, "The quick brown fox jumps over the lazy dog."),
        (6, "Machine learning models can be deployed on mobile devices for edge inference."),
        (7, "Vector databases enable semantic search using embedding similarity."),
        (8, "Neural networks are inspired by biological brain structures."),
    ];

    // Generate embeddings
    println!("🧮 Generating embeddings for {} documents...", knowledge_base.len());
    let mut documents = Vec::new();
    let mut embedding_dim = 0;

    for (id, content) in &knowledge_base {
        let embedding = model.embed(content, true)?;
        embedding_dim = embedding.dimension;
        documents.push(
            Document::new(*id, *content)
                .with_embedding(embedding.vector)
                .with_metadata(format!(r#"{{"id": {}}}"#, id)),
        );
        print!(".");
    }
    println!(" Done!");
    println!("  → Embedding dimension: {}", embedding_dim);
    println!();

    // Create vector index
    let index_dir = "/tmp/cactus-search-example";

    // Ensure directory exists
    std::fs::create_dir_all(index_dir)?;

    println!("📁 Creating index at: {}", index_dir);
    let index = VectorIndex::new(index_dir, embedding_dim)?;

    // Add documents
    println!("📥 Adding {} documents to index...", documents.len());
    index.add(&documents)?;
    println!("✅ Documents indexed!");
    println!();

    // Search queries
    let queries = [
        "What programming language is best for web development?",
        "How can I run AI models on smartphones?",
        "Tell me about memory-safe systems programming",
    ];

    println!("🔍 Performing semantic searches:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    for query in &queries {
        println!();
        println!("📝 Query: \"{}\"", query);
        println!();

        // Generate query embedding
        let query_embedding = model.embed(query, true)?;

        // Search
        let results = index.query(
            &query_embedding.vector,
            QueryOptions {
                top_k: Some(3),
                ..Default::default()
            },
        )?;

        // Display results
        for (rank, result) in results.iter().enumerate() {
            let doc = knowledge_base
                .iter()
                .find(|(id, _)| *id == result.id)
                .map(|(_, content)| *content)
                .unwrap_or("Unknown");

            println!(
                "  {}. [Score: {:.3}] {}",
                rank + 1,
                result.score,
                doc
            );
        }
    }

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("✅ Vector search complete!");

    // Cleanup
    std::fs::remove_dir_all(index_dir).ok();

    Ok(())
}
