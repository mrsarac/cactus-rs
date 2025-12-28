//! Text embeddings example using Cactus Rust bindings
//!
//! Usage: cargo run --example embeddings -- path/to/model
//!
//! Note: Requires an embedding-capable model

use cactus::Model;
use std::env;

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let model_path = args.get(1).map(|s| s.as_str()).unwrap_or("models/embedding-model");

    println!("🌵 Cactus Rust Bindings - Embeddings Example");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Loading model: {}", model_path);

    let model = Model::from_gguf(model_path)?;
    println!("✅ Model loaded successfully!");
    println!();

    // Test texts
    let texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn fox leaps above a sleepy canine.",
        "Machine learning is a subset of artificial intelligence.",
        "I love eating pizza on Friday nights.",
    ];

    println!("📝 Generating embeddings for {} texts...", texts.len());
    println!();

    // Generate embeddings
    let mut embeddings = Vec::new();
    for text in &texts {
        let emb = model.embed(text, true)?; // normalized
        println!("  ✓ \"{}...\" → {} dimensions", &text[..30.min(text.len())], emb.dimension);
        embeddings.push(emb);
    }

    println!();
    println!("📊 Similarity Matrix:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Print similarity matrix
    print!("      ");
    for i in 0..texts.len() {
        print!("  T{}   ", i + 1);
    }
    println!();

    for i in 0..texts.len() {
        print!("T{}  ", i + 1);
        for j in 0..texts.len() {
            let sim = cosine_similarity(&embeddings[i].vector, &embeddings[j].vector);
            print!(" {:.3} ", sim);
        }
        println!();
    }

    println!();
    println!("Legend:");
    println!("  T1: \"{}\"", texts[0]);
    println!("  T2: \"{}\"", texts[1]);
    println!("  T3: \"{}\"", texts[2]);
    println!("  T4: \"{}\"", texts[3]);
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("✅ Embeddings complete!");
    println!();
    println!("💡 Note: T1 and T2 should have high similarity (similar meaning)");
    println!("         T3 and T4 should have low similarity (different topics)");

    Ok(())
}
