//! Simple chat example using Cactus Rust bindings
//!
//! Usage: cargo run --example simple_chat -- path/to/model.gguf

use cactus::{GenerateOptions, Message, Model};
use std::env;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get model path from args
    let args: Vec<String> = env::args().collect();
    let model_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("models/gemma-2-2b-it-Q4_K_M.gguf");

    println!("🌵 Cactus Rust Bindings - Simple Chat Example");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Loading model: {}", model_path);

    // Load model
    let model = Model::from_gguf(model_path)?;
    println!("✅ Model loaded successfully!");
    println!();

    // Create conversation
    let messages = vec![
        Message::system("You are a helpful, concise assistant. Keep responses brief."),
        Message::user("What is the meaning of life? Answer in one sentence."),
    ];

    println!("📝 Prompt: {}", messages.last().unwrap().content);
    println!();
    println!("🤖 Response:");

    // Generate with streaming
    let options = GenerateOptions {
        temperature: Some(0.7),
        max_tokens: Some(100),
        ..Default::default()
    };

    model.complete_streaming(&messages, options, |token, _token_id| {
        print!("{}", token);
        io::stdout().flush().ok();
        true // continue
    })?;

    println!();
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("✅ Generation complete!");

    Ok(())
}
