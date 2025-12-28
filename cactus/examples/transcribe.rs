//! Audio transcription example using Cactus Rust bindings
//!
//! Usage: cargo run --example transcribe -- path/to/whisper/model path/to/audio.wav
//!
//! Note: Requires a Whisper-compatible model (e.g., whisper-tiny, whisper-base)

use cactus::{Model, TranscribeOptions};
use std::env;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        println!("🌵 Cactus Rust Bindings - Audio Transcription Example");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();
        println!("Usage: cargo run --example transcribe -- <model_path> <audio_file>");
        println!();
        println!("Arguments:");
        println!("  model_path  - Path to Whisper model weights folder");
        println!("  audio_file  - Path to audio file (WAV, MP3, etc.)");
        println!();
        println!("Example:");
        println!("  cargo run --example transcribe -- ./whisper-tiny ./speech.wav");
        println!();
        println!("Supported models:");
        println!("  - whisper-tiny   (39M params, fastest)");
        println!("  - whisper-base   (74M params)");
        println!("  - whisper-small  (244M params)");
        println!("  - whisper-medium (769M params)");
        println!("  - whisper-large  (1.5B params, most accurate)");
        return Ok(());
    }

    let model_path = &args[1];
    let audio_path = &args[2];

    println!("🌵 Cactus Rust Bindings - Audio Transcription Example");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Load Whisper model
    println!("📦 Loading model: {}", model_path);
    let model = Model::from_gguf(model_path)?;
    println!("✅ Model loaded!");
    println!();

    // Transcribe with streaming
    println!("🎤 Transcribing: {}", audio_path);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let options = TranscribeOptions::default();

    // Streaming transcription - print as we go
    print!("📝 ");
    let result = model.transcribe_streaming(audio_path, options, |text, _token_id| {
        print!("{}", text);
        io::stdout().flush().ok();
        true // continue
    })?;

    println!();
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("✅ Transcription complete!");

    if !result.language.is_empty() {
        println!("🌍 Language: {}", result.language);
    }
    if result.duration > 0.0 {
        println!("⏱️  Duration: {:.2}s", result.duration);
    }

    // Show segments if available
    if !result.segments.is_empty() {
        println!();
        println!("📊 Segments:");
        for (i, seg) in result.segments.iter().enumerate() {
            println!(
                "  [{:02}] [{:.2}s - {:.2}s] {}",
                i + 1,
                seg.start,
                seg.end,
                seg.text
            );
        }
    }

    Ok(())
}
