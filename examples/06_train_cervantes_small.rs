//! Train a Small GPT-2 Model on Cervantes
//!
//! This example demonstrates training a small transformer model
//! on texts inspired by Miguel de Cervantes and the literary
//! style of the Spanish Golden Age.
//!
//! - **Model size**: ~200K parameters (n_embd=128, n_layers=3, n_heads=1)
//! - **Training time**: Approximately 10-20 minutes on modern CPU
//!
//! This configuration provides a good balance between training
//! speed and model capacity for experimentation with classical
//! Spanish literature style generation.
//!
//! ## What You'll Learn
//!
//! - How model size affects learning quality
//! - The relationship between training time and perplexity
//! - Diminishing returns of training (loss eventually plateaus)
//! - How tokenization quality affects literary text generation
//!
//! ## Output
//!
//! All outputs are saved to: `data/cervantes_small_<timestamp>/`
//!
//! - `training_log.csv` - Step-by-step training metrics
//! - `checkpoint_best.bin` - Model with best validation loss
//! - `checkpoint_step_*.bin` - Periodic checkpoints every 250 steps
//! - `checkpoint_final.bin` - Final model after all steps
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --example 06_train_cervantes_small
//! ```
//!
//! ## Prerequisites
//!
//! Prepare a Cervantes corpus first:
//!
//! ```bash
//! curl -o cervantes.txt https://www.gutenberg.org/files/996/996-0.txt
//! ```
//!
//! ## Default Configuration
//!
//! Model architecture:
//!
//! - Vocabulary size: 1024 tokens
//! - Context length: 128 tokens
//! - Embedding dimension: 128
//! - Number of layers: 3
//! - Number of heads: 1
//!
//! Training hyperparameters:
//!
//! - Training steps: 2000
//! - Learning rate: 0.002
//! - Warmup: 10% of training steps
//! - Gradient clipping: 1.0
//! - Early stopping patience: 5000 steps
//! - Sequence length: 128 tokens
//! - Gradient accumulation: 8 mini-batches
//!
//! Data configuration:
//!
//! - Tokenizer training data: First 500K characters
//! - Model training data: First 500K characters
//! - Validation split: 10%
//!
//! These values are good starting points for exploring
//! generative models trained on classical Spanish literature.

use molineteai::{
    gpt2_trainable::{train_gpt2, TrainableGPT2},
    BPETokenizer, Config,
};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() -> Result<(), Box<dyn std::error::Error>> {

    println!("\n{}", "=".repeat(70));
    println!("  Training Small GPT-2 on Cervantes");
    println!("{}", "=".repeat(70));
    println!();

    // Create timestamped output directory
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let run_dir = format!("data/cervantes_small_{}", timestamp);

    fs::create_dir_all(&run_dir)?;
    println!("📁 Output directory: {}/\n", run_dir);

    // ================================================================
    // 1. Load Training Data
    // ================================================================

    println!("{}", "=".repeat(70));
    println!("1. Loading Cervantes Corpus");
    println!("{}", "=".repeat(70));
    println!();

    let text = fs::read_to_string("cervantes.txt").map_err(|_| {
        "cervantes.txt not found.\nDownload a Cervantes corpus first."
    })?;

    println!(
        "Loaded: {} bytes ({:.2} MB)",
        text.len(),
        text.len() as f64 / 1_000_000.0
    );

    println!("Characters: {}", text.chars().count());

    // ================================================================
    // Data Configuration
    // ================================================================

    let tokenizer_chars = 500_000;
    let model_training_chars = 500_000;
    let validation_fraction = 0.1;

    println!("\nData configuration:");
    println!("  Tokenizer training: first {} characters", tokenizer_chars);
    println!("  Model training: first {} characters", model_training_chars);
    println!("  Validation split: {:.0}%", validation_fraction * 100.0);

    let tokenizer_text: String = text.chars().take(tokenizer_chars).collect();
    let model_text: String = text.chars().take(model_training_chars).collect();

    println!("\n✓ Tokenizer corpus: {} bytes", tokenizer_text.len());
    println!("✓ Model training corpus: {} bytes", model_text.len());

    // ================================================================
    // 2. Train Tokenizer
    // ================================================================

    println!("\n{}", "=".repeat(70));
    println!("2. Training BPE Tokenizer");
    println!("{}", "=".repeat(70));
    println!();

    let mut tokenizer = BPETokenizer::new(1024);
    tokenizer.train(&tokenizer_text, 1024);

    println!(
        "✓ Tokenizer trained: {} tokens in vocabulary",
        tokenizer.vocab_size()
    );

    tokenizer.analyze_vocabulary(&tokenizer_text);

    let tokenizer_path = format!("{}/tokenizer.json", run_dir);
    tokenizer.save(&tokenizer_path)?;

    println!("✓ Tokenizer saved to: {}", tokenizer_path);

    let encoded = tokenizer.encode(&model_text);

    let compression_ratio = model_text.len() as f64 / encoded.len() as f64;

    println!("\nTokenization statistics:");
    println!("  Original bytes: {}", model_text.len());
    println!("  Encoded tokens: {}", encoded.len());
    println!("  Compression ratio: {:.2}", compression_ratio);

    // ================================================================
    // 3. Create Model
    // ================================================================

    println!("\n{}", "=".repeat(70));
    println!("3. Creating Transformer Model");
    println!("{}", "=".repeat(70));
    println!();

    let config = Config {
        vocab_size: tokenizer.vocab_size(),
        n_embd: 128,
        n_layers: 3,
        n_heads: 1,
        block_size: 128,
        dropout_rate: 0.1,
    };

    let mut model = TrainableGPT2::new(&config);

    let num_params = model.num_parameters();

    println!("Model parameters: {}", num_params);
    println!("Size: {:.2}M parameters", num_params as f64 / 1_000_000.0);

    // ================================================================
    // 4. Training
    // ================================================================

    println!("\n{}", "=".repeat(70));
    println!("4. Training Model");
    println!("{}", "=".repeat(70));
    println!();

    let num_steps = 100000;
    let learning_rate = 0.002;
    let seq_len = config.block_size;
    let patience = 5000;
    let warmup_fraction = 0.1;
    let gradient_clip_norm = 1.0;

    let weight_decay = 0.01;

    train_gpt2(
        &mut model,
        &tokenizer,
        &model_text,
        num_steps,
        learning_rate,
        seq_len,
        Some(&run_dir),
        patience,
        warmup_fraction,
        gradient_clip_norm,
        validation_fraction,
        weight_decay,
    );

    // ================================================================
    // 5. Text Generation
    // ================================================================

    println!("\n{}", "=".repeat(70));
    println!("5. Cervantine Text Generation");
    println!("{}", "=".repeat(70));
    println!();

    let test_cases = vec![

        ("En un lugar de la Mancha", 0.8, 80),
        ("Don Quijote dijo", 0.8, 80),
        ("Sancho Panza respondió", 0.8, 80),
        ("El caballero de la triste figura", 0.8, 80),
        ("Los molinos de viento parecían", 0.8, 80),

        ("En un lugar de la Mancha", 0.3, 80),
        ("En un lugar de la Mancha", 1.2, 80),

    ];

    for (prompt, temperature, max_tokens) in test_cases {

        let prompt_tokens = tokenizer.encode(prompt);
        let generated = model.generate(&prompt_tokens, max_tokens, temperature);
        let generated_text = tokenizer.decode(&generated);

        println!("────────────────────────────────────────────────────────────");
        println!("Prompt: \"{}\" (temperature: {})", prompt, temperature);
        println!("────────────────────────────────────────────────────────────");

        println!("{}", generated_text);
        println!();
    }

    println!("{}", "=".repeat(70));
    println!("Summary");
    println!("{}", "=".repeat(70));

    println!("✅ Training completed successfully!");
    println!("Output files in: {}/", run_dir);

    Ok(())
}