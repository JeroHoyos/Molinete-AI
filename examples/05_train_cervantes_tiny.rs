//! Entrenar un Modelo GPT-2 Pequeño con Cervantes (Demostración de Overfitting)
//!
//! Este ejemplo demuestra el entrenamiento de un modelo transformer mínimo usando
//! textos inspirados en la obra de Miguel de Cervantes, especialmente
//! Don Quijote de la Mancha.
//!
//! - **Tamaño del modelo**: ~170K parámetros (n_embd=64, n_layers=2, n_heads=1)
//! - **Tiempo de entrenamiento**: 20–40 minutos en CPU moderno
//! - **Datos de entrenamiento**: Corpus de Cervantes (~5M caracteres)
//! - **Pasos de entrenamiento**: 10000
//!
//! El modelo tiene menos parámetros que tokens de entrenamiento,
//! lo que reduce el overfitting.
//!
//! ## Lo que aprenderás
//!
//! - Bucle completo de entrenamiento
//! - Gradient accumulation y clipping
//! - Learning rate scheduling
//! - Early stopping
//! - Guardado de checkpoints
//! - Generación de texto cervantino
//!
//! ## Salida
//!
//! Todos los resultados se guardan en:
//!
//! `data/cervantes_tiny_<timestamp>/`
//!
//! - `training_log.csv`
//! - `checkpoint_best.bin`
//! - `checkpoint_step_*.bin`
//! - `checkpoint_final.bin`

use molineteai::{
    gpt2_trainable::{train_gpt2, TrainableGPT2},
    BPETokenizer, Config,
};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() -> Result<(), Box<dyn std::error::Error>> {

    println!("\n{}", "=".repeat(70));
    println!("  Entrenando GPT-2 Pequeño con textos de Cervantes");
    println!("  (Inspirado en Don Quijote de la Mancha)");
    println!("{}", "=".repeat(70));
    println!();

    println!("📜 NOTA LITERARIA:");
    println!("Este ejemplo usa un corpus inspirado en la obra de Cervantes.");
    println!("El modelo intentará aprender el estilo narrativo del Siglo de Oro.");
    println!();

    // Crear directorio de salida
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let run_dir = format!("data/cervantes_tiny_{}", timestamp);
    fs::create_dir_all(&run_dir)?;

    println!("📁 Directorio de salida: {}/\n", run_dir);

    // ================================================================
    // 1. Cargar datos
    // ================================================================

    println!("{}", "=".repeat(70));
    println!("1. Cargando textos de Cervantes");
    println!("{}", "=".repeat(70));
    println!();

    let text = fs::read_to_string("cervantes.txt").map_err(|_| {
        "cervantes.txt no encontrado.\nColoca un corpus de Cervantes en el archivo."
    })?;

    println!(
        "Cargado: {} bytes ({:.2} MB)",
        text.len(),
        text.len() as f64 / 1_000_000.0
    );

    println!("Caracteres: {}", text.chars().count());

    // ================================================================
    // Configuración de datos
    // ================================================================

    let tokenizer_chars = text.chars().count();
    let model_training_chars = text.chars().count();
    let validation_fraction = 0.1;

    println!("\nConfiguración de datos:");
    println!("  Entrenamiento tokenizer: {} caracteres", tokenizer_chars);
    println!("  Entrenamiento modelo: {} caracteres", model_training_chars);
    println!("  Validación: {}%", validation_fraction * 100.0);

    let tokenizer_text: String = text.chars().take(tokenizer_chars).collect();
    let model_text: String = text.chars().take(model_training_chars).collect();

    println!("✓ Corpus tokenizer: {} bytes", tokenizer_text.len());
    println!("✓ Corpus entrenamiento: {} bytes", model_text.len());

    // ================================================================
    // 2. Entrenar tokenizer
    // ================================================================

    println!("\n{}", "=".repeat(70));
    println!("2. Entrenando Tokenizer BPE");
    println!("{}", "=".repeat(70));
    println!();

    let mut tokenizer = BPETokenizer::new(512);
    tokenizer.train(&tokenizer_text, 512);

    println!(
        "✓ Tokenizer entrenado: {} tokens en vocabulario",
        tokenizer.vocab_size()
    );

    tokenizer.analyze_vocabulary(&tokenizer_text);

    let tokenizer_path = format!("{}/tokenizer.json", run_dir);
    tokenizer.save(&tokenizer_path)?;

    println!("✓ Tokenizer guardado en {}", tokenizer_path);

    let encoded = tokenizer.encode(&model_text);

    println!("\nEstadísticas de tokenización:");
    println!("  Bytes originales: {}", model_text.len());
    println!("  Tokens: {}", encoded.len());

    // ================================================================
    // 3. Crear modelo
    // ================================================================

    println!("\n{}", "=".repeat(70));
    println!("3. Creando modelo transformer");
    println!("{}", "=".repeat(70));
    println!();

    let config = Config {
        vocab_size: tokenizer.vocab_size(),
        n_embd: 64,
        n_layers: 2,
        n_heads: 1,
        block_size: 64,
        dropout_rate: 0.1,
    };

    let mut model = TrainableGPT2::new(&config);

    let num_params = model.num_parameters();

    println!("Parámetros totales: {}", num_params);
    println!("Tamaño aproximado: {:.1}K parámetros", num_params as f64 / 1000.0);

    // ================================================================
    // 4. Entrenamiento
    // ================================================================

    println!("\n{}", "=".repeat(70));
    println!("4. Entrenando modelo cervantino");
    println!("{}", "=".repeat(70));
    println!();

    let num_steps = 10000;
    let learning_rate = 0.003;
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
    // 5. Generación de texto cervantino
    // ================================================================

    println!("\n{}", "=".repeat(70));
    println!("5. Generación de texto");
    println!("{}", "=".repeat(70));
    println!();

    let test_cases = vec![

        ("En un lugar de la Mancha", 0.8, 80),
        ("Don Quijote dijo", 0.8, 80),
        ("Sancho Panza respondió", 0.8, 80),
        ("El caballero de la triste figura", 0.8, 80),
        ("Sobre los molinos de viento", 0.8, 80),

        ("En un lugar de la Mancha", 0.3, 80),
        ("En un lugar de la Mancha", 1.2, 80),

    ];

    for (prompt, temperature, max_tokens) in test_cases {

        let prompt_tokens = tokenizer.encode(prompt);

        let generated = model.generate(&prompt_tokens, max_tokens, temperature);

        let generated_text = tokenizer.decode(&generated);

        println!("────────────────────────────────────────────");
        println!("Prompt: \"{}\" (temperatura {})", prompt, temperature);
        println!("────────────────────────────────────────────");

        println!("{}", generated_text);
        println!();
    }

    // ================================================================
    // Resumen
    // ================================================================

    println!("\n{}", "=".repeat(70));
    println!("Resumen del entrenamiento");
    println!("{}", "=".repeat(70));

    println!("✅ Entrenamiento completado");

    println!("Archivos generados en {}", run_dir);

    println!("  training_log.csv");
    println!("  tokenizer.json");
    println!("  checkpoint_best.bin");
    println!("  checkpoint_final.bin");

    println!("\nEl modelo ahora intenta imitar el estilo narrativo de Cervantes.");

    Ok(())
}