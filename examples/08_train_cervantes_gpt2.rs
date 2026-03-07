//! Entrenar un modelo GPT-2 mediano con textos de Cervantes
//!
//! Este ejemplo demuestra cómo entrenar un modelo transformer de tamaño óptimo
//! utilizando un corpus inspirado en las obras de Miguel de Cervantes.
//!
//! - **Tamaño del modelo**: ~4M parámetros (n_embd=256, n_layers=4, n_heads=4)
//! - **Tiempo de entrenamiento**: Aproximadamente 1–2 horas en CPU moderna
//! - **Resultados esperados**: La perplejidad baja de 600+ a ~60–80
//! - **Texto generado**: Pasajes coherentes de varias frases con estilo
//!   literario clásico español.
//!
//! **¡Este es el tamaño óptimo de modelo para un corpus de Cervantes!**
//!
//! Los ~4M de parámetros proporcionan una relación ideal entre parámetros
//! y tokens, evitando tanto el subajuste como el sobreajuste cuando se
//! entrena con literatura clásica española.
//!
//! ## Lo que aprenderás
//!
//! - Por qué el tamaño correcto del modelo importa más que hacerlo más grande
//! - Cómo ajustar la capacidad del modelo al tamaño del dataset
//! - La diferencia entre subajuste, ajuste óptimo y sobreajuste
//! - Por qué la atención multi-cabeza mejora la calidad del texto generado
//!
//! ## Salida
//!
//! Todos los resultados se guardan en:
//!
//! `data/cervantes_medium_<timestamp>/`

use molineteai::{
    gpt2_trainable::{train_gpt2, TrainableGPT2},
    BPETokenizer, Config,
};

use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() -> Result<(), Box<dyn std::error::Error>> {

    println!("\n{}", "=".repeat(70));
    println!("  Entrenando GPT-2 Mediano con textos de Cervantes");
    println!("{}", "=".repeat(70));
    println!();

    // Crear directorio de salida con timestamp
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let run_dir = format!("data/cervantes_medium_{}", timestamp);

    fs::create_dir_all(&run_dir)?;

    println!("📁 Directorio de salida: {}/\n", run_dir);

    // ============================================================
    // 1. Cargar datos de entrenamiento
    // ============================================================

    println!("{}", "=".repeat(70));
    println!("1. Cargando corpus de Cervantes");
    println!("{}", "=".repeat(70));
    println!();

    let text = fs::read_to_string("cervantes.txt").map_err(|_| {
        "No se encontró cervantes.txt.\nDescarga un corpus con las obras de Cervantes."
    })?;

    println!(
        "Cargado: {} bytes ({:.2} MB)",
        text.len(),
        text.len() as f64 / 1_000_000.0
    );

    println!("Caracteres: {}", text.chars().count());

    // ============================================================
    // Configuración de datos
    // ============================================================

    let tokenizer_chars = 2_000_000.min(text.chars().count());
    let model_training_chars = 2_000_000.min(text.chars().count());
    let validation_fraction = 0.1;

    println!("\nConfiguración de datos:");
    println!("  Entrenamiento del tokenizer: primeros {} caracteres", tokenizer_chars);
    println!("  Entrenamiento del modelo: primeros {} caracteres", model_training_chars);
    println!("  División de validación: {:.0}%", validation_fraction * 100.0);

    let tokenizer_text: String = text.chars().take(tokenizer_chars).collect();
    let model_text: String = text.chars().take(model_training_chars).collect();

    println!("\n✓ Corpus para tokenizer: {} bytes", tokenizer_text.len());
    println!("✓ Corpus para entrenamiento del modelo: {} bytes", model_text.len());

    // ============================================================
    // 2. Entrenar tokenizer
    // ============================================================

    println!("\n{}", "=".repeat(70));
    println!("2. Entrenando tokenizer BPE");
    println!("{}", "=".repeat(70));
    println!();

    let mut tokenizer = BPETokenizer::new(1536);
    tokenizer.train(&tokenizer_text, 1536);

    println!(
        "✓ Tokenizer entrenado: {} tokens en el vocabulario",
        tokenizer.vocab_size()
    );

    tokenizer.analyze_vocabulary(&tokenizer_text);

    let tokenizer_path = format!("{}/tokenizer.json", run_dir);
    tokenizer.save(&tokenizer_path)?;

    println!("✓ Tokenizer guardado en: {}", tokenizer_path);

    let encoded = tokenizer.encode(&model_text);
    let compression_ratio = model_text.len() as f64 / encoded.len() as f64;

    println!("\nEstadísticas de tokenización:");
    println!("  Bytes originales: {}", model_text.len());
    println!("  Tokens codificados: {}", encoded.len());
    println!("  Ratio de compresión: {:.2}x", compression_ratio);

    // ============================================================
    // 3. Crear modelo
    // ============================================================

    println!("\n{}", "=".repeat(70));
    println!("3. Creando modelo");
    println!("{}", "=".repeat(70));
    println!();

    let config = Config {
        vocab_size: tokenizer.vocab_size(),
        n_embd: 256,
        n_layers: 4,
        n_heads: 4,
        block_size: 256,
        dropout_rate: 0.1,
    };

    let mut model = TrainableGPT2::new(&config);
    let num_params = model.num_parameters();

    println!("Tamaño del modelo:");
    println!("  Parámetros totales: {}", num_params);
    println!("  Tamaño: {:.2}M parámetros", num_params as f64 / 1_000_000.0);

    println!("\nConfiguración óptima para corpus cervantino.");

    // ============================================================
    // 4. Entrenar modelo
    // ============================================================

    println!("\n{}", "=".repeat(70));
    println!("4. Entrenamiento");
    println!("{}", "=".repeat(70));
    println!();

    let num_steps = 100000;
    let learning_rate = 0.0003;
    let seq_len = config.block_size;

    let patience = 5000;
    let warmup_fraction = 0.1;
    let gradient_clip_norm = 1.0;
    let weight_decay = 0.1;

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

    // ============================================================
    // 5. Generación final de texto
    // ============================================================

    println!("\n{}", "=".repeat(70));
    println!("5. Generación de texto cervantino");
    println!("{}", "=".repeat(70));
    println!();

    let prompts = vec![
        ("En un lugar de la Mancha", 0.8),
        ("Don Quijote dijo", 0.9),
        ("Sancho Panza respondió", 0.8),
        ("El caballero de la triste figura", 0.7),
        ("Sobre los molinos de viento", 1.0),
    ];

    for (prompt, temperature) in prompts {

        let prompt_tokens = tokenizer.encode(prompt);
        let generated = model.generate(&prompt_tokens, 80, temperature);

        let generated_text = tokenizer.decode(&generated);
        let display_text: String = generated_text.chars().take(150).collect();

        println!("Prompt: \"{}\" (temperatura={})", prompt, temperature);
        println!("Texto generado:");
        println!("  {}", display_text);

        if generated_text.len() > 150 {
            println!("  ... (truncado)");
        }

        println!();
    }

    println!("{}", "=".repeat(70));
    println!("Resumen");
    println!("{}", "=".repeat(70));

    println!("✅ ¡Entrenamiento completado con éxito!");
    println!("Archivos de salida en: {}", run_dir);

    Ok(())
}