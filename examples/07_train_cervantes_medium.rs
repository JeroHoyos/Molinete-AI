//! Entrenar un modelo GPT-2 Mediano con Shakespeare (TAMAÑO ÓPTIMO)
//!
//! Este ejemplo demuestra el entrenamiento de un modelo transformer de tamaño óptimo:
//! - **Tamaño del modelo**: ~4M parámetros (n_embd=256, n_layers=4, n_heads=4)
//! - **Tiempo de entrenamiento**: Aproximadamente 1-2 horas en una CPU moderna
//! - **Resultados esperados**: La perplejidad baja de 600+ a ~60-80
//! - **Texto generado**: Pasajes coherentes de varias oraciones con estilo shakesperiano
//!
//! **Este es el TAMAÑO ÓPTIMO de modelo para el corpus de Shakespeare.**
//! El conteo de 4M parámetros da una relación ideal de 4:1 entre parámetros y tokens,
//! evitando tanto subajuste como sobreajuste.
//!
//! ## Qué aprenderás
//!
//! - Por qué el tamaño correcto del modelo importa más que "más grande es mejor"
//! - Cómo ajustar la capacidad del modelo al tamaño del dataset
//! - Diferencia entre subajuste, ajuste óptimo y sobreajuste
//! - Por qué la atención multi-cabeza mejora la calidad
//! - Cómo detectar convergencia durante el entrenamiento
//!
//! ## Archivos generados
//!
//! `data/shakespeare_medium_<timestamp>/`
//!
//! - `training_log.csv` → métricas de entrenamiento
//! - `checkpoint_best.bin` → mejor modelo
//! - `checkpoint_step_*.bin` → checkpoints periódicos
//! - `checkpoint_final.bin` → modelo final
//!
//! ## Uso
//!
//! ```bash
//! cargo run --release --example 07_train_shakespeare_medium
//! ```
//!
//! Ejecutar en segundo plano:
//!
//! ```bash
//! cargo run --release --example 07_train_shakespeare_medium > training.log 2>&1 &
//! ```
//!
//! Ver progreso:
//!
//! ```bash
//! tail -f training.log
//! tail -f data/shakespeare_medium_*/training_log.csv
//! ```
//!
//! ## Requisitos
//!
//! Descargar el corpus:
//!
//! ```bash
//! curl -o shakespeare.txt https://www.gutenberg.org/files/100/100-0.txt
//! ```
//!
//! ## Configuración
//!
//! - Vocabulario: 1536 tokens
//! - Contexto: 256 tokens
//! - Dimensión embedding: 256
//! - Capas: 4
//! - Cabezas de atención: 4
//! - Pasos entrenamiento: 8000
//! - Learning rate: 0.0003
//! - Acumulación gradiente: 8
//!
//! Datos:
//!
//! - Tokenizador: primeros 2M caracteres
//! - Modelo: primeros 2M caracteres
//! - Validación: 10%
//!
//! ## Resultados esperados
//!
//! - Pérdida entrenamiento: ~6.5 → ~3.5
//! - Pérdida validación: ~6.5 → ~3.8
//! - Perplejidad entrenamiento: ~650 → ~33
//! - Perplejidad validación: ~650 → ~45
//!
//! Generación inicial:
//!
//! ```text
//! To be, or not to be¿þ§random_gibberish
//! ```
//!
//! Generación final:
//!
//! ```text
//! To be, or not to be, that is the question:
//! Whether 'tis nobler in the mind to suffer
//! The slings and arrows of outrageous fortune,
//! Or to take arms against a sea of troubles,
//! And by opposing end them.
//! ```

use molineteai::{
    gpt2_trainable::{train_gpt2, TrainableGPT2},
    BPETokenizer, Config,
};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() -> Result<(), Box<dyn std::error::Error>> {

    println!("\n{}", "=".repeat(70));
    println!("  Entrenando GPT-2 Mediano con Shakespeare");
    println!("{}", "=".repeat(70));
    println!();

    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let run_dir = format!("data/shakespeare_medium_{}", timestamp);
    fs::create_dir_all(&run_dir)?;
    println!("📁 Directorio de salida: {}/\n", run_dir);

    println!("{}", "=".repeat(70));
    println!("1. Cargando datos de entrenamiento");
    println!("{}", "=".repeat(70));
    println!();

    let text = fs::read_to_string("cervantes.txt").map_err(|_| {
        "No se encontró cervantes.txt"
    })?;

    println!(
        "Cargado: {} bytes ({:.2} MB)",
        text.len(),
        text.len() as f64 / 1_000_000.0
    );
    println!("Caracteres: {}", text.chars().count());

    let tokenizer_chars = 2_000_000.min(text.chars().count());
    let model_training_chars = 2_000_000.min(text.chars().count());
    let validation_fraction = 0.1;

    println!("\nConfiguración de datos:");
    println!("  Tokenizador: primeros {} caracteres", tokenizer_chars);
    println!("  Modelo: primeros {} caracteres", model_training_chars);
    println!("  Validación: {:.0}%", validation_fraction * 100.0);

    let tokenizer_text: String = text.chars().take(tokenizer_chars).collect();
    let model_text: String = text.chars().take(model_training_chars).collect();

    println!("\n✓ Corpus tokenizador: {} bytes", tokenizer_text.len());
    println!("✓ Corpus entrenamiento: {} bytes", model_text.len());

    println!("\n{}", "=".repeat(70));
    println!("2. Entrenando tokenizador");
    println!("{}", "=".repeat(70));
    println!();

    println!("Entrenando tokenizador BPE (vocab=1536)");

    let mut tokenizer = BPETokenizer::new(1536);
    tokenizer.train(&tokenizer_text, 1536);

    println!(
        "✓ Tokenizador entrenado: {} tokens",
        tokenizer.vocab_size()
    );

    tokenizer.analyze_vocabulary(&tokenizer_text);

    let tokenizer_path = format!("{}/tokenizer.json", run_dir);
    tokenizer.save(&tokenizer_path)?;
    println!("✓ Tokenizador guardado en {}", tokenizer_path);

    let encoded = tokenizer.encode(&model_text);
    let compression_ratio = model_text.len() as f64 / encoded.len() as f64;

    println!("\nEstadísticas de tokenización:");
    println!("  Bytes originales: {}", model_text.len());
    println!("  Tokens codificados: {}", encoded.len());
    println!("  Compresión: {:.2}x", compression_ratio);

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

    println!("Configuración del modelo:");
    println!("  Vocabulario: {}", config.vocab_size);
    println!("  Dimensión embedding: {}", config.n_embd);
    println!("  Capas: {}", config.n_layers);
    println!("  Cabezas atención: {}", config.n_heads);
    println!("  Contexto: {}", config.block_size);

    let mut model = TrainableGPT2::new(&config);
    let num_params = model.num_parameters();

    println!("\nTamaño del modelo:");
    println!("  Parámetros: {}", num_params);
    println!("  {:.2}M parámetros", num_params as f64 / 1_000_000.0);

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

    println!("Configuración entrenamiento:");
    println!("  Pasos: {}", num_steps);
    println!("  Learning rate: {}", learning_rate);
    println!("  Longitud secuencia: {}", seq_len);
    println!("  Early stopping: {}", patience);

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

    println!("\n{}", "=".repeat(70));
    println!("5. Generación final");
    println!("{}", "=".repeat(70));
    println!();

    let prompts = vec![
        ("To be, or not to be", 0.8),
        ("ROMEO.", 0.9),
        ("What is", 0.7),
        ("The king", 0.8),
        ("Love", 1.0),
    ];

    for (prompt, temperature) in prompts {

        let prompt_tokens = tokenizer.encode(prompt);
        let generated = model.generate(&prompt_tokens, 80, temperature);
        let generated_text = tokenizer.decode(&generated);

        let display_text: String = generated_text.chars().take(150).collect();

        println!("Prompt: \"{}\" (temperatura={})", prompt, temperature);
        println!("Generado:");
        println!("  {}", display_text);

        if generated_text.len() > 150 {
            println!("  ... (recortado)");
        }

        println!();
    }

    println!("{}", "=".repeat(70));
    println!("Resumen");
    println!("{}", "=".repeat(70));

    println!("✅ Entrenamiento completado");
    println!("Archivos en {}", run_dir);

    println!("Resultados esperados:");
    println!("  Perplejidad validación: ~45");
    println!("  Pérdida validación: ~3.8");
    println!("  Texto: múltiples oraciones coherentes");

    Ok(())
}