//! Demostración de la Infraestructura de Entrenamiento
//!
//! Este ejemplo demuestra la infraestructura de entrenamiento sin entrenar realmente:
//! - Carga de datos y creación de batches
//! - División entrenamiento/validación
//! - Registro de métricas
//!
//! Esto muestra cómo encajan las piezas antes de implementar el bucle completo de entrenamiento.
//!
//! La salida se escribe en: `data/example_training_<timestamp>/`

use molineteai::{compute_dataset_loss, train_val_split, BPETokenizer, TextDataLoader, TrainingLogger};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Demostración de la Infraestructura de Entrenamiento ===\n");

    // Crear un directorio de salida con timestamp
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let run_dir = format!("data/example_training_{}", timestamp);
    fs::create_dir_all(&run_dir)?;

    println!("Directorio de salida: {}\n", run_dir);

    // Cargar una pequeña muestra de texto
    let text = fs::read_to_string("cervantes.txt")
        .expect("cervantes.txt no encontrado - descárgalo de Project Gutenberg");

    // Usar los primeros 100K caracteres para esta demostración
    let text: String = text.chars().take(100_000).collect();
    println!("Se cargaron {} caracteres de texto\n", text.len());

    // Entrenar un tokenizador pequeño
    println!("Entrenando tokenizador (vocab_size=512)...");
    let mut tokenizer = BPETokenizer::new(512);
    tokenizer.train(&text, 512);
    println!(
        "Tokenizador entrenado: {} tokens en el vocabulario\n",
        tokenizer.vocab_size()
    );

    // Analizar el vocabulario para mostrar lo que se aprendió
    tokenizer.analyze_vocabulary(&text);

    // ========================================================================
    // 1. Carga de Datos
    // ========================================================================
    println!("{}", "=".repeat(70));
    println!("1. Carga de Datos");
    println!("{}", "=".repeat(70));

    let seq_len = 64;
    let batch_size = 4;

    let mut loader = TextDataLoader::new(&text, &tokenizer, seq_len, batch_size);

    println!("\nData loader creado:");
    println!("  Longitud de secuencia: {}", seq_len);
    println!("  Tamaño del batch: {}", batch_size);
    println!("  Batches estimados por época: {}\n", loader.num_batches());

    // Obtener algunos batches para demostrar
    println!("Obteniendo batches de ejemplo:");
    for i in 0..3 {
        if let Some((inputs, targets)) = loader.next_batch() {
            println!(
                "  Batch {}: {} secuencias × {} tokens",
                i + 1,
                inputs.len(),
                inputs[0].len()
            );

            // Mostrar la primera secuencia
            if i == 0 {
                println!("\n  Primera secuencia del batch:");
                println!("    Tokens de entrada:  {:?}...", &inputs[0][..10]);
                println!("    Tokens objetivo: {:?}...", &targets[0][..10]);
                println!("    (los targets son los inputs desplazados en 1)");
            }
        }
    }

    // ========================================================================
    // 2. División Train/Val
    // ========================================================================
    println!("\n{}", "=".repeat(70));
    println!("2. División Entrenamiento/Validación");
    println!("{}", "=".repeat(70));

    let all_tokens = tokenizer.encode(&text);
    let (train_tokens, val_tokens) = train_val_split(&all_tokens, 0.1);

    println!("\nDivisión de datos:");
    println!("  Tokens totales: {}", all_tokens.len());
    println!("  Tokens de entrenamiento: {} (90%)", train_tokens.len());
    println!("  Tokens de validación: {} (10%)", val_tokens.len());

    // ========================================================================
    // 3. Cálculo de Loss
    // ========================================================================
    println!("\n{}", "=".repeat(70));
    println!("3. Cálculo de Loss del Dataset");
    println!("{}", "=".repeat(70));

    println!("\nCalculando la loss en el conjunto de validación...");

    // Función de pérdida ficticia (baseline aleatoria)
    let vocab_size = tokenizer.vocab_size();
    let random_loss = (vocab_size as f32).ln(); // Loss para adivinanza aleatoria

    let val_loss = compute_dataset_loss(
        val_tokens,
        seq_len,
        10, // número de batches
        |_input, _target| {
            // En un entrenamiento real esto sería:
            // model.compute_loss(input, target)

            // Para esta demo devolvemos una loss de referencia aleatoria
            random_loss
        },
    );

    println!("  Loss de validación: {:.4}", val_loss);
    println!("  (Baseline aleatorio para vocab_size={})", vocab_size);
    println!("  Perplejidad: {:.2}", val_loss.exp());

    // ========================================================================
    // 4. Logger de Entrenamiento
    // ========================================================================
    println!("\n{}", "=".repeat(70));
    println!("4. Logger de Entrenamiento");
    println!("{}", "=".repeat(70));

    let log_path = format!("{}/training_log.csv", run_dir);
    println!("\nCreando logger de entrenamiento: {}", log_path);

    let mut logger = TrainingLogger::new(&log_path)?;

    println!("\nSimulando pasos de entrenamiento:");
    // Simular mejora de la loss en 10 pasos
    for step in 0..10 {
        let train_loss = random_loss * (1.0 - step as f32 * 0.05); // Mejora ficticia
        let val_loss = random_loss * (1.0 - step as f32 * 0.04);

        let sample = if step % 3 == 0 {
            Some("Ser o no ser") // Ejemplo ficticio
        } else {
            None
        };

        logger.log(step * 10, 0.001, train_loss, val_loss, sample)?;
    }

    println!("\n✅ Log de entrenamiento escrito en: {}", log_path);
    println!("   Puedes verlo con: cat {}", log_path);
    println!("   O importarlo en Excel/Python para graficar");
    println!("\nTodas las salidas se guardaron en: {}/", run_dir);

    // ========================================================================
    // Resumen
    // ========================================================================
    println!("\n{}", "=".repeat(70));
    println!("Resumen");
    println!("{}", "=".repeat(70));

    println!("\n✓ Data loader: crea batches de secuencias de tokens eficientemente");
    println!("✓ División train/val: separa datos para evaluación");
    println!("✓ Cálculo de loss: evalúa el modelo sobre el dataset");
    println!("✓ Logger de entrenamiento: guarda métricas en CSV para análisis");

    Ok(())
}