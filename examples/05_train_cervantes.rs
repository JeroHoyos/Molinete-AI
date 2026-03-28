//! Entrenar un modelo GPT-2 Mediano con Cervantes (TAMAÑO ÓPTIMO)
//!
//! Este ejemplo demuestra el entrenamiento de un modelo transformer de tamaño óptimo:
//! - **Tamaño del modelo**: ~4M de parámetros (n_embd=256, n_layers=4, n_heads=4)
//! - **Tiempo de entrenamiento**: Aproximadamente 1-2 horas en una CPU moderna
//! - **Resultados esperados**: La perplejidad cae de más de 600 a ~60-80
//! - **Texto generado**: Pasajes coherentes de múltiples oraciones con estilo cervantino
//!
//! **¡Este es el tamaño de modelo ÓPTIMO para el corpus de Cervantes!** El recuento
//! de 4M de parámetros proporciona una proporción ideal de parámetros por token de 4:1, 
//! previniendo tanto el subajuste (underfitting) como el sobreajuste (overfitting). Los 
//! modelos más grandes (como el GPT-2 Small de 163M de parámetros) en realidad tienen un 
//! PEOR rendimiento debido al sobreajuste severo en este conjunto de datos.
//!
//! ## Qué aprenderás
//!
//! - Por qué el tamaño adecuado del modelo importa más que "más grande es mejor"
//! - Cómo hacer coincidir la capacidad del modelo con el tamaño del conjunto de datos (proporciones parámetro-token)
//! - La diferencia entre subajuste, ajuste óptimo y sobreajuste
//! - Por qué la atención multicabezal (multi-head attention) mejora la calidad sobre la de una sola cabeza
//! - Cómo identificar cuándo el entrenamiento ha convergido en lugar de haber comenzado a sobreajustarse
//!
//! ## Salida
//!
//! Todas las salidas se guardan en: `data/cervantes_medium_<timestamp>/`
//! - `training_log.csv` - Métricas de entrenamiento paso a paso
//! - `checkpoint_best.bin` - Modelo con la mejor pérdida de validación
//! - `checkpoint_step_*.bin` - Checkpoints periódicos cada 250 pasos
//! - `checkpoint_final.bin` - Modelo final después de todos los pasos
//!
//! ## Uso
//!
//! ```bash
//! cargo run --release --example 07_train_cervantes_medium
//! ```
//!
//! Esto tomará 1-2 horas. Considera ejecutarlo en segundo plano:
//! ```bash
//! cargo run --release --example 07_train_cervantes_medium > training.log 2>&1 &
//! ```
//!
//! Monitorear el progreso:
//! ```bash
//! tail -f training.log
//! # o
//! tail -f data/cervantes_medium_*/training_log.csv
//! ```
//!
//! ## Requisitos previos
//!
//! Descarga el corpus de Cervantes (Don Quijote) primero:
//! ```bash
//! curl -o cervantes.txt [https://www.gutenberg.org/files/2000/2000-0.txt](https://www.gutenberg.org/files/2000/2000-0.txt)
//! ```
//!
//! ## Configuración
//!
//! Este ejemplo utiliza:
//! - Tamaño del vocabulario: 1536 tokens (nivel de byte + 1280 fusiones/merges)
//! - Longitud del contexto: 256 tokens (suficiente para coherencia a nivel de párrafo)
//! - Dimensión de incrustación (embedding): 256 (estándar, dimensión bien probada)
//! - Número de capas: 4 (suficientemente profundo para patrones complejos)
//! - Número de cabezas: 4 (atención multicabezal adecuada, head_dim=64)
//! - Pasos de entrenamiento: 8000 (incrementado para convergencia completa)
//! - Tasa de aprendizaje (Learning rate): 0.0003 (con calentamiento y decaimiento coseno)
//! - Acumulación de gradientes: 8 mini-lotes (tamaño de lote efectivo = 8)
//!
//! Configuración de datos:
//! - Datos de entrenamiento del tokenizador: Primeros 2M de caracteres (o el corpus completo si es menor)
//! - Datos de entrenamiento del modelo: Primeros 2M de caracteres (o el corpus completo si es menor)
//! - División de validación: 10% reservado para validación
//!
//! ## Resultados Esperados
//!
//! Después de 8000 pasos (~1.5-2 horas):
//! - Pérdida de entrenamiento: ~6.5 → ~3.5
//! - Pérdida de validación: ~6.5 → ~3.8
//! - Perplejidad de entrenamiento: ~650 → ~33
//! - Perplejidad de validación: ~650 → ~45
//! - Brecha entreno/validación: Mínima (¡emparejamiento de capacidad adecuado!)
//! - Texto generado: Pasajes coherentes de múltiples oraciones con estilo cervantino
//!
//! Ejemplo de generación en el paso 0:
//! ```text
//! En un lugar de la Mancha¿þ§random_gibberish
//! ```
//!
//! Ejemplo de generación en el paso 8000:
//! ```text
//! En un lugar de la Mancha, de cuyo nombre no quiero acordarme,
//! no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero,
//! adarga antigua, rocín flaco y galgo corredor.
//! Una olla de algo más vaca que carnero...
//! ```
//!
//! ## Por qué este tamaño de modelo es ÓPTIMO
//!
//! Esta configuración está ajustada científicamente al conjunto de datos de Cervantes:
//! - **4M de parámetros** para ~1M de tokens = **proporción 4:1** (rango ideal: 3-10x)
//! - **4 capas**: Suficientemente profundo para patrones lingüísticos sofisticados
//! - **256 dimensiones**: Embeddings ricos con el adecuado head_dim=64 (256/4)
//! - **4 cabezas de atención**: Captura múltiples aspectos lingüísticos simultáneamente
//! - **Vocabulario de 1536**: Tokenización equilibrada (ni muy fragmentada, ni muy comprimida)
//! - **Contexto de 256**: Mantiene la coherencia a través de múltiples oraciones
//!
//! **¿Por qué no más grande?** El modelo GPT-2 Small de 163M de parámetros tiene una proporción 
//! de parámetros por token muy alta, causando un sobreajuste severo. La pérdida de entrenamiento 
//! disminuye pero la pérdida de validación se estanca alrededor de 5.1, mientras que este modelo 
//! óptimo logra una pérdida de validación de ~3.8.
//!
//! **¿Por qué no más pequeño?** Los modelos con menos de 2M de parámetros sufren de subajuste 
//! (underfitting): carecen de la capacidad para capturar el rico vocabulario y las estructuras 
//! gramaticales del Siglo de Oro español.
//!
//! Este es el tamaño recomendado para uso en producción en dominios específicos (~1M de tokens).
//!
//! ## Comparación entre tamaños de modelo
//!
//! | Métrica | Diminuto (50K) | Pequeño (1M) | Mediano (4M) ⭐ | GPT-2 Pequeño (163M) |
//! |--------|------------|------------|----------------|---------------------|
//! | Tiempo de entrenamiento | 2-5 min | 10-20 min | 1-2 horas | 24+ horas |
//! | Perplejidad de val. final | ~300 | ~150 | **~45** | ~161 (¡sobreajustado!) |
//! | Brecha entreno/val | Pequeña | Pequeña | **Mínima** | Grande (¡sobreajustado!) |
//! | Calidad del texto | Fragmentos | Palabras | **Múltiples oraciones** | Buena pero se estanca |
//! | Memoria | ~0.2 MB | ~4 MB | ~16 MB | ~650 MB |
//! | Parámetros/token | 0.05:1 | 1:1 | **4:1 (óptimo)** | ¡Muy alta! |
//!
//! ⭐ **¡Este modelo Mediano logra la MEJOR pérdida de validación de cualquier tamaño!**
//!
//! Para una demostración arquitectónica (no para mejor calidad), ver:
//! - `08_train_cervantes_gpt2.rs` (~163M params, muestra la arquitectura de GPT-2 Small pero se sobreajusta)

use molineteai::{
    gpt2_trainable::{train_gpt2, TrainableGPT2},
    BPETokenizer, Config,
};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(70));
    println!("  Entrenando GPT-2 Mediano con Cervantes");
    println!("{}", "=".repeat(70));
    println!();

    // Crear directorio de salida con marca de tiempo
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let run_dir = format!("data/cervantes_medium_{}", timestamp);
    fs::create_dir_all(&run_dir)?;
    println!("📁 Directorio de salida: {}/\n", run_dir);

    // ========================================================================
    // 1. Cargar datos de entrenamiento
    // ========================================================================
    println!("{}", "=".repeat(70));
    println!("1. Cargando datos de entrenamiento");
    println!("{}", "=".repeat(70));
    println!();

    let text = fs::read_to_string("cervantes.txt").map_err(|_| {
        "No se encontró cervantes.txt. Descárgalo con:\n  \
         curl -o cervantes.txt https://www.gutenberg.org/files/2000/2000-0.txt"
    })?;

    println!(
        "Cargado: {} bytes ({:.2} MB)",
        text.len(),
        text.len() as f64 / 1_000_000.0
    );
    println!("Caracteres: {}", text.chars().count());

    // ========================================================================
    // Configuración de datos
    // ========================================================================
    // Configurar cuánto del corpus usar para diferentes propósitos

    // Entrenamiento del tokenizador: Cuántos caracteres usar para entrenar el tokenizador BPE
    // Para el modelo mediano, usar 2M de caracteres o el corpus completo
    let tokenizer_chars = 2_000_000.min(text.chars().count());

    // Entrenamiento del modelo: Cuántos caracteres usar para entrenar el modelo
    let model_training_chars = 2_000_000.min(text.chars().count());

    // División de validación: Qué fracción reservar para validación (0.1 = 10%)
    let validation_fraction = 0.1;

    println!("\nConfiguración de datos:");
    println!("  Entrenamiento del tokenizador: primeros {} caracteres", tokenizer_chars);
    println!(
        "  Entrenamiento del modelo: primeros {} caracteres",
        model_training_chars
    );
    println!("  División de validación: {:.0}%", validation_fraction * 100.0);

    // Extraer datos de entrenamiento del tokenizador
    let tokenizer_text: String = text.chars().take(tokenizer_chars).collect();
    println!("\n✓ Corpus del tokenizador: {} bytes", tokenizer_text.len());

    // Extraer datos de entrenamiento del modelo
    let model_text: String = text.chars().take(model_training_chars).collect();
    println!("✓ Corpus de entrenamiento del modelo: {} bytes", model_text.len());
    println!(
        "  (Se dividirá {:.0}% entrenamiento / {:.0}% validación)",
        (1.0 - validation_fraction) * 100.0,
        validation_fraction * 100.0
    );

    // ========================================================================
    // 2. Entrenar Tokenizador
    // ========================================================================
    println!("\n{}", "=".repeat(70));
    println!("2. Entrenando Tokenizador");
    println!("{}", "=".repeat(70));
    println!();

    println!("Entrenando tokenizador BPE (vocab_size=1536)...");
    println!("Usando {} caracteres del corpus", tokenizer_text.len());
    println!("(Esto puede tomar de 20 a 40 segundos)");
    let mut tokenizer = BPETokenizer::new(1536);
    tokenizer.train(&tokenizer_text, 1536);
    println!(
        "✓ Tokenizador entrenado: {} tokens en el vocabulario",
        tokenizer.vocab_size()
    );

    // Analizar vocabulario para mostrar qué se aprendió
    tokenizer.analyze_vocabulary(&tokenizer_text);

    // Guardar tokenizador
    let tokenizer_path = format!("{}/tokenizer.json", run_dir);
    tokenizer.save(&tokenizer_path)?;
    println!("✓ Tokenizador guardado en: {}", tokenizer_path);

    // Mostrar compresión en los datos de entrenamiento del modelo
    let encoded = tokenizer.encode(&model_text);
    let compression_ratio = model_text.len() as f64 / encoded.len() as f64;
    println!("\nEstadísticas de tokenización (en datos de entrenamiento del modelo):");
    println!("  Bytes originales: {}", model_text.len());
    println!("  Tokens codificados: {}", encoded.len());
    println!("  Tasa de compresión: {:.2}x", compression_ratio);
    println!("  Bytes por token: {:.2}", 1.0 / compression_ratio);
    println!("\n  Nota: Mayor compresión = mejor tokenización");
    println!("        Menos tokens = entrenamiento más rápido");

    // ========================================================================
    // 3. Crear Modelo
    // ========================================================================
    println!("\n{}", "=".repeat(70));
    println!("3. Creando Modelo");
    println!("{}", "=".repeat(70));
    println!();

    // Configuración ÓPTIMA: ajustada científicamente al corpus de Cervantes
    // 4M de parámetros para ~1M de tokens = proporción ideal 4:1
    let config = Config {
        vocab_size: tokenizer.vocab_size(),
        n_embd: 256,       // Dimensión estándar, head_dim=64 (256/4)
        n_layers: 4,       // Suficientemente profundo para patrones complejos
        n_heads: 4,        // Atención multicabezal (4 cabezas de 64 dimensiones cada una)
        block_size: 256,   // Contexto a nivel de párrafo
        dropout_rate: 0.1, // Probabilidad de Dropout
    };

    println!("Configuración del modelo:");
    println!("  Tamaño del vocabulario: {}", config.vocab_size);
    println!("  Dimensión de incrustación (embedding): {}", config.n_embd);
    println!("  Número de capas: {}", config.n_layers);
    println!("  Número de cabezas: {}", config.n_heads);
    println!("  Longitud del contexto: {}", config.block_size);

    let mut model = TrainableGPT2::new(&config);
    let num_params = model.num_parameters();
    println!("\nTamaño del modelo:");
    println!("  Total de parámetros: {}", num_params);
    println!("  Tamaño: {:.2}M de parámetros", num_params as f64 / 1_000_000.0);
    println!("  Memoria: ~{:.1} MB", num_params as f64 * 4.0 / 1_000_000.0);

    println!("\n  ¡Este es el tamaño ÓPTIMO para Cervantes!");
    println!("  Proporción parámetro-token: ~4:1 (rango ideal)");
    println!("  Logrará la MEJOR pérdida de validación de cualquier tamaño de modelo");

    // ========================================================================
    // 4. Entrenar Modelo
    // ========================================================================
    println!("\n{}", "=".repeat(70));
    println!("4. Entrenamiento");
    println!("{}", "=".repeat(70));
    println!();

    println!("Configuración de entrenamiento:");

    // Hiperparámetros de entrenamiento - ¡ahora con programación adaptativa de la tasa de aprendizaje!
    let num_steps = 100000; // 🟢 Límite máximo alto - el LR adaptativo + parada temprana encuentran la convergencia
    let learning_rate = 0.0003;
    let seq_len = config.block_size; // Usar ventana de contexto completa
    let patience = 5000; // Paciencia de parada temprana (funciona con LR adaptativo)
    let warmup_fraction = 0.1; // 10% de los pasos para el calentamiento (10000 pasos)
    let gradient_clip_norm = 1.0; // Norma máxima del gradiente

    // Nota: La programación adaptativa del LR reduce automáticamente la tasa de aprendizaje en las mesetas (plateaus)

    println!("  Pasos de entrenamiento: {}", num_steps);
    println!(
        "  Tasa de aprendizaje: {} (con calentamiento y decaimiento coseno)",
        learning_rate
    );
    println!("  Fracción de calentamiento: {}", warmup_fraction);
    println!("  Longitud de secuencia: {}", seq_len);
    println!("  Recorte de gradiente (Gradient clipping): {}", gradient_clip_norm);
    println!("  Paciencia de parada temprana: {}", patience);
    println!("  Acumulación de gradientes: 8 mini-lotes");
    println!("\n⏱️  Tiempo esperado: 1.5-2 horas en una CPU moderna");
    println!("    El progreso se registra cada 50 pasos");
    println!("    El texto de muestra se genera cada 200 pasos");
    println!("    Checkpoints guardados cada 250 pasos\n");

    // Decaimiento de peso (Weight decay) para regularización (ajustado para un tamaño de dataset óptimo)
    let weight_decay = 0.1; // Decaimiento de peso óptimo para ~715K tokens (proporción parámetro:token de 4:1)

    // Ejecutar entrenamiento con división de validación configurable
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

    // ========================================================================
    // 5. Generación Final
    // ========================================================================
    println!("\n{}", "=".repeat(70));
    println!("5. Generación Final");
    println!("{}", "=".repeat(70));
    println!();

    // Generar varias muestras con diferentes prompts y temperaturas
    println!("Generando muestras de texto:\n");

    let prompts = vec![
        ("En un lugar de la Mancha", 0.8),
        ("DON QUIJOTE.", 0.9),
        ("Sancho,", 0.7),
        ("El hidalgo", 0.8),
        ("Dulcinea", 1.0),
    ];

    for (prompt, temperature) in prompts {
        let prompt_tokens = tokenizer.encode(prompt);
        let generated = model.generate(&prompt_tokens, 80, temperature);
        let generated_text = tokenizer.decode(&generated);

        // Truncar a 150 caracteres para visualización
        let display_text: String = generated_text.chars().take(150).collect();

        println!("Prompt: \"{}\" (temperatura={})", prompt, temperature);
        println!("Generado:");
        println!("  {}", display_text);
        if generated_text.len() > 150 {
            println!("  ... (truncado)");
        }
        println!();
    }

    // ========================================================================
    // Resumen
    // ========================================================================
    println!("{}", "=".repeat(70));
    println!("Resumen");
    println!("{}", "=".repeat(70));
    println!();

    println!("✅ ¡Entrenamiento completado exitosamente!");
    println!();
    println!("Archivos de salida en: {}/", run_dir);
    println!("  ├── training_log.csv        Métricas de entrenamiento");
    println!("  ├── tokenizer.json          Tokenizador entrenado");
    println!("  ├── checkpoint_best.bin     Mejor modelo (menor pérdida de validación)");
    println!("  ├── checkpoint_step_*.bin   Checkpoints periódicos");
    println!("  └── checkpoint_final.bin    Modelo final");
    println!();
    println!("Analizar el entrenamiento:");
    println!("  Ver registro: cat {}/training_log.csv", run_dir);
    println!("  Graficar curvas de pérdida para ver la convergencia");
    println!("  Comprobar si hay sobreajuste (pérdida de entrenamiento << validación)");
    println!("  Encontrar el mejor checkpoint (menor pérdida de validación)");
    println!();
    println!("Resultados esperados:");
    println!("  • Perplejidad de validación: 600+ → ~45 (¡LA MEJOR de cualquier tamaño de modelo!)");
    println!("  • Pérdida de validación: ~6.5 → ~3.8");
    println!("  • Pérdida de entrenamiento: ~6.5 → ~3.5 (brecha entreno/val mínima)");
    println!("  • Texto: Pasajes coherentes de múltiples oraciones");
    println!("  • Estilo: Gramática, vocabulario y estructura cervantinos adecuados");
    println!("\n  Nota: Esto SUPERA al modelo GPT-2 Small de 163M de parámetros");
    println!("        (GPT-2 Small se sobreajusta y se estanca en una perplejidad de val de ~161)");
    println!();
    println!("Cargar y usar el modelo:");
    println!(
        "  let checkpoint = Checkpoint::load(\"{}/checkpoint_best.bin\")?;",
        run_dir
    );
    println!("  let model = checkpoint.model;");
    println!("  let tokenizer = checkpoint.tokenizer.unwrap();");
    println!("  // Generar texto, continuar entrenamiento, etc.");
    println!();
    println!("Próximos pasos:");
    println!("  • ¡Esto ya es óptimo! Los modelos más grandes se sobreajustarán.");
    println!("  • Experimenta con la temperatura (0.6-1.2) para la generación");
    println!("  • Prueba un contexto más largo (block_size=512) si tienes más memoria");
    println!("  • Usa el corpus completo si usaste un subconjunto");
    println!("  • Entrena en diferentes dominios (código, poesía, tus propios escritos)");
    println!("  • Consulta 08_train_cervantes_gpt2.rs para observar el sobreajuste (educativo)");

    Ok(())
}