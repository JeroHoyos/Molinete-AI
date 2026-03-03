//! Entrenar tokenizadores BPE con múltiples tamaños de vocabulario
//!
//! Este ejemplo demuestra:
//! - Cargar datos de entrenamiento (obras completas de Shakespeare)
//! - Entrenar tokenizadores con distintos tamaños de vocabulario
//! - Analizar la composición del vocabulario y las tasas de compresión
//! - Guardar los tokenizadores y el análisis en directorios de salida con marca de tiempo
//!
//! La salida se escribe en: `data/example_tokenizer_<timestamp>/`
//!
//! # Uso
//!
//! ```bash
//! cargo run --release --example 01_train_tokenizers
//! ```
//!
//! # Tiempo estimado de ejecución
//!
//! Varía significativamente según el hardware (normalmente 2–10 minutos en total).
//! Los tamaños de vocabulario 1536 y 20534 son los que más tardan.
//!
//! # Requisitos previos
//!
//! Descarga primero el corpus de Cervantes:
//! ```bash
//! curl -o cervantes.txt https://www.gutenberg.org/cache/epub/2000/pg2000.txt
//! ```

use molineteai::BPETokenizer;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(70));
    println!("  Entrenamiento de Tokenizador BPE");
    println!("{}", "=".repeat(70));

    // Crear un directorio de salida con marca de tiempo
    // Utiliza la marca de tiempo Unix para un marcado de tiempo simple y sin dependencias
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let run_dir = format!("data/example_tokenizer_{}", timestamp);
    fs::create_dir_all(&run_dir)?;

    println!("\nDirectorio de salida: {}\n", run_dir);

    // Load Cervantes corpus
    // Espera "cervantes.txt" en el directorio de trabajo actual (normalmente la raíz del proyecto)
    println!("Cargando datos de entrenamiento...");
    let text = match fs::read_to_string("cervantes.txt") {
        Ok(t) => t,
        Err(_) => {
            eprintln!("\nError: cervantes.txt no encontrado en el directorio actual.");
            eprintln!("Por favor, ejecuta el programa desde la raíz del proyecto y descarga el corpus:");
            eprintln!("  curl -o cervantes.txt https://www.gutenberg.org/cache/epub/2000/pg2000.txt\n");
            std::process::exit(1);
        }
    };

    println!(
        "Corpus Cargado: {} bytes ({:.2} MB)\n",
        text.len(),
        text.len() as f64 / 1_000_000.0
    );

    // Definir tamaños de vocabulario a entrenar
    // Coinciden con los usados en los ejemplos:
    // 256   = solo nivel byte (sin merges) - línea base
    // 512   = modelo muy pequeño
    // 1024  = modelo pequeño
    // 1536  = modelo mediano
    // 20534 = modelo grande
    let vocab_sizes = vec![256, 512, 1024, 1536, 20534];

    let mut summary_data = Vec::new();

    // Entrenar tokenizadores en cada tamaño de vocabulario
    for &vocab_size in &vocab_sizes {
        println!("{}", "=".repeat(70));
        println!("Entrenando tokenizador con vocab_size = {}", vocab_size);
        println!("{}", "=".repeat(70));

        let start = SystemTime::now();

        // Crear y entrenar un tokenizador
        let mut tokenizer = BPETokenizer::new(vocab_size);
        tokenizer.train(&text, vocab_size);

        let duration = start.elapsed()?;

        // Codifique el corpus completo para medir la compresión
        println!("Codificando de corpus completo para medir la compresión...");
        let encode_start = SystemTime::now();
        let encoded = tokenizer.encode(&text);
        let encode_duration = encode_start.elapsed()?;

        let compression_ratio = text.len() as f64 / encoded.len() as f64;
        let bytes_per_token = 1.0 / compression_ratio;

        // Display results
        println!("\nResultados:");
        println!("  Tiempo de entrenamiento: {:.2}s", duration.as_secs_f64());
        println!("  Tiempo de codificación: {:.2}s", encode_duration.as_secs_f64());
        println!("  Tamaño del vocabulario: {}", tokenizer.vocab_size());
        println!("  Tamaño original: {} bytes", text.len());
        println!("  Longitud codificada: {} tokens", encoded.len());
        println!("  Ratio de compresión: {:.2}x", compression_ratio);
        println!("  Bytes por token: {:.2}", bytes_per_token);

        // Prueba encode/decode round-trip
        println!("\nProbando codificación round-trip...");
        let test_text = "En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor.";
        let test_encoded = tokenizer.encode(test_text);
        let test_decoded = tokenizer.decode(&test_encoded);

        if test_text == test_decoded {
            println!("  ✓ Prueba round-trip SUPERADA");
        } else {
            println!("  ✗ Prueba round-trip FALLIDA");
            println!("    Original: {}", test_text);
            println!("    Decodificado: {}", test_decoded);
            return Err("La prueba round-trip falló".into());
        }

        // Guardar el tokenizador entrenado
        let save_path = format!("{}/tokenizer_{}.json", run_dir, vocab_size);
        tokenizer.save(&save_path)?;
        println!("\nGuardado en: {}", save_path);

        // Muestra de análisis de vocabulario para vocabularios grandes
        if vocab_size > 256 {
            tokenizer.analyze_vocabulary(&text);
        }

        // Guardar datos para el resumen
        summary_data.push((
            vocab_size,
            duration.as_secs_f64(),
            encode_duration.as_secs_f64(),
            encoded.len(),
            compression_ratio,
        ));

        println!();
    }

    // Write summary file
    println!("{}", "=".repeat(70));
    println!("Generando Resumen...");
    println!("{}", "=".repeat(70));

    let summary_path = format!("{}/summary.txt", run_dir);
    let mut summary = String::new();

    summary.push_str(&format!("{}\n", "=".repeat(70)));
    summary.push_str("  Resumen del Entrenamiento de Tokenizadores\n");
    summary.push_str(&format!("{}\n\n", "=".repeat(70)));

    summary.push_str(&format!(
        "Corpus: cervantes.txt ({} bytes, {:.2} MB)\n",
        text.len(),
        text.len() as f64 / 1_000_000.0
    ));
    summary.push_str(&format!("Tamaños de vocabulario entrenados: {:?}\n\n", vocab_sizes));

    summary.push_str("Resultados del Entrenamiento:\n");
    summary.push_str(&format!("{}\n", "-".repeat(70)));
    summary.push_str(&format!(
        "{:<10} {:>12} {:>12} {:>12} {:>12}\n",
        "Vocab", "Train(s)", "Encode(s)", "Tokens", "Compress"
    ));
    summary.push_str(&format!("{}\n", "-".repeat(70)));

    for (vocab_size, train_time, encode_time, token_count, ratio) in &summary_data {
        summary.push_str(&format!(
            "{:<10} {:>12.2} {:>12.2} {:>12} {:>11.2}x\n",
            vocab_size, train_time, encode_time, token_count, ratio
        ));
    }

    summary.push_str(&format!("\n{}\n", "-".repeat(70)));

    summary.push_str("\nObservaciones:\n\n");
    summary.push_str("1. Ratio de Compresión: Los vocabularios más grandes logran mejor compresión\n");
    summary.push_str("   (menos tokens necesarios para representar el mismo texto)\n\n");
    summary.push_str("2. Tiempo de Entrenamiento: Aumenta con el tamaño del vocabulario debido a más fusiones\n");
    summary.push_str("   (pero usa optimización de muestreo para vocabularios muy grandes)\n\n");
    summary.push_str("3. Compensaciones:\n");
    summary.push_str("   - Vocabulario más grande = mejor compresión = secuencias más cortas\n");
    summary.push_str("   - Shorter sequences = faster training and inference for the model\n");
    summary.push_str("   - Pero: tablas de embeddings más grandes y más parámetros que aprender.\n\n");

    summary.push_str("Tamaños de vocabulario comunes en la práctica:\n");
    summary.push_str("  - GPT-2: 50,257 tokens\n");
    summary.push_str("  - GPT-3: 50,257 tokens\n");
    summary.push_str("  - Modelos educativos: 512-5000 tokens\n\n");

    summary.push_str(&format!("{}\n", "=".repeat(70)));

    fs::write(&summary_path, summary)?;
    println!("\nResumen escrito en: {}", summary_path);

    println!("\n{}", "=".repeat(70));
    println!("  ¡Entrenamiento Completo!");
    println!("{}", "=".repeat(70));
    println!("\nTodos los tokenizadores fueron guardados en: {}/", run_dir);
    println!(
        "\nPara usar un tokenizador en tu código:\n  \
        let tokenizer = BPETokenizer::load(\"{}tokenizer_1024.json\")?;",
        run_dir
    );
    println!();

    Ok(())
}
