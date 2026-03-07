//! Demostración de la Arquitectura del Modelo
//!
//! Este ejemplo demuestra la arquitectura del modelo GPT-2:
//! - Creación de modelos de diferentes tamaños
//! - Comprensión del número de parámetros
//! - Forward pass a través del modelo
//! - Inspección de salidas intermedias de las capas
//!
//! Esto muestra **solo la arquitectura** (forward pass). El entrenamiento requeriría
//! backpropagation, optimización y un bucle de entrenamiento (no incluido en la Fase 3).
//!
//! # Uso
//!
//! ```bash
//! cargo run --release --example 03_model_architecture
//! ```
//!
//! # Tiempo de ejecución esperado
//!
//! Menos de 5 segundos

use molineteai::{Config, GPT2};

fn main() {
    println!("\n{}", "=".repeat(70));
    println!("  Demostración de la Arquitectura del Modelo GPT-2");
    println!("{}", "=".repeat(70));

    // ========== Configuraciones del Modelo ==========
    println!("\n{}", "─".repeat(70));
    println!("1. Configuraciones del Modelo");
    println!("{}", "─".repeat(70));

    let vocab_size = 512; // Vocabulario pequeño para demostración

    let configs = vec![
        ("Tiny", Config::tiny(vocab_size)),
        ("Small", Config::small(vocab_size)),
        ("Medium", Config::medium(vocab_size)),
        ("GPT-2 Small", Config::gpt2_small(vocab_size)),
    ];

    println!(
        "\n{:<10} {:<8} {:<8} {:<8} {:<12} {:<12}",
        "Config", "Vocab", "Embd", "Heads", "Layers", "BlockSize"
    );
    println!("{}", "-".repeat(70));

    for (name, config) in &configs {
        println!(
            "{:<10} {:<8} {:<8} {:<8} {:<12} {:<12}",
            name,
            config.vocab_size,
            config.n_embd,
            config.n_heads,
            config.n_layers,
            config.block_size
        );
    }

    // ========== Conteo de Parámetros ==========
    println!("\n{}", "─".repeat(70));
    println!("2. Conteo de Parámetros");
    println!("{}", "─".repeat(70));

    println!("\nCreando modelos y contando parámetros...\n");

    for (name, config) in &configs {
        let model = GPT2::new(config);
        let params = model.count_parameters();

        println!("Modelo {}:", name);
        println!(
            "  Parámetros totales: {:>12} ({:.2}M)",
            params,
            params as f32 / 1_000_000.0
        );

        // Estimación de memoria (4 bytes por parámetro f32)
        let memory_mb = (params * 4) as f32 / 1_000_000.0;
        println!("  Memoria (pesos): {:>10.1} MB", memory_mb);
        println!();
    }

    // ========== Forward Pass ==========
    println!("{}", "─".repeat(70));
    println!("3. Forward Pass");
    println!("{}", "─".repeat(70));

    let config = Config::tiny(vocab_size);
    let model = GPT2::new(&config);

    println!("\nUsando el modelo Tiny para la demostración del forward pass");
    println!("  Tamaño de vocabulario: {}", config.vocab_size);
    println!("  Dimensión de embedding: {}", config.n_embd);
    println!("  Capas: {}", config.n_layers);

    let batch_size = 2;
    let seq_len = 8;

    let tokens = vec![
        vec![1, 2, 3, 4, 5, 6, 7, 8],
        vec![10, 20, 30, 40, 50, 60, 70, 80],
    ];

    println!("\nForma de entrada: [batch={}, seq_len={}]", batch_size, seq_len);
    println!("  Tokens Batch 0: {:?}", tokens[0]);
    println!("  Tokens Batch 1: {:?}", tokens[1]);

    println!("\nEjecutando forward pass...");
    let start = std::time::Instant::now();
    let logits = model.forward(&tokens);
    let elapsed = start.elapsed();

    println!("\nSalida (logits):");
    println!("  Forma: {:?}", logits.shape);
    println!(
        "  Esperado: [batch={}, seq_len={}, vocab_size={}]",
        batch_size, seq_len, vocab_size
    );
    println!("  Tiempo: {:.3}ms", elapsed.as_secs_f64() * 1000.0);

    // ========== Benchmarks de Rendimiento ==========
    println!("\n{}", "─".repeat(70));
    println!("3b. Benchmarks de Rendimiento");
    println!("{}", "─".repeat(70));

    println!("\nMidiendo todos los tamaños de modelo con secuencias de 8 tokens:");
    println!("(Ejecutando múltiples iteraciones para mediciones precisas)\n");

    let benchmark_configs = vec![
        ("Tiny", Config::tiny(vocab_size)),
        ("Small", Config::small(vocab_size)),
        ("Medium", Config::medium(vocab_size)),
        ("GPT-2 Small", Config::gpt2_small(vocab_size)),
    ];

    let single_token = vec![vec![42]];

    for (name, config) in &benchmark_configs {
        let model = GPT2::new(config);

        let _ = model.forward(&tokens);

        let iterations = 10;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = model.forward(&tokens);
        }
        let elapsed = start.elapsed();
        let avg_time = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

        let start_single = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = model.forward(&single_token);
        }
        let elapsed_single = start_single.elapsed();
        let avg_time_single = elapsed_single.as_secs_f64() * 1000.0 / iterations as f64;

        println!("Modelo {}:", name);
        println!("  8 tokens:  {:>8.3} ms/forward", avg_time);
        println!("  1 token:   {:>8.3} ms/forward", avg_time_single);
        println!();
    }

    println!("\nLogits de ejemplo en la posición [0, 0, :]:");
    println!("  (predicciones para el primer token del primer batch)");
    let sample_size = 10.min(vocab_size);
    print!("  Primeros {} valores: [", sample_size);
    for i in 0..sample_size {
        print!("{:.3}", logits.data[i]);
        if i < sample_size - 1 {
            print!(", ");
        }
    }
    println!("]");

    // ========== Desglose de la Arquitectura ==========
    println!("\n{}", "─".repeat(70));
    println!("4. Desglose de la Arquitectura");
    println!("{}", "─".repeat(70));

    println!("\nPara el modelo Tiny:");

    println!("\n  Embedding de Tokens:");
    println!("    Entrada: [batch, seq_len] = [2, 8]");
    println!(
        "    Salida: [batch, seq_len, n_embd] = [2, 8, {}]",
        config.n_embd
    );
    println!(
        "    Parámetros: {} × {} = {}",
        config.vocab_size,
        config.n_embd,
        config.vocab_size * config.n_embd
    );

    println!("\n  Embedding Posicional:");
    println!("    Posiciones: 0..{}", seq_len - 1);
    println!(
        "    Salida: [1, seq_len, n_embd] = [1, 8, {}]",
        config.n_embd
    );
    println!(
        "    Parámetros: {} × {} = {}",
        config.block_size,
        config.n_embd,
        config.block_size * config.n_embd
    );

    println!("\n  Bloque Transformer (×{}):", config.n_layers);
    println!("    Cada bloque contiene:");
    println!("      - Layer Norm 1");
    println!("      - Atención Multi-Cabeza ({} cabezas)", config.n_heads);
    println!(
        "        • Proyecciones Q, K, V: {} → {}",
        config.n_embd,
        config.n_embd * 3
    );
    println!(
        "        • Proyección de salida: {} → {}",
        config.n_embd, config.n_embd
    );
    println!("      - Layer Norm 2");
    println!("      - MLP:");
    println!(
        "        • Expansión: {} → {}",
        config.n_embd,
        config.n_embd * 4
    );
    println!("        • Activación GELU");
    println!(
        "        • Proyección: {} → {}",
        config.n_embd * 4,
        config.n_embd
    );

    println!("\n  Layer Norm Final:");
    println!("    Entrada/Salida: [batch, seq_len, n_embd]");

    println!("\n  LM Head (Proyección de Salida):");
    println!(
        "    Entrada: [batch, seq_len, n_embd] = [2, 8, {}]",
        config.n_embd
    );
    println!(
        "    Salida: [batch, seq_len, vocab_size] = [2, 8, {}]",
        config.vocab_size
    );
    println!(
        "    Parámetros: {} × {} = {}",
        config.n_embd,
        config.vocab_size,
        config.n_embd * config.vocab_size
    );

    // ========== Atención Multi-Cabeza ==========
    println!("\n{}", "─".repeat(70));
    println!("5. Detalles de Atención Multi-Cabeza");
    println!("{}", "─".repeat(70));

    println!("\nAtención de una sola cabeza (modelo Tiny):");
    let tiny_config = Config::tiny(vocab_size);
    let tiny_head_dim = tiny_config.n_embd / tiny_config.n_heads;
    println!(
        "  n_embd={}, n_heads={}, head_dim={}",
        tiny_config.n_embd, tiny_config.n_heads, tiny_head_dim
    );
    println!(
        "  Las {} dimensiones completas atienden como una sola unidad",
        tiny_config.n_embd
    );
    println!("  Simple pero limitado en los patrones que puede aprender");

    println!("\nAtención multi-cabeza (modelo GPT-2 Small):");
    let gpt2_config = Config::gpt2_small(vocab_size);
    let gpt2_head_dim = gpt2_config.n_embd / gpt2_config.n_heads;
    println!(
        "  n_embd={}, n_heads={}, head_dim={}",
        gpt2_config.n_embd, gpt2_config.n_heads, gpt2_head_dim
    );
    println!(
        "  Dividir {} dimensiones en {} cabezas independientes de {} dimensiones cada una",
        gpt2_config.n_embd, gpt2_config.n_heads, gpt2_head_dim
    );
    println!("  Cada cabeza aprende diferentes patrones de atención en paralelo");

    // ========== Atención Causal ==========
    println!("\n{}", "─".repeat(70));
    println!("6. Atención Causal (Autoregresiva)");
    println!("{}", "─".repeat(70));

    println!("\nEn modelado de lenguaje, predecimos el SIGUIENTE token.");
    println!("La posición i no puede ver posiciones i+1, i+2, ... (el futuro)");

    println!("\nCreando máscara causal para seq_len=4:");
    let demo_seq_len = 4;
    let mut mask_data = vec![0.0; demo_seq_len * demo_seq_len];
    for i in 0..demo_seq_len {
        for j in 0..demo_seq_len {
            if j > i {
                mask_data[i * demo_seq_len + j] = 1.0;
            }
        }
    }

    println!("\nMáscara (1 = enmascarado, 0 = visible):");
    println!("  Pos:  0  1  2  3");
    for i in 0..demo_seq_len {
        print!("    {}: [", i);
        for j in 0..demo_seq_len {
            let val = mask_data[i * demo_seq_len + j];
            if val == 0.0 {
                print!("✓  ");
            } else {
                print!("✗  ");
            }
        }
        println!("]  la posición {} puede ver posiciones 0..={}", i, i);
    }

    println!("\nCómo funciona:");
    println!("  1. Calcular puntajes de atención entre todas las posiciones");
    println!("  2. Establecer posiciones futuras (mask=1) en -∞");
    println!("  3. Aplicar softmax: exp(-∞) = 0, sin atención al futuro");
    println!("  4. Cada posición solo atiende a sí misma y al pasado");

    // ========== Resumen ==========
    println!("\n{}", "=".repeat(70));
    println!("  Resumen");
    println!("{}", "=".repeat(70));

    println!("\n✓ Arquitectura GPT-2 implementada desde cero");
    println!("✓ Forward pass funcionando para inferencia");
    println!("✓ Múltiples tamaños de modelo disponibles (tiny → GPT-2 Small)");
    println!("✓ Todos los componentes: embeddings, atención, MLP, layer norm");

    println!("\nDecisiones arquitectónicas clave:");
    println!("  • Self-attention multi-cabeza (operaciones de atención en paralelo)");
    println!("  • Enmascaramiento causal (evita ver tokens futuros)");
    println!("  • Conexiones residuales (ayudan al flujo del gradiente)");
    println!("  • Normalización de capa (estabiliza activaciones)");
    println!("  • Activación GELU (suave y efectiva en práctica)");
    println!("  • Expansión 4× en el MLP (mayor capacidad representacional)");
    println!();
}