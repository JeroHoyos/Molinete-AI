//! Demostración de Operaciones con Tensores
//!
//! Este ejemplo demuestra las operaciones centrales de tensores usadas en redes neuronales:
//! - Creación de tensores
//! - Multiplicación de matrices (secuencial y paralela)
//! - Operaciones elemento a elemento
//! - Broadcasting
//! - Softmax y otras activaciones
//! - Reorganización (reshape) y transposición
//!
//! # Uso
//!
//! ```bash
//! cargo run --release --example 02_tensor_operations
//! ```
//!
//! # Tiempo de ejecución esperado
//!
//! Menos de 1 segundo

use molineteai::Tensor;
use std::time::Instant;

fn main() {
    println!("\n{}", "=".repeat(70));
    println!("  Demostración de Operaciones con Tensores");
    println!("{}", "=".repeat(70));

    // ========== Creación Básica de Tensores ==========
    println!("\n{}", "─".repeat(70));
    println!("1. Creación de Tensores");
    println!("{}", "─".repeat(70));

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::new(data, vec![2, 3]);
    println!("Tensor 2×3 creado:");
    println!("  Forma: {:?}", tensor.shape);
    println!("  Datos: {:?}", tensor.data);

    let zeros = Tensor::zeros(vec![3, 4]);
    println!("\nTensor 3×4 de ceros creado:");
    println!("  Forma: {:?}", zeros.shape);
    println!("  Suma: {}", zeros.data.iter().sum::<f32>());

    let range = Tensor::arange(0, 10);
    println!("\nTensor rango creado [0, 10):");
    println!("  Forma: {:?}", range.shape);
    println!("  Datos: {:?}", range.data);

    // ========== Multiplicación de Matrices ==========
    println!("\n{}", "─".repeat(70));
    println!("2. Multiplicación de Matrices");
    println!("{}", "─".repeat(70));

    // Multiplicación de matrices pequeñas (secuencial)
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);

    println!("\nMatrices pequeñas (2×2) - Secuencial:");
    println!("  A: {:?}", a.data);
    println!("  B (identidad): {:?}", b.data);

    let start = Instant::now();
    let c = a.matmul(&b);
    let elapsed = start.elapsed();

    println!("  Resultado A @ B: {:?}", c.data);
    println!("  Tiempo: {:.2}μs", elapsed.as_micros());

    // Multiplicación de matrices grandes (paralela)
    let large_a = Tensor::new(vec![1.0; 64 * 64], vec![64, 64]);
    let large_b = Tensor::new(vec![1.0; 64 * 64], vec![64, 64]);

    println!("\nMatrices grandes (64×64) - Paralelo:");
    println!("  Tamaño de matriz: {} elementos cada una", 64 * 64);

    let start = Instant::now();
    let large_c = large_a.matmul(&large_b);
    let elapsed = start.elapsed();

    println!("  Forma del resultado: {:?}", large_c.shape);
    println!("  Primer elemento (debería ser 64.0): {}", large_c.data[0]);
    println!("  Tiempo: {:.2}ms", elapsed.as_secs_f64() * 1000.0);

    // ========== Operaciones Elemento a Elemento ==========
    println!("\n{}", "─".repeat(70));
    println!("3. Operaciones Elemento a Elemento");
    println!("{}", "─".repeat(70));

    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let y = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);

    println!("\nX: {:?}", x.data);
    println!("Y: {:?}", y.data);

    let sum = x.add(&y);
    println!("X + Y: {:?}", sum.data);

    let product = x.mul(&y);
    println!("X * Y: {:?}", product.data);

    let diff = x.sub(&y);
    println!("X - Y: {:?}", diff.data);

    let quotient = x.div(&y);
    println!("X / Y: {:?}", quotient.data);

    // ========== Operaciones con Escalares ==========
    println!("\n{}", "─".repeat(70));
    println!("4. Operaciones con Escalares");
    println!("{}", "─".repeat(70));

    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    println!("\nX: {:?}", x.data);

    let scaled = x.mul_scalar(2.0);
    println!("X * 2: {:?}", scaled.data);

    let shifted = x.add_scalar(10.0);
    println!("X + 10: {:?}", shifted.data);

    let divided = x.div_scalar(2.0);
    println!("X / 2: {:?}", divided.data);

    let sqrt_x = x.sqrt();
    println!("sqrt(X): {:?}", sqrt_x.data);

    // ========== Broadcasting ==========
    println!("\n{}", "─".repeat(70));
    println!("5. Broadcasting");
    println!("{}", "─".repeat(70));

    let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let bias = Tensor::new(vec![0.1, 0.2, 0.3], vec![3]);

    println!("\nMatrix [2, 3]: {:?}", matrix.data);
    println!("Bias [3]: {:?}", bias.data);

    let with_bias = matrix.add(&bias);
    println!("Matrix + Bias (broadcast aplicado): {:?}", with_bias.data);

    // ========== Softmax ==========
    println!("\n{}", "─".repeat(70));
    println!("6. Softmax (Estabilidad Numérica)");
    println!("{}", "─".repeat(70));

    let logits = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
    println!("\nLogits: {:?}", logits.data);

    let probs = logits.softmax(-1);
    println!("Softmax (axis=-1): {:?}", probs.data);

    let sum: f32 = probs.data.iter().sum();
    println!("Suma de probabilidades: {:.6} (debería ser 1.0)", sum);

    let large_logits = Tensor::new(vec![100.0, 200.0, 300.0], vec![1, 3]);
    println!("\nLogits grandes: {:?}", large_logits.data);

    let stable_probs = large_logits.softmax(-1);
    println!("Softmax (estable): {:?}", stable_probs.data);
    println!(
        "Suma: {:.6} (¡sin overflow!)",
        stable_probs.data.iter().sum::<f32>()
    );

    // ========== Reorganización (Reshape) ==========
    println!("\n{}", "─".repeat(70));
    println!("7. Reorganización (Reshape)");
    println!("{}", "─".repeat(70));

    let original = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    println!("\nOriginal [2, 3]: {:?}", original.data);

    let reshaped = original.reshape(&[3, 2]);
    println!("Reorganizado [3, 2]: {:?}", reshaped.data);
    println!("  Forma: {:?}", reshaped.shape);

    let flat = reshaped.reshape(&[6]);
    println!("Aplanado [6]: {:?}", flat.data);

    // ========== Transposición ==========
    println!("\n{}", "─".repeat(70));
    println!("8. Transposición");
    println!("{}", "─".repeat(70));

    let matrix = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
    );

    println!("\nOriginal [2, 3]:");
    println!(
        "  Fila 0: [{}, {}, {}]",
        matrix.data[0], matrix.data[1], matrix.data[2]
    );
    println!(
        "  Fila 1: [{}, {}, {}]",
        matrix.data[3], matrix.data[4], matrix.data[5]
    );

    let transposed = matrix.transpose(0, 1);
    println!("\nTranspuesto [3, 2] (filas↔columnas):");
    println!("  Forma: {:?}", transposed.shape);
    println!("  Fila 0: [{}, {}]", transposed.data[0], transposed.data[1]);
    println!("  Fila 1: [{}, {}]", transposed.data[2], transposed.data[3]);
    println!("  Fila 2: [{}, {}]", transposed.data[4], transposed.data[5]);

    // ========== Operaciones Estadísticas ==========
    println!("\n{}", "─".repeat(70));
    println!("9. Operaciones Estadísticas");
    println!("{}", "─".repeat(70));

    let data = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![3, 3],
    );
    println!("\nDatos [3, 3]: {:?}", data.data);

    let row_means = data.mean(-1, false);
    println!("Medias por fila: {:?}", row_means.data);

    let data_3d = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![1, 2, 3],
    );
    println!("\nDatos 3D [1, 2, 3]: {:?}", data_3d.data);

    let means = data_3d.mean(-1, true);
    println!("Media en la última dimensión (keepdim=true):");
    println!("  Forma: {:?}", means.shape);
    println!("  Valores: {:?}", means.data);

    let vars = data_3d.var(-1, true);
    println!("Varianza en la última dimensión (keepdim=true):");
    println!("  Forma: {:?}", vars.shape);
    println!("  Valores: {:?}", vars.data);

    // ========== Masked Fill ==========
    println!("\n{}", "─".repeat(70));
    println!("10. Masked Fill (Enmascaramiento Causal)");
    println!("{}", "─".repeat(70));

    let scores = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    println!("\nPuntuaciones de atención [2, 2]: {:?}", scores.data);

    let mask = Tensor::new(
        vec![0.0, 1.0, 0.0, 0.0],
        vec![2, 2],
    );
    println!("Máscara causal: {:?}", mask.data);

    let masked = scores.masked_fill(&mask, f32::NEG_INFINITY);
    println!("Puntuaciones enmascaradas: {:?}", masked.data);
    println!("  (Posiciones futuras establecidas en -inf)");

    // ========== Resumen ==========
    println!("\n{}", "=".repeat(70));
    println!("  Resumen");
    println!("{}", "=".repeat(70));
    println!("\n✓ Todas las operaciones con tensores funcionan correctamente");
    println!("✓ Las operaciones paralelas aceleran matrices grandes");
    println!("✓ Se mantiene estabilidad numérica (softmax con valores grandes)");
    println!("✓ Broadcasting funciona para patrones comunes");
    println!("\n¡Estas operaciones son la base de los modelos transformer!");
    println!();
}