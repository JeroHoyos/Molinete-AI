//! Utilidades de Gradientes
//!
//! Este módulo proporciona utilidades para trabajar con gradientes durante el entrenamiento.
//! Estas operaciones son esenciales para la estabilidad y el monitoreo del entrenamiento.
//!
//! ## Componentes
//!
//! - **Cálculo de la Norma del Gradiente**: Medir la magnitud de los gradientes
//! - **Gradient Clipping**: Prevenir la explosión de gradientes mediante escalado
//!
//! ## ¿Por qué Gradient Clipping?
//!
//! Durante el entrenamiento, algunos batches pueden producir gradientes muy grandes
//! que desestabilizan el modelo. El gradient clipping previene esto escalando los
//! gradientes cuando su norma supera un umbral.
//!
//! Sin clipping:
//! ```text
//! Step 1000: Loss = 3.2
//! Step 1001: Loss = 287.5  (¡explosión de gradientes!)
//! Step 1002: Loss = NaN    (entrenamiento falló)
//! ```
//!
//! Con clipping:
//! ```text
//! Step 1000: Loss = 3.2
//! Step 1001: Loss = 3.3  (el gradiente fue recortado)
//! Step 1002: Loss = 3.1  (se recuperó)
//! ```
//!
//! ## Algoritmo
//!
//! ```text
//! norm = √(Σ gradient²)  // Calcular norma L2
//! if norm > max_norm:
//!     gradients *= (max_norm / norm)  // Escalar proporcionalmente
//! ```
//!
//! Esto garantiza que todos los gradientes se escalen por el mismo factor,
//! preservando sus magnitudes relativas mientras se limita la magnitud total de la actualización.
//!
//! ## Ejemplo
//!
//! ```rust,no_run
//! use feste::gradients::{compute_grad_norm, clip_gradients};
//! # use feste::gpt2_trainable::GPT2Gradients;
//!
//! # let grads: GPT2Gradients = todo!();
//! // Calcular la norma del gradiente para monitoreo
//! let norm = compute_grad_norm(&grads);
//! println!("Norma del gradiente: {:.4}", norm);
//!
//! // Recortar si es demasiado grande
//! let mut grads_clipped = grads;
//! clip_gradients(&mut grads_clipped, 1.0);
//! ```

use crate::gpt2_trainable::GPT2Gradients;
use rayon::prelude::*;

/// Calcula la norma L2 de todos los gradientes
///
/// La norma del gradiente es la raíz cuadrada de la suma de todos los valores
/// de gradiente al cuadrado a través de todos los parámetros del modelo.
/// Esto produce un único número que representa la magnitud global de la actualización.
///
/// # Argumentos
///
/// * `grads` - Gradientes de todos los parámetros del modelo
///
/// # Retorna
///
/// La norma L2: √(Σ g²) para todos los valores de gradiente g
///
/// # Rendimiento
///
/// Usa computación paralela con Rayon para mejor rendimiento en CPUs multinúcleo.
/// El cálculo se paraleliza dentro de cada tensor para maximizar el rendimiento.
///
/// # Ejemplo
///
/// ```rust,no_run
/// # use feste::gradients::compute_grad_norm;
/// # use feste::gpt2_trainable::GPT2Gradients;
/// # let grads: GPT2Gradients = todo!();
/// let norm = compute_grad_norm(&grads);
/// if norm > 5.0 {
///     println!("Advertencia: Norma de gradiente grande: {:.2}", norm);
/// }
/// ```
pub fn compute_grad_norm(grads: &GPT2Gradients) -> f32 {
    // Función auxiliar para calcular suma de cuadrados en paralelo
    let sum_sq_parallel = |data: &Vec<f32>| -> f32 { data.par_iter().map(|&val| val * val).sum() };

    let mut sum_sq = 0.0;

    // Embeddings de token y posición
    sum_sq += sum_sq_parallel(&grads.token_embedding.data);
    sum_sq += sum_sq_parallel(&grads.position_embedding.data);

    // Todos los bloques transformer
    for block_grad in &grads.block_grads {
        // LayerNorm 1
        sum_sq += sum_sq_parallel(&block_grad.ln1_gamma.data);
        sum_sq += sum_sq_parallel(&block_grad.ln1_beta.data);

        // Attention
        sum_sq += sum_sq_parallel(&block_grad.attn.q_weight.data);
        sum_sq += sum_sq_parallel(&block_grad.attn.q_bias.data);
        sum_sq += sum_sq_parallel(&block_grad.attn.k_weight.data);
        sum_sq += sum_sq_parallel(&block_grad.attn.k_bias.data);
        sum_sq += sum_sq_parallel(&block_grad.attn.v_weight.data);
        sum_sq += sum_sq_parallel(&block_grad.attn.v_bias.data);
        sum_sq += sum_sq_parallel(&block_grad.attn.out_weight.data);
        sum_sq += sum_sq_parallel(&block_grad.attn.out_bias.data);

        // LayerNorm 2
        sum_sq += sum_sq_parallel(&block_grad.ln2_gamma.data);
        sum_sq += sum_sq_parallel(&block_grad.ln2_beta.data);

        // MLP (red feedforward)
        sum_sq += sum_sq_parallel(&block_grad.mlp.fc1_weight.data);
        sum_sq += sum_sq_parallel(&block_grad.mlp.fc1_bias.data);
        sum_sq += sum_sq_parallel(&block_grad.mlp.fc2_weight.data);
        sum_sq += sum_sq_parallel(&block_grad.mlp.fc2_bias.data);
    }

    // Layer norm final
    sum_sq += sum_sq_parallel(&grads.ln_final_gamma.data);
    sum_sq += sum_sq_parallel(&grads.ln_final_beta.data);

    // Peso de proyección de salida
    sum_sq += sum_sq_parallel(&grads.output_weight.data);

    sum_sq.sqrt()
}

/// Recorta los gradientes a una norma máxima
///
/// Cuando la norma del gradiente supera `max_norm`, todos los gradientes se escalan
/// proporcionalmente para llevar la norma exactamente a `max_norm`. Esto previene
/// la explosión de gradientes mientras preserva la dirección de la actualización.
///
/// # Argumentos
///
/// * `grads` - Gradientes a recortar (modificados en el lugar)
/// * `max_norm` - Norma máxima permitida (típicamente 1.0)
///
/// # Algoritmo
///
/// ```text
/// norm = compute_grad_norm(grads)
/// if norm > max_norm:
///     scale = max_norm / norm
///     for all gradients g:
///         g *= scale
/// ```
///
/// # Rendimiento
///
/// Solo realiza el escalado si es necesario. Usa computación paralela con
/// Rayon para mejor rendimiento en CPUs multinúcleo.
///
/// # Ejemplo
///
/// ```rust,no_run
/// # use feste::gradients::clip_gradients;
/// # use feste::gpt2_trainable::GPT2Gradients;
/// # let mut grads: GPT2Gradients = todo!();
/// // Recortar gradientes a norma 1.0 (práctica estándar)
/// clip_gradients(&mut grads, 1.0);
/// ```
pub fn clip_gradients(grads: &mut GPT2Gradients, max_norm: f32) {
    let norm = compute_grad_norm(grads);

    // Solo recortar si la norma supera el umbral
    if norm > max_norm {
        let scale = max_norm / norm;

        // Función auxiliar para escalar datos del tensor en paralelo
        let scale_parallel = |data: &mut Vec<f32>| {
            data.par_iter_mut().for_each(|val| *val *= scale);
        };

        // Escalar todos los gradientes por el mismo factor

        // Embeddings de token y posición
        scale_parallel(&mut grads.token_embedding.data);
        scale_parallel(&mut grads.position_embedding.data);

        // Todos los bloques transformer
        for block_grad in &mut grads.block_grads {
            // LayerNorm 1
            scale_parallel(&mut block_grad.ln1_gamma.data);
            scale_parallel(&mut block_grad.ln1_beta.data);

            // Attention
            scale_parallel(&mut block_grad.attn.q_weight.data);
            scale_parallel(&mut block_grad.attn.q_bias.data);
            scale_parallel(&mut block_grad.attn.k_weight.data);
            scale_parallel(&mut block_grad.attn.k_bias.data);
            scale_parallel(&mut block_grad.attn.v_weight.data);
            scale_parallel(&mut block_grad.attn.v_bias.data);
            scale_parallel(&mut block_grad.attn.out_weight.data);
            scale_parallel(&mut block_grad.attn.out_bias.data);

            // LayerNorm 2
            scale_parallel(&mut block_grad.ln2_gamma.data);
            scale_parallel(&mut block_grad.ln2_beta.data);

            // MLP (red feedforward)
            scale_parallel(&mut block_grad.mlp.fc1_weight.data);
            scale_parallel(&mut block_grad.mlp.fc1_bias.data);
            scale_parallel(&mut block_grad.mlp.fc2_weight.data);
            scale_parallel(&mut block_grad.mlp.fc2_bias.data);
        }

        // Layer norm final
        scale_parallel(&mut grads.ln_final_gamma.data);
        scale_parallel(&mut grads.ln_final_beta.data);

        // Peso de proyección de salida
        scale_parallel(&mut grads.output_weight.data);
    }
}