//! Mecanismo de Self-Attention
//!
//! La atención es la innovación central de los transformers. Permite que cada
//! posición atienda a todas las posiciones anteriores, aprendiendo relaciones contextuales.
//!
//! ## Atención Escalada por Producto Punto (Scaled Dot-Product Attention)
//!
//! ```text
//! Q, K, V = x @ W_q, x @ W_k, x @ W_v
//! scores = (Q @ K^T) / √d_k
//! attn_weights = softmax(masked_scores)
//! output = attn_weights @ V
//! ```
//!
//! ## ¿Por qué Escalar?
//!
//! Dividimos por √d_k para evitar que los productos punto crezcan demasiado,
//! lo que llevaría a que softmax opere en regiones con gradientes extremadamente pequeños.
//!
//! ## Enmascaramiento Causal
//!
//! Para modelado de lenguaje, enmascaramos posiciones futuras para que cada token
//! solo pueda atenderse a sí mismo y a los tokens anteriores. Esto es crucial
//! para la generación autorregresiva.
//!
//! ## Backward Pass
//!
//! El backward pass a través de la atención involucra:
//! 1. Retropropagación por la proyección de salida
//! 2. Retropropagación por la suma ponderada por atención (V)
//! 3. Retropropagación por softmax (con gradientes por fila)
//! 4. Retropropagación por el producto punto escalado
//! 5. Retropropagación por las proyecciones Q, K, V
//!
//! El backward de softmax es particularmente interesante: debemos tener en cuenta
//! que softmax acopla todos los elementos dentro de cada fila.

use super::dropout::{DropoutCache, TrainableDropout};
use super::linear::{LinearCache, TrainableLinear};
use crate::tensor::Tensor;

/// Self-attention de una sola cabeza
///
/// Esto implementa una cabeza de atención. La atención multi-cabeza ejecutaría
/// múltiples copias de esto en paralelo.
pub struct TrainableSingleHeadAttention {
    pub q_proj: TrainableLinear,
    pub k_proj: TrainableLinear,
    pub v_proj: TrainableLinear,
    pub out_proj: TrainableLinear,
    pub attn_dropout: TrainableDropout,
    pub resid_dropout: TrainableDropout,
    pub n_embd: usize,
}

impl TrainableSingleHeadAttention {
    /// Crea una nueva capa de atención
    ///
    /// # Argumentos
    ///
    /// * `n_embd` - Dimensión del embedding
    /// * `dropout_rate` - Probabilidad de dropout
    /// * `seed` - Semilla aleatoria para inicialización
    pub fn new(n_embd: usize, dropout_rate: f32, seed: u64) -> Self {
        Self {
            q_proj: TrainableLinear::new(n_embd, n_embd, seed),
            k_proj: TrainableLinear::new(n_embd, n_embd, seed + 1),
            v_proj: TrainableLinear::new(n_embd, n_embd, seed + 2),
            out_proj: TrainableLinear::new(n_embd, n_embd, seed + 3),
            attn_dropout: TrainableDropout::new(dropout_rate),
            resid_dropout: TrainableDropout::new(dropout_rate),
            n_embd,
        }
    }

    /// Forward pass: atención por producto punto escalado con enmascaramiento causal
    ///
    /// # Argumentos
    ///
    /// * `x` - Tensor de entrada [seq_len, n_embd]
    ///
    /// # Retorna
    ///
    /// Tupla de (output, cache)
    pub fn forward(&self, x: &Tensor) -> (Tensor, AttentionCache) {
        let seq_len = x.shape[0];

        // Proyección a Q, K, V
        let (q, q_cache) = self.q_proj.forward(x);
        let (k, k_cache) = self.k_proj.forward(x);
        let (v, v_cache) = self.v_proj.forward(x);

        // Atención por producto punto escalado
        let scale = (self.n_embd as f32).sqrt();
        let scores = q.matmul(&k.transpose(-2, -1)).mul_scalar(1.0 / scale);

        // Máscara causal: evitar atender a posiciones futuras
        let mut mask = vec![0.0; seq_len * seq_len];
        for i in 0..seq_len {
            for j in i + 1..seq_len {
                mask[i * seq_len + j] = 1.0;
            }
        }
        let mask_tensor = Tensor::new(mask, vec![seq_len, seq_len]);
        let masked_scores = scores.masked_fill(&mask_tensor, -1e9);

        // Softmax -> pesos de atención
        let attn_weights = masked_scores.softmax(-1);

        // Aplicar dropout a los pesos de atención
        let (attn_weights_dropped, attn_dropout_cache) = self.attn_dropout.forward(&attn_weights);

        // Aplicar atención a los valores
        let attn_out = attn_weights_dropped.matmul(&v);

        // Proyección de salida
        let (y_proj, out_cache) = self.out_proj.forward(&attn_out);

        // Aplicar dropout residual
        let (y, resid_dropout_cache) = self.resid_dropout.forward(&y_proj);

        let cache = AttentionCache {
            x: x.clone(),
            q,
            k,
            v,
            attn_weights,
            #[allow(dead_code)]
            attn_out,
            q_cache,
            k_cache,
            v_cache,
            out_cache,
            attn_dropout_cache,
            resid_dropout_cache,
        };

        (y, cache)
    }

    /// Backward pass a través de la atención
    ///
    /// # Argumentos
    ///
    /// * `grad_out` - Gradiente de la siguiente capa
    /// * `cache` - Valores almacenados del forward pass
    ///
    /// # Retorna
    ///
    /// Gradientes para todos los parámetros y la entrada
    pub fn backward(&self, grad_out: &Tensor, cache: &AttentionCache) -> AttentionGradients {
        let seq_len = cache.x.shape[0];
        let scale = (self.n_embd as f32).sqrt();

        // Retropropagación por dropout residual
        let grad_y_proj = self
            .resid_dropout
            .backward(grad_out, &cache.resid_dropout_cache);

        // Retropropagación por proyección de salida
        let out_grads = self.out_proj.backward(&grad_y_proj, &cache.out_cache);

        // Retropropagación por atención: grad_v = attn_weights^T @ grad_attn_out
        let grad_v = cache.attn_weights.transpose(-2, -1).matmul(&out_grads.x);

        // grad_attn_weights = grad_attn_out @ v^T
        let grad_attn_weights_dropped = out_grads.x.matmul(&cache.v.transpose(-2, -1));

        // Retropropagación por dropout de atención
        let grad_attn_weights = self
            .attn_dropout
            .backward(&grad_attn_weights_dropped, &cache.attn_dropout_cache);

        // Retropropagación por softmax (por fila)
        // gradiente softmax: grad_scores = attn * (grad_attn - sum(grad_attn * attn))
        let mut grad_scores_data = Vec::new();
        for i in 0..seq_len {
            let start = i * seq_len;
            let end = start + seq_len;
            let attn_row = &cache.attn_weights.data[start..end];
            let grad_attn_row = &grad_attn_weights.data[start..end];

            let dot_product: f32 = attn_row
                .iter()
                .zip(grad_attn_row.iter())
                .map(|(a, g)| a * g)
                .sum();

            for j in 0..seq_len {
                let grad_score = attn_row[j] * (grad_attn_row[j] - dot_product);
                grad_scores_data.push(grad_score);
            }
        }
        let grad_scores = Tensor::new(grad_scores_data, vec![seq_len, seq_len]);

        // Retropropagación por Q @ K^T escalado
        let grad_q = grad_scores.matmul(&cache.k).mul_scalar(1.0 / scale);
        let grad_k = grad_scores
            .transpose(-2, -1)
            .matmul(&cache.q)
            .mul_scalar(1.0 / scale);

        // Retropropagación por proyecciones Q, K, V
        let q_grads = self.q_proj.backward(&grad_q, &cache.q_cache);
        let k_grads = self.k_proj.backward(&grad_k, &cache.k_cache);
        let v_grads = self.v_proj.backward(&grad_v, &cache.v_cache);

        // Acumular gradientes hacia la entrada (Q, K, V conectan a la misma entrada)
        let mut grad_x_data = vec![0.0; cache.x.data.len()];
        for (i, grad_x_val) in grad_x_data.iter_mut().enumerate() {
            *grad_x_val = q_grads.x.data[i] + k_grads.x.data[i] + v_grads.x.data[i];
        }
        let grad_x = Tensor::new(grad_x_data, cache.x.shape.clone());

        AttentionGradients {
            q_weight: q_grads.weight,
            q_bias: q_grads.bias,
            k_weight: k_grads.weight,
            k_bias: k_grads.bias,
            v_weight: v_grads.weight,
            v_bias: v_grads.bias,
            out_weight: out_grads.weight,
            out_bias: out_grads.bias,
            x: grad_x,
        }
    }
}

/// Cache para el backward pass de atención
pub struct AttentionCache {
    pub x: Tensor,
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub attn_weights: Tensor,
    #[allow(dead_code)]
    pub attn_out: Tensor,
    pub q_cache: LinearCache,
    pub k_cache: LinearCache,
    pub v_cache: LinearCache,
    pub out_cache: LinearCache,
    pub attn_dropout_cache: DropoutCache,
    pub resid_dropout_cache: DropoutCache,
}

/// Gradientes de la atención
pub struct AttentionGradients {
    pub q_weight: Tensor,
    pub q_bias: Tensor,
    pub k_weight: Tensor,
    pub k_bias: Tensor,
    pub v_weight: Tensor,
    pub v_bias: Tensor,
    pub out_weight: Tensor,
    pub out_bias: Tensor,
    pub x: Tensor,
}