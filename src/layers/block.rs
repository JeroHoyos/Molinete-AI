//! Bloque Transformer
//!
//! Un bloque transformer es el componente fundamental de los modelos GPT.
//! Combina capas de atención y feedforward con conexiones residuales
//! y normalización por capas (Layer Normalization).
//!
//! ## Arquitectura
//!
//! ```text
//! x → LayerNorm → Attention → (+) → LayerNorm → MLP → (+) → output
//! │                             ↑                        ↑
//! └─────────────────────────────┘                        │
//! └──────────────────────────────────────────────────────┘
//! ```
//!
//! ## Pre-Norm vs Post-Norm
//!
//! Usamos **pre-norm** (LayerNorm antes de las subcapas) en lugar de post-norm:
//! - Entrenamiento más estable
//! - Mejor flujo de gradientes
//! - Estándar en transformers modernos (GPT-2, GPT-3)
//!
//! ## Conexiones Residuales
//!
//! Las conexiones residuales son críticas para entrenar redes profundas:
//! - Permiten que los gradientes fluyan directamente hacia atrás
//! - Previenen el desvanecimiento del gradiente
//! - Hacen posible entrenar modelos muy profundos (100+ capas)
//!
//! ## Backward Pass
//!
//! El backward pass a través de conexiones residuales requiere una
//! acumulación cuidadosa de gradientes.
//! En cada conexión residual, los gradientes se dividen en dos caminos
//! que luego deben sumarse.

use super::attention::{AttentionCache, AttentionGradients, TrainableSingleHeadAttention};
use super::layer_norm::{LayerNormCache, TrainableLayerNorm};
use super::mlp::{MLPCache, MLPGradients, TrainableMLP};
use crate::tensor::Tensor;

/// Bloque transformer que combina atención y MLP con conexiones residuales
pub struct TrainableTransformerBlock {
    pub ln1: TrainableLayerNorm,
    pub attn: TrainableSingleHeadAttention,
    pub ln2: TrainableLayerNorm,
    pub mlp: TrainableMLP,
}

impl TrainableTransformerBlock {
    /// Crea un nuevo bloque transformer
    ///
    /// # Argumentos
    ///
    /// * `n_embd` - Dimensión del embedding
    /// * `dropout_rate` - Probabilidad de dropout
    /// * `seed` - Semilla aleatoria para inicialización
    pub fn new(n_embd: usize, dropout_rate: f32, seed: u64) -> Self {
        Self {
            ln1: TrainableLayerNorm::new(n_embd),
            attn: TrainableSingleHeadAttention::new(n_embd, dropout_rate, seed),
            ln2: TrainableLayerNorm::new(n_embd),
            mlp: TrainableMLP::new(n_embd, dropout_rate, seed + 1000),
        }
    }

    /// Forward pass: atención + MLP con conexiones residuales
    ///
    /// # Argumentos
    ///
    /// * `x` - Tensor de entrada [seq_len, n_embd]
    ///
    /// # Retorna
    ///
    /// Tupla de (output, cache)
    pub fn forward(&self, x: &Tensor) -> (Tensor, BlockCache) {
        // Primer sub-bloque: LayerNorm → Attention → Residual
        let (ln1_out, ln1_cache) = self.ln1.forward(x);
        let (attn_out, attn_cache) = self.attn.forward(&ln1_out);
        let x_after_attn = x.add(&attn_out); // Conexión residual

        // Segundo sub-bloque: LayerNorm → MLP → Residual
        let (ln2_out, ln2_cache) = self.ln2.forward(&x_after_attn);
        let (mlp_out, mlp_cache) = self.mlp.forward(&ln2_out);
        let y = x_after_attn.add(&mlp_out); // Conexión residual

        let cache = BlockCache {
            #[allow(dead_code)]
            x: x.clone(),
            ln1_cache,
            attn_cache,
            #[allow(dead_code)]
            x_after_attn,
            ln2_cache,
            mlp_cache,
        };

        (y, cache)
    }

    /// Backward pass a través del bloque transformer
    ///
    /// # Argumentos
    ///
    /// * `grad_out` - Gradiente de la siguiente capa
    /// * `cache` - Valores almacenados del forward pass
    ///
    /// # Retorna
    ///
    /// Gradientes para todos los parámetros y la entrada
    pub fn backward(&self, grad_out: &Tensor, cache: &BlockCache) -> BlockGradients {
        // Retropropagación por la segunda conexión residual
        // El gradiente fluye tanto hacia el camino del MLP como directamente
        // hacia la primera conexión residual
        let grad_mlp_out = grad_out.clone();
        let mut grad_x_after_attn = grad_out.clone();

        // Retropropagación por el camino del MLP
        let mlp_grads = self.mlp.backward(&grad_mlp_out, &cache.mlp_cache);
        let ln2_grads = self.ln2.backward(&mlp_grads.x, &cache.ln2_cache);

        // Acumular gradiente proveniente del camino LN2
        for i in 0..grad_x_after_attn.data.len() {
            grad_x_after_attn.data[i] += ln2_grads.x.data[i];
        }

        // Retropropagación por la primera conexión residual
        let grad_attn_out = grad_x_after_attn.clone();
        let mut grad_x = grad_x_after_attn;

        // Retropropagación por el camino de atención
        let attn_grads = self.attn.backward(&grad_attn_out, &cache.attn_cache);
        let ln1_grads = self.ln1.backward(&attn_grads.x, &cache.ln1_cache);

        // Acumular gradiente proveniente del camino LN1
        for i in 0..grad_x.data.len() {
            grad_x.data[i] += ln1_grads.x.data[i];
        }

        BlockGradients {
            ln1_gamma: ln1_grads.gamma,
            ln1_beta: ln1_grads.beta,
            attn: attn_grads,
            ln2_gamma: ln2_grads.gamma,
            ln2_beta: ln2_grads.beta,
            mlp: mlp_grads,
            x: grad_x,
        }
    }
}

/// Cache para el backward pass del bloque transformer
pub struct BlockCache {
    #[allow(dead_code)]
    pub x: Tensor,
    pub ln1_cache: LayerNormCache,
    pub attn_cache: AttentionCache,
    #[allow(dead_code)]
    pub x_after_attn: Tensor,
    pub ln2_cache: LayerNormCache,
    pub mlp_cache: MLPCache,
}

/// Gradientes del bloque transformer
pub struct BlockGradients {
    pub ln1_gamma: Tensor,
    pub ln1_beta: Tensor,
    pub attn: AttentionGradients,
    pub ln2_gamma: Tensor,
    pub ln2_beta: Tensor,
    pub mlp: MLPGradients,
    pub x: Tensor,
}