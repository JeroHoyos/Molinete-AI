//! Perceptrón Multicapa (MLP)
//!
//! El MLP es una red feedforward de dos capas usada en cada bloque transformer.
//! Proporciona la capacidad del modelo para aprender transformaciones complejas.
//!
//! ## Arquitectura
//!
//! ```text
//! x → Linear1 → GELU → Linear2 → y
//! ```
//!
//! ## Factor de Expansión
//!
//! GPT-2 usa una expansión de 4×:
//! - Entrada: n_embd
//! - Oculta: n_embd × 4
//! - Salida: n_embd
//!
//! Este patrón de expansión y luego compresión es crucial para la capacidad del modelo.
//!
//! ## ¿Por qué 4x?
//!
//! La expansión 4× está determinada empíricamente:
//! - Proporciona suficiente capacidad para transformaciones complejas
//! - No es tan grande como para dominar el número de parámetros
//! - Es estándar en muchas arquitecturas transformer

use super::activation::{gelu_backward, gelu_forward};
use super::dropout::{DropoutCache, TrainableDropout};
use super::linear::{LinearCache, TrainableLinear};
use crate::tensor::Tensor;

/// MLP (red feedforward) con activación GELU
pub struct TrainableMLP {
    pub fc1: TrainableLinear,
    pub fc2: TrainableLinear,
    pub resid_dropout: TrainableDropout,
}

impl TrainableMLP {
    /// Crea un nuevo MLP con expansión 4x
    ///
    /// # Argumentos
    ///
    /// * `n_embd` - Dimensión del embedding
    /// * `dropout_rate` - Probabilidad de dropout
    /// * `seed` - Semilla aleatoria para inicialización
    pub fn new(n_embd: usize, dropout_rate: f32, seed: u64) -> Self {
        let hidden = n_embd * 4; // GPT-2 usa expansión 4x
        Self {
            fc1: TrainableLinear::new(n_embd, hidden, seed),
            fc2: TrainableLinear::new(hidden, n_embd, seed + 1000),
            resid_dropout: TrainableDropout::new(dropout_rate),
        }
    }

    /// Paso forward: x → fc1 → GELU → fc2
    ///
    /// # Argumentos
    ///
    /// * `x` - Tensor de entrada [seq_len, n_embd]
    ///
    /// # Retorna
    ///
    /// Tupla de (output, cache)
    pub fn forward(&self, x: &Tensor) -> (Tensor, MLPCache) {
        let (h, fc1_cache) = self.fc1.forward(x);
        let h_activated = gelu_forward(&h);
        let (y_proj, fc2_cache) = self.fc2.forward(&h_activated);

        // Aplicar dropout residual
        let (y, resid_dropout_cache) = self.resid_dropout.forward(&y_proj);

        let cache = MLPCache {
            fc1_cache,
            h, // Guardar pre-activación para GELU backward
            #[allow(dead_code)]
            h_activated,
            fc2_cache,
            resid_dropout_cache,
        };

        (y, cache)
    }

    /// Paso backward a través del MLP
    ///
    /// Usa la regla de la cadena a través de fc2, GELU y fc1
    ///
    /// # Argumentos
    ///
    /// * `grad_out` - Gradiente desde la siguiente capa
    /// * `cache` - Valores almacenados del paso forward
    ///
    /// # Retorna
    ///
    /// Gradientes para todos los parámetros y la entrada
    pub fn backward(&self, grad_out: &Tensor, cache: &MLPCache) -> MLPGradients {
        // Retropropagar a través del dropout residual
        let grad_y_proj = self
            .resid_dropout
            .backward(grad_out, &cache.resid_dropout_cache);

        // Retropropagar a través de fc2
        let fc2_grads = self.fc2.backward(&grad_y_proj, &cache.fc2_cache);

        // Retropropagar a través de GELU
        let grad_h = gelu_backward(&fc2_grads.x, &cache.h);

        // Retropropagar a través de fc1
        let fc1_grads = self.fc1.backward(&grad_h, &cache.fc1_cache);

        MLPGradients {
            fc1_weight: fc1_grads.weight,
            fc1_bias: fc1_grads.bias,
            fc2_weight: fc2_grads.weight,
            fc2_bias: fc2_grads.bias,
            x: fc1_grads.x,
        }
    }
}

/// Caché para el paso backward del MLP
pub struct MLPCache {
    pub fc1_cache: LinearCache,
    pub h: Tensor, // Pre-activación (necesaria para GELU backward)
    #[allow(dead_code)]
    pub h_activated: Tensor,
    pub fc2_cache: LinearCache,
    pub resid_dropout_cache: DropoutCache,
}

/// Gradientes para el MLP
pub struct MLPGradients {
    pub fc1_weight: Tensor,
    pub fc1_bias: Tensor,
    pub fc2_weight: Tensor,
    pub fc2_bias: Tensor,
    pub x: Tensor,
}