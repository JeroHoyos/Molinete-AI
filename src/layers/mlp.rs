//! Perceptrón Multicapa (MLP)
//!
//! El MLP es una red prealimentada (feedforward) de dos capas que se usa en cada bloque transformer.
//! Proporciona la capacidad del modelo para aprender transformaciones complejas.
//!
//! ## Arquitectura
//!
//! ```text
//! x → Lineal1 → GELU → Lineal2 → y
//! ```
//!
//! ## Factor de Expansión
//!
//! GPT-2 usa una expansión de 4×:
//! - Entrada: n_embd
//! - Oculta: n_embd × 4
//! - Salida: n_embd
//!
//! Este patrón de expansión y posterior compresión es crucial para la capacidad del modelo.
//!
//! ## ¿Por qué 4x?
//!
//! La expansión de 4× se determina empíricamente:
//! - Proporciona suficiente capacidad para transformaciones complejas
//! - No es tan grande como para dominar el recuento de parámetros
//! - Es un estándar en muchas arquitecturas transformer

use super::activation::{gelu_backward, gelu_forward};
use super::dropout::{CacheDropout, DropoutEntrenable};
use super::linear::{CacheLineal, LinealEntrenable};
use crate::tensor::Tensor;

/// MLP (red prealimentada) con activación GELU
pub struct MLPEntrenable {
    pub fc1: LinealEntrenable,
    pub fc2: LinealEntrenable,
    pub dropout_resid: DropoutEntrenable,
}

impl MLPEntrenable {
    /// Crea un nuevo MLP con expansión de 4x
    ///
    /// # Argumentos
    ///
    /// * `n_embd` - Dimensión de los embeddings (incrustaciones)
    /// * `tasa_dropout` - Probabilidad de dropout
    /// * `semilla` - Semilla aleatoria para la inicialización
    pub fn new(n_embd: usize, tasa_dropout: f32, semilla: u64) -> Self {
        let oculta = n_embd * 4; // GPT-2 usa una expansión de 4x
        Self {
            fc1: LinealEntrenable::new(n_embd, oculta, semilla),
            fc2: LinealEntrenable::new(oculta, n_embd, semilla + 1000),
            dropout_resid: DropoutEntrenable::new(tasa_dropout),
        }
    }

    /// Paso hacia adelante: x → fc1 → GELU → fc2
    ///
    /// # Argumentos
    ///
    /// * `x` - Tensor de entrada [long_sec, n_embd]
    ///
    /// # Retorna
    ///
    /// Tupla de (salida, cache)
    pub fn forward(&self, x: &Tensor) -> (Tensor, CacheMLP) {
        let (h, cache_fc1) = self.fc1.forward(x);
        let h_activada = gelu_forward(&h);
        let (y_proy, cache_fc2) = self.fc2.forward(&h_activada);

        // Aplicar dropout residual
        let (y, cache_dropout_resid) = self.dropout_resid.forward(&y_proy);

        let cache = CacheMLP {
            cache_fc1,
            h, // Guardar pre-activación para el paso hacia atrás de GELU
            #[allow(dead_code)]
            h_activada,
            cache_fc2,
            cache_dropout_resid,
        };

        (y, cache)
    }

    /// Paso hacia atrás a través del MLP
    ///
    /// Usa la regla de la cadena a través de fc2, GELU y fc1
    ///
    /// # Argumentos
    ///
    /// * `grad_salida` - Gradiente de la siguiente capa
    /// * `cache` - Valores almacenados en caché del paso hacia adelante
    ///
    /// # Retorna
    ///
    /// Gradientes para todos los parámetros y la entrada
    pub fn backward(&self, grad_salida: &Tensor, cache: &CacheMLP) -> GradientesMLP {
        // Retropropagar a través del dropout residual
        let grad_y_proy = self
            .dropout_resid
            .backward(grad_salida, &cache.cache_dropout_resid);

        // Retropropagar a través de fc2
        let grads_fc2 = self.fc2.backward(&grad_y_proy, &cache.cache_fc2);

        // Retropropagar a través de GELU
        let grad_h = gelu_backward(&grads_fc2.x, &cache.h);

        // Retropropagar a través de fc1
        let grads_fc1 = self.fc1.backward(&grad_h, &cache.cache_fc1);

        GradientesMLP {
            peso_fc1: grads_fc1.peso,
            sesgo_fc1: grads_fc1.sesgo,
            peso_fc2: grads_fc2.peso,
            sesgo_fc2: grads_fc2.sesgo,
            x: grads_fc1.x,
        }
    }
}

/// Caché para el paso hacia atrás del MLP
pub struct CacheMLP {
    pub cache_fc1: CacheLineal,
    pub h: Tensor, // Pre-activación (necesaria para el paso hacia atrás de GELU)
    #[allow(dead_code)]
    pub h_activada: Tensor,
    pub cache_fc2: CacheLineal,
    pub cache_dropout_resid: CacheDropout,
}

/// Gradientes para el MLP
pub struct GradientesMLP {
    pub peso_fc1: Tensor,
    pub sesgo_fc1: Tensor,
    pub peso_fc2: Tensor,
    pub sesgo_fc2: Tensor,
    pub x: Tensor,
}