//! Bloque Transformer
//!
//! Un bloque transformer es el componente fundamental de los modelos GPT.
//! Combina capas de atención y feedforward (alimentación hacia adelante) con 
//! conexiones residuales y normalización de capa.
//!
//! ## Arquitectura
//!
//! ```text
//! x → NormCapa → Atención → (+) → NormCapa → MLP → (+) → salida
//! │                           ↑                      ↑
//! └───────────────────────────┘                      │
//! └──────────────────────────────────────────────────┘
//! ```
//!
//! ## Pre-Norm vs Post-Norm
//!
//! Usamos **pre-norm** (Normalización de Capa antes de las subcapas) en lugar de post-norm:
//! - Entrenamiento más estable
//! - Mejor flujo de gradientes
//! - Es el estándar en transformers modernos (GPT-2, GPT-3)
//!
//! ## Conexiones Residuales
//!
//! Las conexiones residuales son críticas para entrenar redes profundas:
//! - Permiten que los gradientes fluyan directamente hacia atrás
//! - Previenen los gradientes desvanecientes (vanishing gradients)
//! - Habilitan el entrenamiento de modelos muy profundos (más de 100 capas)
//!
//! ## Paso Hacia Atrás (Backward Pass)
//!
//! El paso hacia atrás a través de las conexiones residuales requiere una acumulación 
//! cuidadosa de gradientes. En cada conexión residual, los gradientes se dividen en 
//! dos caminos que deben sumarse.

use super::attention::{CacheAtencion, GradientesAtencion, AtencionUnaCabezaEntrenable};
use super::layer_norm::{CacheNormCapa, NormCapaEntrenable};
use super::mlp::{CacheMLP, GradientesMLP, MLPEntrenable};
use crate::tensor::Tensor;

/// Bloque transformer que combina atención y MLP con conexiones residuales
pub struct BloqueTransformerEntrenable {
    pub ln1: NormCapaEntrenable,
    pub atencion: AtencionUnaCabezaEntrenable,
    pub ln2: NormCapaEntrenable,
    pub mlp: MLPEntrenable,
}

impl BloqueTransformerEntrenable {
    /// Crea un nuevo bloque transformer
    ///
    /// # Argumentos
    ///
    /// * `n_embd` - Dimensión de los embeddings (incrustaciones)
    /// * `tasa_dropout` - Probabilidad de dropout
    /// * `semilla` - Semilla aleatoria para la inicialización
    pub fn new(n_embd: usize, tasa_dropout: f32, semilla: u64) -> Self {
        Self {
            ln1: NormCapaEntrenable::new(n_embd),
            atencion: AtencionUnaCabezaEntrenable::new(n_embd, tasa_dropout, semilla),
            ln2: NormCapaEntrenable::new(n_embd),
            mlp: MLPEntrenable::new(n_embd, tasa_dropout, semilla + 1000),
        }
    }

    /// Paso hacia adelante: atención + MLP con conexiones residuales
    ///
    /// # Argumentos
    ///
    /// * `x` - Tensor de entrada [long_sec, n_embd]
    ///
    /// # Retorna
    ///
    /// Tupla de (salida, cache)
    pub fn forward(&self, x: &Tensor) -> (Tensor, CacheBloque) {
        // Primer sub-bloque: NormCapa → Atención → Residual
        let (salida_ln1, cache_ln1) = self.ln1.forward(x);
        let (salida_atencion, cache_atencion) = self.atencion.forward(&salida_ln1);
        let x_despues_atencion = x.add(&salida_atencion); // Conexión residual

        // Segundo sub-bloque: NormCapa → MLP → Residual
        let (salida_ln2, cache_ln2) = self.ln2.forward(&x_despues_atencion);
        let (salida_mlp, cache_mlp) = self.mlp.forward(&salida_ln2);
        let y = x_despues_atencion.add(&salida_mlp); // Conexión residual

        let cache = CacheBloque {
            #[allow(dead_code)]
            x: x.clone(),
            cache_ln1,
            cache_atencion,
            #[allow(dead_code)]
            x_despues_atencion,
            cache_ln2,
            cache_mlp,
        };

        (y, cache)
    }

    /// Paso hacia atrás a través del bloque transformer
    ///
    /// # Argumentos
    ///
    /// * `grad_salida` - Gradiente de la siguiente capa
    /// * `cache` - Valores almacenados en caché del paso hacia adelante
    ///
    /// # Retorna
    ///
    /// Gradientes para todos los parámetros y la entrada
    pub fn backward(&self, grad_salida: &Tensor, cache: &CacheBloque) -> GradientesBloque {
        // Retropropagar a través de la segunda conexión residual
        // El gradiente fluye tanto hacia la ruta del MLP como directamente a la primera ruta residual
        let grad_salida_mlp = grad_salida.clone();
        let mut grad_x_despues_atencion = grad_salida.clone();

        // Retropropagar a través de la ruta del MLP
        let grads_mlp = self.mlp.backward(&grad_salida_mlp, &cache.cache_mlp);
        let grads_ln2 = self.ln2.backward(&grads_mlp.x, &cache.cache_ln2);

        // Acumular gradiente de la ruta de LN2
        for i in 0..grad_x_despues_atencion.datos.len() {
            grad_x_despues_atencion.datos[i] += grads_ln2.x.datos[i];
        }

        // Retropropagar a través de la primera conexión residual
        let grad_salida_atencion = grad_x_despues_atencion.clone();
        let mut grad_x = grad_x_despues_atencion;

        // Retropropagar a través de la ruta de atención
        let grads_atencion = self.atencion.backward(&grad_salida_atencion, &cache.cache_atencion);
        let grads_ln1 = self.ln1.backward(&grads_atencion.x, &cache.cache_ln1);

        // Acumular gradiente de la ruta de LN1
        for i in 0..grad_x.datos.len() {
            grad_x.datos[i] += grads_ln1.x.datos[i];
        }

        GradientesBloque {
            ln1_gamma: grads_ln1.gamma,
            ln1_beta: grads_ln1.beta,
            atencion: grads_atencion,
            ln2_gamma: grads_ln2.gamma,
            ln2_beta: grads_ln2.beta,
            mlp: grads_mlp,
            x: grad_x,
        }
    }
}

/// Caché para el paso hacia atrás del bloque transformer
pub struct CacheBloque {
    #[allow(dead_code)]
    pub x: Tensor,
    pub cache_ln1: CacheNormCapa,
    pub cache_atencion: CacheAtencion,
    #[allow(dead_code)]
    pub x_despues_atencion: Tensor,
    pub cache_ln2: CacheNormCapa,
    pub cache_mlp: CacheMLP,
}

/// Gradientes para el bloque transformer
pub struct GradientesBloque {
    pub ln1_gamma: Tensor,
    pub ln1_beta: Tensor,
    pub atencion: GradientesAtencion,
    pub ln2_gamma: Tensor,
    pub ln2_beta: Tensor,
    pub mlp: GradientesMLP,
    pub x: Tensor,
}