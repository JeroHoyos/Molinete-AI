//! # El Bloque Transformer — La Orden de Caballería Completa
//!
//! Un bloque transformer es el nivel de organización más importante de GPT-2.
//! Combina Atención y MLP con conexiones residuales y normalización de capa,
//! formando la "orden de caballería" que procesa cada capa de la red.
//!
//! ## La Arquitectura de la Orden (Pre-Norm)
//!
//! ```text
//! x → NormCapa → Atención → (+) → NormCapa → MLP → (+) → salida
//! │                          ↑                       ↑
//! └──────────────────────────┘                       │
//! └──────────────────────────────────────────────────┘
//! ```
//!
//! ## Las Conexiones Residuales — El Camino del Escudero
//!
//! Las conexiones residuales (`x + subcapa(x)`) son esenciales para entrenar
//! redes profundas: el gradiente puede fluir directamente hacia atrás a través
//! de la suma, sin pasar por la subcapa — como un escudero que toma el camino corto.
//!
//! ## Pre-Norm: Normalizar Antes de Actuar
//!
//! GPT-2 normaliza ANTES de cada subcapa (pre-norm), no después.
//! Esto da un entrenamiento más estable y mejor flujo de gradientes.

use super::attention::{CacheAtencion, GradientesAtencion, AtencionUnaCabezaEntrenable};
use super::layer_norm::{CacheNormCapa, NormCapaEntrenable};
use super::mlp::{CacheMLP, GradientesMLP, MLPEntrenable};
use crate::tensor::Tensor;

/// Bloque transformer — la unidad fundamental de la orden de caballería
pub struct BloqueTransformerEntrenable {
    pub ln1: NormCapaEntrenable,                  // normalización antes de la atención
    pub atencion: AtencionUnaCabezaEntrenable,    // el mecanismo de escucha
    pub ln2: NormCapaEntrenable,                  // normalización antes del MLP
    pub mlp: MLPEntrenable,                       // el taller de transformaciones
}

impl BloqueTransformerEntrenable {
    pub fn new(n_embd: usize, tasa_dropout: f32, semilla: u64) -> Self {
        Self {
            ln1: NormCapaEntrenable::new(n_embd),
            atencion: AtencionUnaCabezaEntrenable::new(n_embd, tasa_dropout, semilla),
            ln2: NormCapaEntrenable::new(n_embd),
            mlp: MLPEntrenable::new(n_embd, tasa_dropout, semilla + 1000),
        }
    }

    /// Paso hacia adelante: atención + MLP con conexiones residuales
    pub fn forward(&self, x: &Tensor) -> (Tensor, CacheBloque) {
        // Primer sub-bloque: NormCapa → Atención → Residual
        let (salida_ln1, cache_ln1) = self.ln1.forward(x);
        let (salida_atencion, cache_atencion) = self.atencion.forward(&salida_ln1);
        let x_despues_atencion = x.add(&salida_atencion); // conexión residual: el escudero toma el atajo

        // Segundo sub-bloque: NormCapa → MLP → Residual
        let (salida_ln2, cache_ln2) = self.ln2.forward(&x_despues_atencion);
        let (salida_mlp, cache_mlp) = self.mlp.forward(&salida_ln2);
        let y = x_despues_atencion.add(&salida_mlp); // segunda conexión residual

        let cache = CacheBloque {
            #[allow(dead_code)]
            x: x.clone(),
            cache_ln1, cache_atencion,
            #[allow(dead_code)]
            x_despues_atencion,
            cache_ln2, cache_mlp,
        };

        (y, cache)
    }

    /// Paso hacia atrás — el error se bifurca por las conexiones residuales
    ///
    /// En cada conexión residual `y = x + f(x)`, el gradiente se divide:
    /// - Un camino va directamente hacia atrás (la conexión residual).
    /// - El otro camino pasa por la subcapa f(x).
    /// Ambos se suman antes de continuar hacia la capa anterior.
    pub fn backward(&self, grad_salida: &Tensor, cache: &CacheBloque) -> GradientesBloque {
        // Segunda conexión residual: el gradiente se bifurca en dos caminos
        let grad_salida_mlp = grad_salida.clone();
        let mut grad_x_despues_atencion = grad_salida.clone(); // camino directo (skip)

        // Camino del MLP
        let grads_mlp = self.mlp.backward(&grad_salida_mlp, &cache.cache_mlp);
        let grads_ln2 = self.ln2.backward(&grads_mlp.x, &cache.cache_ln2);

        // Acumulamos: skip + camino MLP
        for i in 0..grad_x_despues_atencion.datos.len() {
            grad_x_despues_atencion.datos[i] += grads_ln2.x.datos[i];
        }

        // Primera conexión residual: otra bifurcación
        let grad_salida_atencion = grad_x_despues_atencion.clone();
        let mut grad_x = grad_x_despues_atencion;

        // Camino de la Atención
        let grads_atencion = self.atencion.backward(&grad_salida_atencion, &cache.cache_atencion);
        let grads_ln1 = self.ln1.backward(&grads_atencion.x, &cache.cache_ln1);

        // Acumulamos: skip + camino Atención
        for i in 0..grad_x.datos.len() {
            grad_x.datos[i] += grads_ln1.x.datos[i];
        }

        GradientesBloque {
            ln1_gamma: grads_ln1.gamma, ln1_beta: grads_ln1.beta,
            atencion: grads_atencion,
            ln2_gamma: grads_ln2.gamma, ln2_beta: grads_ln2.beta,
            mlp: grads_mlp,
            x: grad_x,
        }
    }
}

/// Tesoro del bloque — guarda los cachés de todas sus subcapas
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

/// Los gradientes de toda la orden de caballería
pub struct GradientesBloque {
    pub ln1_gamma: Tensor, pub ln1_beta: Tensor,
    pub atencion: GradientesAtencion,
    pub ln2_gamma: Tensor, pub ln2_beta: Tensor,
    pub mlp: GradientesMLP,
    pub x: Tensor,
}
