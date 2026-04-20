//! # La Red Prealimentada — El Taller del Alquimista
//!
//! Después de que la Atención recopila contexto de toda la secuencia, el MLP
//! es el alquimista que transforma esa información en conocimiento útil.
//! Expande los datos a un espacio cuatro veces más grande (donde hay más "espacio
//! para pensar"), aplica la activación GELU, y los comprime de vuelta.
//!
//! ## La Arquitectura del Taller
//!
//! ```text
//! x → Lineal1 (×4) → GELU → Lineal2 (÷4) → y
//! ```
//!
//! ## ¿Por qué 4×?
//!
//! La expansión de 4× es empírica: suficiente para representaciones complejas
//! sin dominar el presupuesto de parámetros. Es el estándar en GPT-2, BERT y más.

use super::activation::{gelu_backward, gelu_forward};
use super::dropout::{CacheDropout, DropoutEntrenable};
use super::linear::{CacheLineal, LinealEntrenable};
use crate::tensor::Tensor;

/// El perceptrón multicapa — el taller de transformaciones del transformer
pub struct MLPEntrenable {
    pub fc1: LinealEntrenable,          // Expansión: n_embd → 4·n_embd
    pub fc2: LinealEntrenable,          // Compresión: 4·n_embd → n_embd
    pub dropout_resid: DropoutEntrenable, // Regularización residual
}

impl MLPEntrenable {
    /// Construye el taller con expansión de 4× (estándar GPT-2)
    pub fn new(n_embd: usize, tasa_dropout: f32, semilla: u64) -> Self {
        let oculta = n_embd * 4; // el espacio expandido del alquimista
        Self {
            fc1: LinealEntrenable::new(n_embd, oculta, semilla),
            fc2: LinealEntrenable::new(oculta, n_embd, semilla + 1000),
            dropout_resid: DropoutEntrenable::new(tasa_dropout),
        }
    }

    /// Paso hacia adelante: expansión → activación → compresión
    pub fn forward(&self, x: &Tensor) -> (Tensor, CacheMLP) {
        // 1. Expansión: abrimos el espacio de representación
        let (h, cache_fc1) = self.fc1.forward(x);
        // 2. Activación GELU: la no-linealidad suave del alquimista
        let h_activada = gelu_forward(&h);
        // 3. Compresión: volvemos a la dimensionalidad del transformer
        let (y_proy, cache_fc2) = self.fc2.forward(&h_activada);
        // 4. Dropout residual: el ejercicio de la humildad
        let (y, cache_dropout_resid) = self.dropout_resid.forward(&y_proy);

        let cache = CacheMLP {
            cache_fc1,
            h, // guardamos la pre-activación para el backward de GELU
            #[allow(dead_code)]
            h_activada,
            cache_fc2,
            cache_dropout_resid,
        };

        (y, cache)
    }

    /// Paso hacia atrás — el error recorre el taller en sentido inverso
    pub fn backward(&self, grad_salida: &Tensor, cache: &CacheMLP) -> GradientesMLP {
        // 4→3: Retropropagar a través del dropout
        let grad_y_proy = self.dropout_resid.backward(grad_salida, &cache.cache_dropout_resid);
        // 3→2: Retropropagar a través de fc2 (la compresión)
        let grads_fc2 = self.fc2.backward(&grad_y_proy, &cache.cache_fc2);
        // 2→1: Retropropagar a través de GELU usando la pre-activación guardada
        let grad_h = gelu_backward(&grads_fc2.x, &cache.h);
        // 1→0: Retropropagar a través de fc1 (la expansión)
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

/// El cofre del taller — guarda la pre-activación para el backward de GELU
pub struct CacheMLP {
    pub cache_fc1: CacheLineal,
    pub h: Tensor,               // Pre-activación: antes de GELU, necesaria para gelu_backward
    #[allow(dead_code)]
    pub h_activada: Tensor,
    pub cache_fc2: CacheLineal,
    pub cache_dropout_resid: CacheDropout,
}

/// Los cinco gradientes del taller
pub struct GradientesMLP {
    pub peso_fc1: Tensor,
    pub sesgo_fc1: Tensor,
    pub peso_fc2: Tensor,
    pub sesgo_fc2: Tensor,
    pub x: Tensor, // el gradiente que sube hacia la capa anterior
}
