//! # La Atención — El Don de Escuchar a Todos los Caballeros
//!
//! Si hay una innovación que distingue al Transformer de sus predecesores, es la Atención.
//! Así como Don Quijote escucha los relatos de cada personaje y los pondera según su importancia,
//! cada token de la secuencia "presta atención" a todos los demás, aprendiendo qué información
//! es relevante para su propio significado.
//!
//! ## El Mecanismo: Q, K y V
//!
//! Cada token genera tres vectores a partir de sus embeddings:
//!
//! - **Query (Q)**: "¿Qué busco saber?"
//! - **Key (K)**: "¿De qué trato yo?"
//! - **Value (V)**: "¿Qué información ofrezco?"
//!
//! Las puntuaciones de atención miden cuánto coincide cada Query con cada Key:
//!
//! ```text
//! puntuaciones = (Q @ Kᵀ) / √d_k     ← escalamos para evitar gradientes diminutos
//! pesos = softmax(puntuaciones_enmascaradas)
//! salida = pesos @ V
//! ```
//!
//! ## La Máscara Causal — El Pacto del Caballero
//!
//! Para el modelado de lenguaje, ningún token puede "espiar" el futuro.
//! La máscara causal pone -∞ en todas las posiciones futuras, que tras el softmax
//! se convierten en 0% de atención. Así, cada token solo ve su pasado.

use super::dropout::{CacheDropout, DropoutEntrenable};
use super::linear::{CacheLineal, LinealEntrenable};
use crate::tensor::Tensor;

/// Autoatención de una sola cabeza — un caballero que escucha a todos
pub struct AtencionUnaCabezaEntrenable {
    pub proy_q: LinealEntrenable,       // proyección para las Queries
    pub proy_k: LinealEntrenable,       // proyección para las Keys
    pub proy_v: LinealEntrenable,       // proyección para los Values
    pub proy_salida: LinealEntrenable,  // proyección de la salida final
    pub dropout_atencion: DropoutEntrenable, // regularización en los pesos de atención
    pub dropout_resid: DropoutEntrenable,    // regularización en la salida
    pub n_embd: usize,
}

impl AtencionUnaCabezaEntrenable {
    pub fn new(n_embd: usize, tasa_dropout: f32, semilla: u64) -> Self {
        Self {
            proy_q: LinealEntrenable::new(n_embd, n_embd, semilla),
            proy_k: LinealEntrenable::new(n_embd, n_embd, semilla + 1),
            proy_v: LinealEntrenable::new(n_embd, n_embd, semilla + 2),
            proy_salida: LinealEntrenable::new(n_embd, n_embd, semilla + 3),
            dropout_atencion: DropoutEntrenable::new(tasa_dropout),
            dropout_resid: DropoutEntrenable::new(tasa_dropout),
            n_embd,
        }
    }

    /// Paso hacia adelante — el caballero escucha y sintetiza
    pub fn forward(&self, x: &Tensor) -> (Tensor, CacheAtencion) {
        let long_sec = x.forma[0];

        // 1. Cada token proyecta su Query, Key y Value
        let (q, cache_q) = self.proy_q.forward(x);
        let (k, cache_k) = self.proy_k.forward(x);
        let (v, cache_v) = self.proy_v.forward(x);

        // 2. Puntuaciones: ¿cuánto coincide cada Query con cada Key?
        // Dividimos por √d_k para que los productos punto no sean demasiado grandes
        let escala = (self.n_embd as f32).sqrt();
        let puntuaciones = q.matmul(&k.transpose(-2, -1)).mul_scalar(1.0 / escala);

        // 3. Máscara causal: el pacto del caballero — nadie mira al futuro
        let mut mascara = vec![0.0; long_sec * long_sec];
        for i in 0..long_sec {
            for j in i + 1..long_sec {
                mascara[i * long_sec + j] = 1.0; // posiciones futuras quedan tapadas
            }
        }
        let tensor_mascara = Tensor::new(mascara, vec![long_sec, long_sec]);
        let puntuaciones_enmascaradas = puntuaciones.masked_fill(&tensor_mascara, -1e9);

        // 4. Softmax: convertir puntuaciones en porcentajes de atención
        let pesos_atencion = puntuaciones_enmascaradas.softmax(-1);
        let (pesos_atencion_descartados, cache_dropout_atencion) =
            self.dropout_atencion.forward(&pesos_atencion);

        // 5. Aplicar atención: cada posición recibe una mezcla ponderada de los Values
        let salida_atencion = pesos_atencion_descartados.matmul(&v);

        // 6. Proyección final y dropout residual
        let (y_proy, cache_salida) = self.proy_salida.forward(&salida_atencion);
        let (y, cache_dropout_resid) = self.dropout_resid.forward(&y_proy);

        let cache = CacheAtencion {
            x: x.clone(), q, k, v, pesos_atencion,
            #[allow(dead_code)]
            salida_atencion,
            cache_q, cache_k, cache_v, cache_salida,
            cache_dropout_atencion, cache_dropout_resid,
        };

        (y, cache)
    }

    /// Paso hacia atrás — el error vuelve por cada camino de atención
    pub fn backward(&self, grad_salida: &Tensor, cache: &CacheAtencion) -> GradientesAtencion {
        let long_sec = cache.x.forma[0];
        let escala = (self.n_embd as f32).sqrt();

        // Retropropagar por: dropout residual → proyección salida → matmul con V
        let grad_y_proy = self.dropout_resid.backward(grad_salida, &cache.cache_dropout_resid);
        let grads_salida = self.proy_salida.backward(&grad_y_proy, &cache.cache_salida);

        // Gradiente para V: pesos_atenciónᵀ @ grad_salida_atencion
        let grad_v = cache.pesos_atencion.transpose(-2, -1).matmul(&grads_salida.x);

        // Gradiente para los pesos de atención
        let grad_pesos_atencion_descartados = grads_salida.x.matmul(&cache.v.transpose(-2, -1));
        let grad_pesos_atencion = self.dropout_atencion.backward(
            &grad_pesos_atencion_descartados, &cache.cache_dropout_atencion);

        // Retropropagar a través del softmax (por fila — la parte matemáticamente interesante)
        // grad_puntuaciones = aten · (grad_aten - Σ(grad_aten · aten))
        let mut datos_grad_puntuaciones = Vec::new();
        for i in 0..long_sec {
            let inicio = i * long_sec;
            let fin = inicio + long_sec;
            let fila_atencion = &cache.pesos_atencion.datos[inicio..fin];
            let fila_grad_atencion = &grad_pesos_atencion.datos[inicio..fin];

            // El producto punto mide "cuánto del gradiente es absorbido por la normalización"
            let producto_punto: f32 = fila_atencion.iter().zip(fila_grad_atencion.iter())
                .map(|(a, g)| a * g).sum();

            for j in 0..long_sec {
                let grad_puntuacion = fila_atencion[j] * (fila_grad_atencion[j] - producto_punto);
                datos_grad_puntuaciones.push(grad_puntuacion);
            }
        }
        let grad_puntuaciones = Tensor::new(datos_grad_puntuaciones, vec![long_sec, long_sec]);

        // Retropropagar a través del escalado Q @ Kᵀ / √d_k
        let grad_q = grad_puntuaciones.matmul(&cache.k).mul_scalar(1.0 / escala);
        let grad_k = grad_puntuaciones.transpose(-2, -1).matmul(&cache.q).mul_scalar(1.0 / escala);

        // Retropropagar a través de las proyecciones Q, K, V
        let grads_q = self.proy_q.backward(&grad_q, &cache.cache_q);
        let grads_k = self.proy_k.backward(&grad_k, &cache.cache_k);
        let grads_v = self.proy_v.backward(&grad_v, &cache.cache_v);

        // Acumular gradientes de Q, K y V hacia la misma entrada x
        let mut datos_grad_x = vec![0.0; cache.x.datos.len()];
        for (i, val) in datos_grad_x.iter_mut().enumerate() {
            *val = grads_q.x.datos[i] + grads_k.x.datos[i] + grads_v.x.datos[i];
        }

        GradientesAtencion {
            peso_q: grads_q.peso, sesgo_q: grads_q.sesgo,
            peso_k: grads_k.peso, sesgo_k: grads_k.sesgo,
            peso_v: grads_v.peso, sesgo_v: grads_v.sesgo,
            peso_salida: grads_salida.peso, sesgo_salida: grads_salida.sesgo,
            x: Tensor::new(datos_grad_x, cache.x.forma.clone()),
        }
    }
}

/// Tesoro de atención — todos los tensores del forward que necesita el backward
pub struct CacheAtencion {
    pub x: Tensor, pub q: Tensor, pub k: Tensor, pub v: Tensor,
    pub pesos_atencion: Tensor,
    #[allow(dead_code)]
    pub salida_atencion: Tensor,
    pub cache_q: CacheLineal, pub cache_k: CacheLineal,
    pub cache_v: CacheLineal, pub cache_salida: CacheLineal,
    pub cache_dropout_atencion: CacheDropout, pub cache_dropout_resid: CacheDropout,
}

/// Los nueve gradientes de la capa de atención
pub struct GradientesAtencion {
    pub peso_q: Tensor, pub sesgo_q: Tensor,
    pub peso_k: Tensor, pub sesgo_k: Tensor,
    pub peso_v: Tensor, pub sesgo_v: Tensor,
    pub peso_salida: Tensor, pub sesgo_salida: Tensor,
    pub x: Tensor,
}
