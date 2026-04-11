//! Mecanismo de Autoatención (Self-Attention)
//!
//! La atención es la innovación central de los transformers. Permite que cada posición
//! preste atención a todas las posiciones anteriores, aprendiendo relaciones contextuales.
//!
//! ## Atención de Producto Punto Escalado (Scaled Dot-Product Attention)
//!
//! ```text
//! Q, K, V = x @ W_q, x @ W_k, x @ W_v
//! puntuaciones = (Q @ K^T) / √d_k
//! pesos_atencion = softmax(puntuaciones_enmascaradas)
//! salida = pesos_atencion @ V
//! ```
//!
//! ## ¿Por qué escalar?
//!
//! Dividimos por √d_k para evitar que los productos punto crezcan demasiado,
//! lo cual empujaría al softmax hacia regiones con gradientes insignificantemente pequeños (desvanecientes).
//!
//! ## Enmascaramiento Causal (Causal Masking)
//!
//! Para el modelado de lenguaje, enmascaramos las posiciones futuras para que cada token solo
//! pueda prestar atención a sí mismo y a los tokens anteriores. Esto es crucial para la generación autorregresiva.
//!
//! ## Paso Hacia Atrás (Backward Pass)
//!
//! El paso hacia atrás a través de la atención implica:
//! 1. Retropropagación a través de la proyección de salida
//! 2. Retropropagación a través de la suma ponderada por atención (V)
//! 3. Retropropagación a través del softmax (con gradientes por fila)
//! 4. Retropropagación a través del producto punto escalado
//! 5. Retropropagación a través de las proyecciones Q, K, V
//!
//! El paso hacia atrás de softmax es particularmente interesante: necesitamos tener en cuenta
//! el hecho de que softmax acopla todos los elementos en cada fila.

use super::dropout::{CacheDropout, DropoutEntrenable};
use super::linear::{CacheLineal, LinealEntrenable};
use crate::tensor::Tensor;

/// Autoatención de una sola cabeza
///
/// Esto implementa una cabeza de atención. La atención multicabeza (multi-head) ejecutaría múltiples
/// copias de esto en paralelo.
pub struct AtencionUnaCabezaEntrenable {
    pub proy_q: LinealEntrenable,
    pub proy_k: LinealEntrenable,
    pub proy_v: LinealEntrenable,
    pub proy_salida: LinealEntrenable,
    pub dropout_atencion: DropoutEntrenable,
    pub dropout_resid: DropoutEntrenable,
    pub n_embd: usize,
}

impl AtencionUnaCabezaEntrenable {
    /// Crea una nueva capa de atención
    ///
    /// # Argumentos
    ///
    /// * `n_embd` - Dimensión de los embeddings (incrustaciones)
    /// * `tasa_dropout` - Probabilidad de dropout
    /// * `semilla` - Semilla aleatoria para la inicialización
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

    /// Paso hacia adelante: atención de producto punto escalado con enmascaramiento causal
    ///
    /// # Argumentos
    ///
    /// * `x` - Tensor de entrada [long_sec, n_embd]
    ///
    /// # Retorna
    ///
    /// Tupla de (salida, cache)
    pub fn forward(&self, x: &Tensor) -> (Tensor, CacheAtencion) {
        let long_sec = x.forma[0];

        // Proyectar a Q, K, V
        let (q, cache_q) = self.proy_q.forward(x);
        let (k, cache_k) = self.proy_k.forward(x);
        let (v, cache_v) = self.proy_v.forward(x);

        // Atención de producto punto escalado
        let escala = (self.n_embd as f32).sqrt();
        let puntuaciones = q.matmul(&k.transpose(-2, -1)).mul_scalar(1.0 / escala);

        // Máscara causal: previene prestar atención a posiciones futuras
        let mut mascara = vec![0.0; long_sec * long_sec];
        for i in 0..long_sec {
            for j in i + 1..long_sec {
                mascara[i * long_sec + j] = 1.0;
            }
        }
        let tensor_mascara = Tensor::new(mascara, vec![long_sec, long_sec]);
        let puntuaciones_enmascaradas = puntuaciones.masked_fill(&tensor_mascara, -1e9);

        // Softmax -> pesos de atención
        let pesos_atencion = puntuaciones_enmascaradas.softmax(-1);

        // Aplicar dropout a los pesos de atención
        let (pesos_atencion_descartados, cache_dropout_atencion) = self.dropout_atencion.forward(&pesos_atencion);

        // Aplicar atención a los valores (V)
        let salida_atencion = pesos_atencion_descartados.matmul(&v);

        // Proyección de salida
        let (y_proy, cache_salida) = self.proy_salida.forward(&salida_atencion);

        // Aplicar dropout residual
        let (y, cache_dropout_resid) = self.dropout_resid.forward(&y_proy);

        let cache = CacheAtencion {
            x: x.clone(),
            q,
            k,
            v,
            pesos_atencion,
            #[allow(dead_code)]
            salida_atencion,
            cache_q,
            cache_k,
            cache_v,
            cache_salida,
            cache_dropout_atencion,
            cache_dropout_resid,
        };

        (y, cache)
    }

    /// Paso hacia atrás a través de la atención
    ///
    /// # Argumentos
    ///
    /// * `grad_salida` - Gradiente de la siguiente capa
    /// * `cache` - Valores almacenados en caché del paso hacia adelante
    ///
    /// # Retorna
    ///
    /// Gradientes para todos los parámetros y la entrada
    pub fn backward(&self, grad_salida: &Tensor, cache: &CacheAtencion) -> GradientesAtencion {
        let long_sec = cache.x.forma[0];
        let escala = (self.n_embd as f32).sqrt();

        // Retropropagar a través del dropout residual
        let grad_y_proy = self
            .dropout_resid
            .backward(grad_salida, &cache.cache_dropout_resid);

        // Retropropagar a través de la proyección de salida
        let grads_salida = self.proy_salida.backward(&grad_y_proy, &cache.cache_salida);

        // Retropropagar a través de la atención: grad_v = pesos_atencion^T @ grad_salida_atencion
        let grad_v = cache.pesos_atencion.transpose(-2, -1).matmul(&grads_salida.x);

        // grad_pesos_atencion = grad_salida_atencion @ v^T
        let grad_pesos_atencion_descartados = grads_salida.x.matmul(&cache.v.transpose(-2, -1));

        // Retropropagar a través del dropout de atención
        let grad_pesos_atencion = self
            .dropout_atencion
            .backward(&grad_pesos_atencion_descartados, &cache.cache_dropout_atencion);

        // Retropropagar a través de softmax (por fila)
        // gradiente de softmax: grad_puntuaciones = aten * (grad_aten - sum(grad_aten * aten))
        let mut datos_grad_puntuaciones = Vec::new();
        for i in 0..long_sec {
            let inicio = i * long_sec;
            let fin = inicio + long_sec;
            let fila_atencion = &cache.pesos_atencion.datos[inicio..fin];
            let fila_grad_atencion = &grad_pesos_atencion.datos[inicio..fin];

            // Calcular producto punto para esta fila
            let producto_punto: f32 = fila_atencion
                .iter()
                .zip(fila_grad_atencion.iter())
                .map(|(a, g)| a * g)
                .sum();

            // Aplicar fórmula del gradiente de softmax
            for j in 0..long_sec {
                let grad_puntuacion = fila_atencion[j] * (fila_grad_atencion[j] - producto_punto);
                datos_grad_puntuaciones.push(grad_puntuacion);
            }
        }
        let grad_puntuaciones = Tensor::new(datos_grad_puntuaciones, vec![long_sec, long_sec]);

        // Retropropagar a través del escalado Q @ K^T
        let grad_q = grad_puntuaciones.matmul(&cache.k).mul_scalar(1.0 / escala);
        let grad_k = grad_puntuaciones
            .transpose(-2, -1)
            .matmul(&cache.q)
            .mul_scalar(1.0 / escala);

        // Retropropagar a través de proyecciones Q, K, V
        let grads_q = self.proy_q.backward(&grad_q, &cache.cache_q);
        let grads_k = self.proy_k.backward(&grad_k, &cache.cache_k);
        let grads_v = self.proy_v.backward(&grad_v, &cache.cache_v);

        // Acumular gradientes para la entrada (Q, K, V se conectan a la misma entrada)
        let mut datos_grad_x = vec![0.0; cache.x.datos.len()];
        for (i, valor_grad_x) in datos_grad_x.iter_mut().enumerate() {
            *valor_grad_x = grads_q.x.datos[i] + grads_k.x.datos[i] + grads_v.x.datos[i];
        }
        let grad_x = Tensor::new(datos_grad_x, cache.x.forma.clone());

        GradientesAtencion {
            peso_q: grads_q.peso,
            sesgo_q: grads_q.sesgo,
            peso_k: grads_k.peso,
            sesgo_k: grads_k.sesgo,
            peso_v: grads_v.peso,
            sesgo_v: grads_v.sesgo,
            peso_salida: grads_salida.peso,
            sesgo_salida: grads_salida.sesgo,
            x: grad_x,
        }
    }
}

/// Caché para el paso hacia atrás de atención
pub struct CacheAtencion {
    pub x: Tensor,
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub pesos_atencion: Tensor,
    #[allow(dead_code)]
    pub salida_atencion: Tensor,
    pub cache_q: CacheLineal,
    pub cache_k: CacheLineal,
    pub cache_v: CacheLineal,
    pub cache_salida: CacheLineal,
    pub cache_dropout_atencion: CacheDropout,
    pub cache_dropout_resid: CacheDropout,
}

/// Gradientes para atención
pub struct GradientesAtencion {
    pub peso_q: Tensor,
    pub sesgo_q: Tensor,
    pub peso_k: Tensor,
    pub sesgo_k: Tensor,
    pub peso_v: Tensor,
    pub sesgo_v: Tensor,
    pub peso_salida: Tensor,
    pub sesgo_salida: Tensor,
    pub x: Tensor,
}