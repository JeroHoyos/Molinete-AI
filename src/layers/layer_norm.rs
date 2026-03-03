//! Normalización por Capa (Layer Normalization)
//!
//! La normalización por capa es crucial para entrenar redes profundas. Normaliza
//! las activaciones para que tengan media cero y varianza unitaria, y luego aplica
//! parámetros aprendibles de escala (gamma) y desplazamiento (beta).
//!
//! ## La Parte Complicada: Paso Backward
//!
//! El backward de layer norm es complejo porque la media y la varianza dependen
//! de TODOS los elementos en el grupo normalizado. Esto crea dependencias que
//! requieren un cálculo cuidadoso de los gradientes.
//!
//! ## Paso Forward
//!
//! ```text
//! 1. mean = E[x] = sum(x) / N
//! 2. var = E[(x - mean)²] = sum((x - mean)²) / N
//! 3. x_norm = (x - mean) / √(var + ε)
//! 4. y = γ * x_norm + β
//! ```
//!
//! donde:
//! - ε (epsilon) previene la división por cero
//! - γ (gamma) es la escala aprendible
//! - β (beta) es el desplazamiento aprendible
//!
//! ## Paso Backward
//!
//! Los gradientes son:
//!
//! ```text
//! grad_γ = sum(grad_y * x_norm)
//! grad_β = sum(grad_y)
//! grad_x_norm = grad_y * γ
//! ```
//!
//! La parte complicada es retropropagar a través de la normalización:
//!
//! ```text
//! grad_x = (1/√var) * (grad_x_norm - E[grad_x_norm] - x_norm * E[grad_x_norm * x_norm])
//! ```
//!
//! Esta fórmula tiene en cuenta:
//! 1. Cada elemento afecta la media (primer término E)
//! 2. Cada elemento afecta la varianza (segundo término E)
//! 3. El gradiente directo a través de x_norm
//!
//! ## ¿Por qué Layer Norm?
//!
//! - **Estabilidad en el entrenamiento**: Previene el cambio interno de covarianza
//! - **Convergencia más rápida**: Las activaciones normalizadas entrenan más rápido
//! - **Menos sensible a la inicialización**: La normalización reduce el impacto de una mala inicialización
//! - **Funciona con cualquier tamaño de batch**: A diferencia de batch norm, no depende de estadísticas del batch

use crate::tensor::Tensor;

/// Capa de normalización por capa
///
/// Normaliza las activaciones a lo largo de la dimensión de características
/// y aplica escala y desplazamiento aprendibles.
pub struct TrainableLayerNorm {
    pub gamma: Tensor, // Parámetro de escala [n_embd]
    pub beta: Tensor,  // Parámetro de desplazamiento [n_embd]
    pub eps: f32,      // Constante pequeña para estabilidad numérica
}

impl TrainableLayerNorm {
    /// Crea una nueva capa de normalización por capa
    ///
    /// # Argumentos
    ///
    /// * `normalized_shape` - Tamaño de la dimensión de características a normalizar
    ///
    /// # Inicialización
    ///
    /// - gamma inicializado en 1.0 (sin escala al inicio)
    /// - beta inicializado en 0.0 (sin desplazamiento al inicio)
    /// - eps = 1e-5 (valor estándar)
    pub fn new(normalized_shape: usize) -> Self {
        Self {
            gamma: Tensor::new(vec![1.0; normalized_shape], vec![normalized_shape]),
            beta: Tensor::new(vec![0.0; normalized_shape], vec![normalized_shape]),
            eps: 1e-5,
        }
    }

    /// Paso forward
    ///
    /// Normaliza la entrada a media cero y varianza unitaria, luego aplica escala/desplazamiento
    ///
    /// # Argumentos
    ///
    /// * `x` - Tensor de entrada [seq_len, n_embd]
    ///
    /// # Retorna
    ///
    /// Tupla de (output, cache) donde:
    /// - output: Tensor normalizado [seq_len, n_embd]
    /// - cache: Almacena valores necesarios para el paso backward
    pub fn forward(&self, x: &Tensor) -> (Tensor, LayerNormCache) {
        // Calcular estadísticas a lo largo de la última dimensión
        let mean = x.mean(-1, true);
        let variance = x.var(-1, true);
        let std = variance.add_scalar(self.eps).sqrt();

        // Normalizar
        let x_centered = x.sub(&mean);
        let x_norm = x_centered.div(&std);

        // Aplicar escala y desplazamiento aprendibles
        let y = x_norm.mul(&self.gamma).add(&self.beta);

        let cache = LayerNormCache {
            x: x.clone(),
            x_norm,
            #[allow(dead_code)]
            mean,
            std,
        };

        (y, cache)
    }

    /// Paso backward
    ///
    /// Calcula gradientes para gamma, beta y la entrada. El gradiente de la
    /// entrada es complejo porque la normalización crea dependencias entre todos los elementos.
    ///
    /// # Argumentos
    ///
    /// * `grad_out` - Gradiente desde la siguiente capa [seq_len, n_embd]
    /// * `cache` - Valores almacenados del paso forward
    ///
    /// # Retorna
    ///
    /// Gradientes para gamma, beta y x
    pub fn backward(&self, grad_out: &Tensor, cache: &LayerNormCache) -> LayerNormGradients {
        let n_embd = self.gamma.data.len();
        let seq_len = grad_out.shape[0];

        // Calcular grad_gamma y grad_beta acumulando sobre la secuencia
        let mut grad_gamma = vec![0.0; n_embd];
        let mut grad_beta = vec![0.0; n_embd];
        for i in 0..seq_len {
            for j in 0..n_embd {
                let idx = i * n_embd + j;
                grad_gamma[j] += grad_out.data[idx] * cache.x_norm.data[idx];
                grad_beta[j] += grad_out.data[idx];
            }
        }

        // Retropropagar a través de la escala: grad_x_norm = grad_out * gamma
        let grad_x_norm = grad_out.mul(&self.gamma);

        // Retropropagar a través de la normalización (¡la parte compleja!)
        // Esto tiene en cuenta las dependencias de media y varianza
        let mut grad_x_data = vec![0.0; seq_len * n_embd];

        for i in 0..seq_len {
            let row_start = i * n_embd;
            let row_end = row_start + n_embd;

            let grad_x_norm_row = &grad_x_norm.data[row_start..row_end];
            let x_norm_row = &cache.x_norm.data[row_start..row_end];
            let std_val = cache.std.data[i];

            // Calcular la media de los gradientes (dependencia de la media)
            let mean_grad: f32 = grad_x_norm_row.iter().sum::<f32>() / n_embd as f32;

            // Calcular la media de (grad * x_norm) (dependencia de la varianza)
            let mean_grad_x: f32 = grad_x_norm_row
                .iter()
                .zip(x_norm_row.iter())
                .map(|(g, x)| g * x)
                .sum::<f32>()
                / n_embd as f32;

            // Fórmula final del gradiente
            for j in 0..n_embd {
                let idx = row_start + j;
                grad_x_data[idx] =
                    (grad_x_norm_row[j] - mean_grad - x_norm_row[j] * mean_grad_x) / std_val;
            }
        }

        LayerNormGradients {
            gamma: Tensor::new(grad_gamma, vec![n_embd]),
            beta: Tensor::new(grad_beta, vec![n_embd]),
            x: Tensor::new(grad_x_data, cache.x.shape.clone()),
        }
    }
}

/// Cache para el paso backward de layer norm
pub struct LayerNormCache {
    pub x: Tensor,
    pub x_norm: Tensor,
    #[allow(dead_code)]
    pub mean: Tensor,
    pub std: Tensor,
}

/// Gradientes para layer norm
pub struct LayerNormGradients {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub x: Tensor,
}