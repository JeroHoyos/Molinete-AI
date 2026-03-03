//! Capa Lineal (Totalmente Conectada)
//!
//! La capa lineal es el bloque fundamental de las redes neuronales.
//! Realiza una transformación afín: y = x @ W + b
//!
//! ## Paso Forward
//!
//! ```text
//! Entrada: x [seq_len, in_features]
//! Peso:    W [in_features, out_features]
//! Bias:    b [out_features]
//! Salida:  y = x @ W + b [seq_len, out_features]
//! ```
//!
//! ## Paso Backward
//!
//! Usando la regla de la cadena:
//! ```text
//! grad_W = x^T @ grad_y
//! grad_b = sum(grad_y, axis=0)
//! grad_x = grad_y @ W^T
//! ```
//!
//! ## ¿Por qué estos gradientes?
//!
//! - **grad_W**: Cada peso W[i,j] afecta la salida y[*,j] a través de la entrada x[*,i]
//! - **grad_b**: Cada bias b[j] afecta todas las salidas y[*,j] por igual
//! - **grad_x**: Necesario para retropropagar a la capa anterior
//!
//! ## Notas de Implementación
//!
//! - Usa inicialización He: scale = √(2/in_features)
//! - Bias inicializado en cero (práctica común)
//! - Guarda en caché la entrada x para el paso backward

use crate::tensor::Tensor;

/// Función auxiliar para inicialización aleatoria
///
/// Usa un LCG (Generador Congruencial Lineal) simple para inicialización reproducible.
/// El parámetro scale controla la magnitud de los pesos iniciales.
pub fn random_init(size: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut rng = seed;
    (0..size)
        .map(|_| {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let val = ((rng / 65536) % 32768) as f32 / 32768.0;
            (val - 0.5) * 2.0 * scale
        })
        .collect()
}

/// Capa lineal (totalmente conectada)
///
/// Realiza y = x @ W + b donde:
/// - W: matriz de pesos [in_features, out_features]
/// - b: vector bias [out_features]
pub struct TrainableLinear {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl TrainableLinear {
    /// Crea una nueva capa lineal con inicialización He
    ///
    /// # Argumentos
    ///
    /// * `in_features` - Dimensión de entrada
    /// * `out_features` - Dimensión de salida
    /// * `seed` - Semilla aleatoria para reproducibilidad
    ///
    /// # Inicialización
    ///
    /// Usa inicialización He: scale = √(2/in_features)
    /// Esto ayuda a prevenir gradientes que desaparecen o explotan en redes profundas.
    pub fn new(in_features: usize, out_features: usize, seed: u64) -> Self {
        let scale = (2.0 / in_features as f32).sqrt();
        Self {
            weight: Tensor::new(
                random_init(in_features * out_features, seed, scale),
                vec![in_features, out_features],
            ),
            bias: Tensor::new(vec![0.0; out_features], vec![out_features]),
        }
    }

    /// Paso forward
    ///
    /// Calcula y = x @ W + b y guarda x para el paso backward
    ///
    /// # Argumentos
    ///
    /// * `x` - Tensor de entrada [seq_len, in_features]
    ///
    /// # Retorna
    ///
    /// Tupla de (output, cache) donde:
    /// - output: [seq_len, out_features]
    /// - cache: almacena x para el paso backward
    pub fn forward(&self, x: &Tensor) -> (Tensor, LinearCache) {
        let y = x.matmul(&self.weight).add(&self.bias);
        let cache = LinearCache { x: x.clone() };
        (y, cache)
    }

    /// Paso backward
    ///
    /// Calcula los gradientes para pesos, bias y entrada
    ///
    /// # Argumentos
    ///
    /// * `grad_out` - Gradiente desde la siguiente capa [seq_len, out_features]
    /// * `cache` - Valores almacenados del paso forward
    ///
    /// # Retorna
    ///
    /// Gradientes para weight, bias y x
    pub fn backward(&self, grad_out: &Tensor, cache: &LinearCache) -> LinearGradients {
        // grad_W = x^T @ grad_out
        let grad_weight = cache.x.transpose(-2, -1).matmul(grad_out);

        // grad_b = suma de grad_out a lo largo de todas las dimensiones excepto la última
        let grad_bias_data: Vec<f32> = (0..self.bias.data.len())
            .map(|i| {
                let mut sum = 0.0;
                for row in 0..grad_out.shape[0] {
                    sum += grad_out.data[row * grad_out.shape[1] + i];
                }
                sum
            })
            .collect();
        let grad_bias = Tensor::new(grad_bias_data, self.bias.shape.clone());

        // grad_x = grad_out @ W^T
        let grad_x = grad_out.matmul(&self.weight.transpose(-2, -1));

        LinearGradients {
            weight: grad_weight,
            bias: grad_bias,
            x: grad_x,
        }
    }
}

/// Caché para el paso backward de la capa lineal
pub struct LinearCache {
    pub x: Tensor,
}

/// Gradientes para la capa lineal
pub struct LinearGradients {
    pub weight: Tensor,
    pub bias: Tensor,
    pub x: Tensor, // Gradiente a pasar a la capa anterior
}