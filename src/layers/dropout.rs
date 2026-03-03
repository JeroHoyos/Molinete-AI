//! Capa de Dropout
//!
//! Dropout es una técnica de regularización que anula aleatoriamente activaciones
//! durante el entrenamiento para prevenir el sobreajuste (overfitting).
//! Durante la inferencia, los valores se pasan sin modificación.

use crate::tensor::Tensor;

/// Capa de dropout entrenable
///
/// Esta capa descarta aleatoriamente activaciones durante el entrenamiento
/// para regularización.
pub struct TrainableDropout {
    pub rate: f32,
    pub training: bool,
}

impl TrainableDropout {
    /// Crea una nueva capa de dropout
    ///
    /// # Argumentos
    ///
    /// * `rate` - Probabilidad de dropout (0.0 = sin dropout, 1.0 = descartar todo)
    pub fn new(rate: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&rate),
            "La tasa de dropout debe estar entre 0.0 y 1.0"
        );
        Self {
            rate,
            training: true,
        }
    }

    /// Forward pass con almacenamiento en caché para backward
    ///
    /// # Argumentos
    ///
    /// * `x` - Tensor de entrada
    ///
    /// # Retorna
    ///
    /// Tupla de (output, cache) donde cache almacena la máscara de dropout
    pub fn forward(&self, x: &Tensor) -> (Tensor, DropoutCache) {
        if !self.training || self.rate == 0.0 {
            // Sin dropout - simplemente pasar los valores
            let cache = DropoutCache {
                mask: None,
                scale: 1.0,
            };
            return (x.clone(), cache);
        }

        if self.rate >= 1.0 {
            // Descartar todo
            let cache = DropoutCache {
                mask: Some(vec![false; x.data.len()]),
                scale: 1.0,
            };
            return (Tensor::zeros(x.shape.clone()), cache);
        }

        // Aplicar dropout con escalado
        let scale = 1.0 / (1.0 - self.rate);
        let mut mask = Vec::with_capacity(x.data.len());
        let mut output = Tensor::zeros(x.shape.clone());

        for i in 0..x.data.len() {
            let keep = rand::random::<f32>() > self.rate;
            mask.push(keep);
            if keep {
                output.data[i] = x.data[i] * scale;
            }
        }

        let cache = DropoutCache {
            mask: Some(mask),
            scale,
        };

        (output, cache)
    }

    /// Backward pass a través de dropout
    ///
    /// # Argumentos
    ///
    /// * `grad_output` - Gradiente que fluye desde la siguiente capa
    /// * `cache` - Máscara de dropout almacenada del forward pass
    ///
    /// # Retorna
    ///
    /// Gradiente con respecto a la entrada
    pub fn backward(&self, grad_output: &Tensor, cache: &DropoutCache) -> Tensor {
        if let Some(mask) = &cache.mask {
            // Aplicar la misma máscara a los gradientes
            let mut grad_input = Tensor::zeros(grad_output.shape.clone());
            for (i, &keep) in mask.iter().enumerate() {
                if keep {
                    grad_input.data[i] = grad_output.data[i] * cache.scale;
                }
                // else: el gradiente es cero (el valor fue descartado)
            }
            grad_input
        } else {
            // No se aplicó dropout, simplemente pasar el gradiente
            grad_output.clone()
        }
    }
}

/// Cache para el backward pass de dropout
pub struct DropoutCache {
    /// Máscara de dropout (true = conservado, false = descartado)
    /// None si el dropout estaba deshabilitado
    pub mask: Option<Vec<bool>>,
    /// Factor de escalado aplicado a los valores conservados
    pub scale: f32,
}