//! Capa de Dropout 
//!
//! Dropout es una técnica de regularización que pone a cero de forma aleatoria las activaciones
//! durante el entrenamiento para prevenir el sobreajuste (overfitting). Durante la inferencia,
//! pasa los valores sin cambios.

use crate::tensor::Tensor;

/// Capa de dropout entrenable
///
/// Esta capa descarta activaciones aleatoriamente durante el entrenamiento como regularización.
pub struct DropoutEntrenable {
    pub tasa: f32,
    pub entrenando: bool,
}

impl DropoutEntrenable {
    /// Crea una nueva capa de dropout
    ///
    /// # Argumentos
    ///
    /// * `tasa` - Probabilidad de dropout (0.0 = sin dropout, 1.0 = descartar todo)
    pub fn new(tasa: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&tasa),
            "La tasa de dropout debe estar entre 0.0 y 1.0"
        );
        Self {
            tasa,
            entrenando: true,
        }
    }

    /// Paso hacia adelante (forward) con almacenamiento en caché para el paso hacia atrás
    ///
    /// # Argumentos
    ///
    /// * `x` - Tensor de entrada
    ///
    /// # Retorna
    ///
    /// Tupla de (salida, cache) donde cache almacena la máscara de dropout
    pub fn forward(&self, x: &Tensor) -> (Tensor, CacheDropout) {
        if !self.entrenando || self.tasa == 0.0 {
            // Sin dropout - simplemente dejar pasar
            let cache = CacheDropout {
                mascara: None,
                escala: 1.0,
            };
            return (x.clone(), cache);
        }

        if self.tasa >= 1.0 {
            // Descartar todo
            let cache = CacheDropout {
                mascara: Some(vec![false; x.datos.len()]),
                escala: 1.0,
            };
            return (Tensor::ceros(x.forma.clone()), cache);
        }

        // Aplicar dropout con escalado
        let escala = 1.0 / (1.0 - self.tasa);
        let mut mascara = Vec::with_capacity(x.datos.len());
        let mut salida = Tensor::ceros(x.forma.clone());

        for i in 0..x.datos.len() {
            let mantener = rand::random::<f32>() > self.tasa;
            mascara.push(mantener);
            if mantener {
                salida.datos[i] = x.datos[i] * escala;
            }
        }

        let cache = CacheDropout {
            mascara: Some(mascara),
            escala,
        };

        (salida, cache)
    }

    /// Paso hacia atrás (backward) a través de dropout
    ///
    /// # Argumentos
    ///
    /// * `grad_salida` - Gradiente que fluye hacia atrás desde la siguiente capa
    /// * `cache` - Máscara de dropout almacenada en caché desde el paso hacia adelante
    ///
    /// # Retorna
    ///
    /// Gradiente con respecto a la entrada
    pub fn backward(&self, grad_salida: &Tensor, cache: &CacheDropout) -> Tensor {
        if let Some(mascara) = &cache.mascara {
            // Aplicar la misma máscara a los gradientes
            let mut grad_entrada = Tensor::ceros(grad_salida.forma.clone());
            for (i, &mantener) in mascara.iter().enumerate() {
                if mantener {
                    grad_entrada.datos[i] = grad_salida.datos[i] * cache.escala;
                }
                // else: el gradiente es cero (el valor fue descartado)
            }
            grad_entrada
        } else {
            // No se aplicó dropout, simplemente pasar el gradiente
            grad_salida.clone()
        }
    }
}

/// Caché para el paso hacia atrás del dropout
pub struct CacheDropout {
    /// Máscara de dropout (true = mantenido, false = descartado)
    /// None si el dropout estaba desactivado
    pub mascara: Option<Vec<bool>>,
    /// Factor de escalado aplicado a los valores mantenidos
    pub escala: f32,
}