//! Normalización de Capa (Layer Normalization)
//!
//! La normalización de capa es crucial para entrenar redes profundas. Normaliza
//! las activaciones para que tengan media cero y varianza unitaria, y luego aplica
//! parámetros aprendibles de escala (gamma) y desplazamiento (beta).
//!
//! ## La Parte Complicada: Paso Hacia Atrás (Backward Pass)
//!
//! El paso hacia atrás de la normalización de capa es complejo porque la media y la varianza
//! dependen de TODOS los elementos en el grupo normalizado. Esto crea dependencias que
//! requieren un cálculo de gradientes cuidadoso.
//!
//! ## Paso Hacia Adelante (Forward Pass)
//!
//! ```text
//! 1. media = E[x] = sum(x) / N
//! 2. var = E[(x - media)²] = sum((x - media)²) / N
//! 3. x_norm = (x - media) / √(var + ε)
//! 4. y = γ * x_norm + β
//! ```
//!
//! donde:
//! - ε (épsilon) previene la división por cero
//! - γ (gamma) es la escala aprendible
//! - β (beta) es el desplazamiento aprendible
//!
//! ## Paso Hacia Atrás (Backward Pass)
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
//! 1. Cada elemento afecta a la media (primer término E)
//! 2. Cada elemento afecta a la varianza (segundo término E)
//! 3. El gradiente directo a través de x_norm
//!
//! ## ¿Por Qué Normalización de Capa?
//!
//! - **Estabilidad de entrenamiento**: Previene el desplazamiento de covariables interno
//! - **Convergencia más rápida**: Las activaciones normalizadas se entrenan más rápido
//! - **Menos sensible a la inicialización**: La normalización reduce el impacto de una mala inicialización
//! - **Funciona con cualquier tamaño de lote**: A diferencia de la normalización por lotes (batch norm), no depende de las estadísticas del lote

use crate::tensor::Tensor;

/// Capa de normalización de capa
///
/// Normaliza las activaciones a lo largo de la dimensión de características y aplica
/// escala y desplazamiento aprendibles.
pub struct NormCapaEntrenable {
    pub gamma: Tensor, // Parámetro de escala [n_embd]
    pub beta: Tensor,  // Parámetro de desplazamiento [n_embd]
    pub eps: f32,      // Constante pequeña para estabilidad numérica
}

impl NormCapaEntrenable {
    /// Crea una nueva capa de normalización de capa
    ///
    /// # Argumentos
    ///
    /// * `forma_normalizada` - Tamaño de la dimensión de características a normalizar
    ///
    /// # Inicialización
    ///
    /// - gamma inicializado a 1.0 (sin escala inicialmente)
    /// - beta inicializado a 0.0 (sin desplazamiento inicialmente)
    /// - eps = 1e-5 (valor estándar)
    pub fn new(forma_normalizada: usize) -> Self {
        Self {
            gamma: Tensor::new(vec![1.0; forma_normalizada], vec![forma_normalizada]),
            beta: Tensor::new(vec![0.0; forma_normalizada], vec![forma_normalizada]),
            eps: 1e-5,
        }
    }

    /// Paso hacia adelante
    ///
    /// Normaliza la entrada a media cero y varianza unitaria, luego aplica escala/desplazamiento
    ///
    /// # Argumentos
    ///
    /// * `x` - Tensor de entrada [long_sec, n_embd]
    ///
    /// # Retorna
    ///
    /// Tupla de (salida, cache) donde:
    /// - salida: Tensor normalizado [long_sec, n_embd]
    /// - cache: Almacena valores necesarios para el paso hacia atrás
    pub fn forward(&self, x: &Tensor) -> (Tensor, CacheNormCapa) {
        // Calcular estadísticas a lo largo de la última dimensión
        let media = x.mean(-1, true);
        let varianza = x.var(-1, true);
        let desv_est = varianza.add_scalar(self.eps).sqrt();

        // Normalizar
        let x_centrado = x.sub(&media);
        let x_norm = x_centrado.div(&desv_est);

        // Aplicar escala y desplazamiento aprendibles
        let y = x_norm.mul(&self.gamma).add(&self.beta);

        let cache = CacheNormCapa {
            x: x.clone(),
            x_norm,
            #[allow(dead_code)]
            media,
            desv_est,
        };

        (y, cache)
    }

    /// Paso hacia atrás
    ///
    /// Calcula los gradientes para gamma, beta y la entrada. El gradiente de la entrada es
    /// complejo porque la normalización crea dependencias entre todos los elementos.
    ///
    /// # Argumentos
    ///
    /// * `grad_salida` - Gradiente de la siguiente capa [long_sec, n_embd]
    /// * `cache` - Valores almacenados en caché del paso hacia adelante
    ///
    /// # Retorna
    ///
    /// Gradientes para gamma, beta y la entrada
    pub fn backward(&self, grad_salida: &Tensor, cache: &CacheNormCapa) -> GradientesNormCapa {
        let n_embd = self.gamma.datos.len();
        let long_sec = grad_salida.forma[0];

        // Calcular grad_gamma y grad_beta acumulando a lo largo de la secuencia
        let mut grad_gamma = vec![0.0; n_embd];
        let mut grad_beta = vec![0.0; n_embd];
        for i in 0..long_sec {
            for j in 0..n_embd {
                let idx = i * n_embd + j;
                grad_gamma[j] += grad_salida.datos[idx] * cache.x_norm.datos[idx];
                grad_beta[j] += grad_salida.datos[idx];
            }
        }

        // Retropropagar a través de la escala: grad_x_norm = grad_salida * gamma
        let grad_x_norm = grad_salida.mul(&self.gamma);

        // Retropropagar a través de la normalización (¡la parte compleja!)
        // Esto tiene en cuenta las dependencias de la media y la varianza
        let mut datos_grad_x = vec![0.0; long_sec * n_embd];

        for i in 0..long_sec {
            let inicio_fila = i * n_embd;
            let fin_fila = inicio_fila + n_embd;

            let fila_grad_x_norm = &grad_x_norm.datos[inicio_fila..fin_fila];
            let fila_x_norm = &cache.x_norm.datos[inicio_fila..fin_fila];
            let valor_desv_est = cache.desv_est.datos[i];

            // Calcular media de los gradientes (tiene en cuenta la dependencia de la media)
            let media_grad: f32 = fila_grad_x_norm.iter().sum::<f32>() / n_embd as f32;

            // Calcular media de (grad * x_norm) (tiene en cuenta la dependencia de la varianza)
            let media_grad_x: f32 = fila_grad_x_norm
                .iter()
                .zip(fila_x_norm.iter())
                .map(|(g, x)| g * x)
                .sum::<f32>()
                / n_embd as f32;

            // Fórmula final del gradiente
            for j in 0..n_embd {
                let idx = inicio_fila + j;
                datos_grad_x[idx] =
                    (fila_grad_x_norm[j] - media_grad - fila_x_norm[j] * media_grad_x) / valor_desv_est;
            }
        }

        GradientesNormCapa {
            gamma: Tensor::new(grad_gamma, vec![n_embd]),
            beta: Tensor::new(grad_beta, vec![n_embd]),
            x: Tensor::new(datos_grad_x, cache.x.forma.clone()),
        }
    }
}

/// Caché para el paso hacia atrás de la normalización de capa
pub struct CacheNormCapa {
    pub x: Tensor,
    pub x_norm: Tensor,
    #[allow(dead_code)]
    pub media: Tensor,
    pub desv_est: Tensor,
}

/// Gradientes para la normalización de capa
pub struct GradientesNormCapa {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub x: Tensor,
}