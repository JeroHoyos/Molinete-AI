//! # El Equilibrio del Sabio — Normalización de Capa
//!
//! En las profundas mazmorras de una red neuronal, los valores pueden desbocarse
//! como caballos sin freno: unos enormes, otros diminutos, provocando inestabilidad.
//! La Normalización de Capa es el escudero que mantiene el orden: normaliza cada
//! vector de activación para que tenga media cero y varianza unitaria, y luego
//! aplica escala (γ) y desplazamiento (β) aprendibles.
//!
//! ## El Proceso del Equilibrio
//!
//! ```text
//! 1. media  = E[x]              ← promedio de la secuencia
//! 2. var    = E[(x - media)²]   ← dispersión de los valores
//! 3. x_norm = (x - media) / √(var + ε)   ← normalización
//! 4. y      = γ · x_norm + β    ← escala y desplazamiento aprendibles
//! ```
//!
//! ## La Parte Complicada: el Paso Hacia Atrás
//!
//! La media y la varianza dependen de TODOS los elementos del vector, creando
//! dependencias cruzadas que complican el cálculo del gradiente. La fórmula
//! resultante tiene en cuenta tanto la dependencia de la media como la de la varianza.

use crate::tensor::Tensor;

/// Capa de normalización — el árbitro del equilibrio neuronal
pub struct NormCapaEntrenable {
    pub gamma: Tensor, // γ: escala aprendible, inicializado a 1.0
    pub beta: Tensor,  // β: desplazamiento aprendible, inicializado a 0.0
    pub eps: f32,      // ε: pequeña constante para evitar división por cero
}

impl NormCapaEntrenable {
    /// Crea una nueva capa de normalización
    ///
    /// γ = 1 (sin escala inicial) y β = 0 (sin desplazamiento inicial).
    /// ε = 1e-5 es el valor estándar de la industria.
    pub fn new(forma_normalizada: usize) -> Self {
        Self {
            gamma: Tensor::new(vec![1.0; forma_normalizada], vec![forma_normalizada]),
            beta: Tensor::new(vec![0.0; forma_normalizada], vec![forma_normalizada]),
            eps: 1e-5,
        }
    }

    /// Paso hacia adelante — el árbitro normaliza cada vector de la secuencia
    pub fn forward(&self, x: &Tensor) -> (Tensor, CacheNormCapa) {
        // 1. Estadísticas a lo largo de la última dimensión
        let media = x.mean(-1, true);
        let varianza = x.var(-1, true);
        let desv_est = varianza.add_scalar(self.eps).sqrt(); // ε previene división por cero

        // 2. Normalización: centra y escala a media=0, varianza=1
        let x_centrado = x.sub(&media);
        let x_norm = x_centrado.div(&desv_est);

        // 3. Reescalado aprendible: el modelo ajusta la normalización según lo necesite
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

    /// Paso hacia atrás — la parte más delicada del pergamino
    ///
    /// La fórmula del gradiente tiene en cuenta que la media y la varianza
    /// dependen de todos los elementos de la fila:
    ///
    /// ```text
    /// grad_x = (1/√var) · (grad_x_norm - E[grad_x_norm] - x_norm · E[grad_x_norm · x_norm])
    /// ```
    pub fn backward(&self, grad_salida: &Tensor, cache: &CacheNormCapa) -> GradientesNormCapa {
        let n_embd = self.gamma.datos.len();
        let long_sec = grad_salida.forma[0];

        // Gradientes de γ y β: se acumulan sumando sobre toda la secuencia
        let mut grad_gamma = vec![0.0; n_embd];
        let mut grad_beta = vec![0.0; n_embd];
        for i in 0..long_sec {
            for j in 0..n_embd {
                let idx = i * n_embd + j;
                grad_gamma[j] += grad_salida.datos[idx] * cache.x_norm.datos[idx];
                grad_beta[j] += grad_salida.datos[idx];
            }
        }

        // Retropropagar a través de la escala: grad_x_norm = grad_salida · γ
        let grad_x_norm = grad_salida.mul(&self.gamma);

        // La parte compleja: retropropagar a través de la normalización
        let mut datos_grad_x = vec![0.0; long_sec * n_embd];

        for i in 0..long_sec {
            let inicio_fila = i * n_embd;
            let fila_grad_x_norm = &grad_x_norm.datos[inicio_fila..inicio_fila + n_embd];
            let fila_x_norm = &cache.x_norm.datos[inicio_fila..inicio_fila + n_embd];
            let valor_desv_est = cache.desv_est.datos[i];

            // Media de los gradientes: tiene en cuenta la dependencia de la media
            let media_grad: f32 = fila_grad_x_norm.iter().sum::<f32>() / n_embd as f32;

            // Media de (grad · x_norm): tiene en cuenta la dependencia de la varianza
            let media_grad_x: f32 = fila_grad_x_norm.iter().zip(fila_x_norm.iter())
                .map(|(g, x)| g * x).sum::<f32>() / n_embd as f32;

            // Fórmula final: combina los tres caminos del gradiente
            for j in 0..n_embd {
                let idx = inicio_fila + j;
                datos_grad_x[idx] =
                    (fila_grad_x_norm[j] - media_grad - fila_x_norm[j] * media_grad_x)
                    / valor_desv_est;
            }
        }

        GradientesNormCapa {
            gamma: Tensor::new(grad_gamma, vec![n_embd]),
            beta: Tensor::new(grad_beta, vec![n_embd]),
            x: Tensor::new(datos_grad_x, cache.x.forma.clone()),
        }
    }
}

/// Tesoro del forward — guarda x_norm y desv_est para el backward
pub struct CacheNormCapa {
    pub x: Tensor,
    pub x_norm: Tensor,    // El vector normalizado, necesario para grad_gamma
    #[allow(dead_code)]
    pub media: Tensor,
    pub desv_est: Tensor,  // La desviación estándar, necesaria para escalar el gradiente
}

/// Los tres gradientes de la normalización de capa
pub struct GradientesNormCapa {
    pub gamma: Tensor, // ∂L/∂γ
    pub beta: Tensor,  // ∂L/∂β
    pub x: Tensor,     // ∂L/∂x
}
