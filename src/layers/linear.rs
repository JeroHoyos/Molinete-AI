//! Capa Lineal (Totalmente Conectada)
//!
//! La capa lineal es el bloque de construcción fundamental de las redes neuronales.
//! Realiza una transformación afín: y = x @ W + b
//!
//! ## Paso Hacia Adelante (Forward Pass)
//!
//! ```text
//! Entrada:  x [long_sec, caract_entrada]
//! Peso:     W [caract_entrada, caract_salida]
//! Sesgo:    b [caract_salida]
//! Salida:   y = x @ W + b [long_sec, caract_salida]
//! ```
//!
//! ## Paso Hacia Atrás (Backward Pass)
//!
//! Usando la regla de la cadena:
//! ```text
//! grad_W = x^T @ grad_y
//! grad_b = sum(grad_y, axis=0)
//! grad_x = grad_y @ W^T
//! ```
//!
//! ## ¿Por Qué Estos Gradientes?
//!
//! - **grad_W**: Cada peso W[i,j] afecta la salida y[*,j] a través de la entrada x[*,i]
//! - **grad_b**: Cada sesgo b[j] afecta todas las salidas y[*,j] por igual
//! - **grad_x**: Necesario para retropropagar a la capa anterior
//!
//! ## Notas de Implementación
//!
//! - Usa inicialización de He: escala = √(2/caract_entrada)
//! - El sesgo se inicializa a cero (práctica común)
//! - Almacena en caché la entrada x para el paso hacia atrás

use crate::tensor::Tensor;

/// Función auxiliar para inicialización aleatoria
///
/// Usa un LCG (Generador Congruencial Lineal) simple para una inicialización reproducible.
/// El parámetro de escala controla la magnitud de los pesos iniciales.
pub fn inicializacion_aleatoria(tamano: usize, semilla: u64, escala: f32) -> Vec<f32> {
    let mut rng = semilla;
    (0..tamano)
        .map(|_| {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let valor = ((rng / 65536) % 32768) as f32 / 32768.0;
            (valor - 0.5) * 2.0 * escala
        })
        .collect()
}

/// Capa lineal (totalmente conectada)
///
/// Realiza y = x @ W + b donde:
/// - W: matriz de pesos [caract_entrada, caract_salida]
/// - b: vector de sesgo [caract_salida]
pub struct LinealEntrenable {
    pub peso: Tensor,
    pub sesgo: Tensor,
}

impl LinealEntrenable {
    /// Crea una nueva capa lineal con inicialización de He
    ///
    /// # Argumentos
    ///
    /// * `caract_entrada` - Dimensión de entrada
    /// * `caract_salida` - Dimensión de salida
    /// * `semilla` - Semilla aleatoria para reproducibilidad
    ///
    /// # Inicialización
    ///
    /// Usa inicialización de He: escala = √(2/caract_entrada)
    /// Esto ayuda a prevenir gradientes desvanecientes/explosivos en redes profundas.
    pub fn new(caract_entrada: usize, caract_salida: usize, semilla: u64) -> Self {
        let escala = (2.0 / caract_entrada as f32).sqrt();
        Self {
            peso: Tensor::new(
                inicializacion_aleatoria(caract_entrada * caract_salida, semilla, escala),
                vec![caract_entrada, caract_salida],
            ),
            sesgo: Tensor::new(vec![0.0; caract_salida], vec![caract_salida]),
        }
    }

    /// Paso hacia adelante (Forward pass)
    ///
    /// Calcula y = x @ W + b y almacena x en caché para el paso hacia atrás
    ///
    /// # Argumentos
    ///
    /// * `x` - Tensor de entrada [long_sec, caract_entrada]
    ///
    /// # Retorna
    ///
    /// Tupla de (salida, cache) donde:
    /// - salida: [long_sec, caract_salida]
    /// - cache: almacena x para el paso hacia atrás
    pub fn forward(&self, x: &Tensor) -> (Tensor, CacheLineal) {
        let y = x.matmul(&self.peso).add(&self.sesgo);
        let cache = CacheLineal { x: x.clone() };
        (y, cache)
    }

    /// Paso hacia atrás (Backward pass)
    ///
    /// Calcula los gradientes para los pesos, el sesgo y la entrada
    ///
    /// # Argumentos
    ///
    /// * `grad_salida` - Gradiente de la siguiente capa [long_sec, caract_salida]
    /// * `cache` - Valores almacenados en caché del paso hacia adelante
    ///
    /// # Retorna
    ///
    /// Gradientes para el peso, el sesgo y la entrada
    pub fn backward(&self, grad_salida: &Tensor, cache: &CacheLineal) -> GradientesLineales {
        // grad_W = x^T @ grad_salida
        let grad_peso = cache.x.transpose(-2, -1).matmul(grad_salida);

        // grad_b = sum(grad_salida) a lo largo de todas las dims excepto la última
        let datos_grad_sesgo: Vec<f32> = (0..self.sesgo.datos.len())
            .map(|i| {
                let mut suma = 0.0;
                for fila in 0..grad_salida.forma[0] {
                    suma += grad_salida.datos[fila * grad_salida.forma[1] + i];
                }
                suma
            })
            .collect();
        let grad_sesgo = Tensor::new(datos_grad_sesgo, self.sesgo.forma.clone());

        // grad_x = grad_salida @ W^T
        let grad_x = grad_salida.matmul(&self.peso.transpose(-2, -1));

        GradientesLineales {
            peso: grad_peso,
            sesgo: grad_sesgo,
            x: grad_x,
        }
    }
}

/// Caché para el paso hacia atrás de la capa lineal
pub struct CacheLineal {
    pub x: Tensor,
}

/// Gradientes para la capa lineal
pub struct GradientesLineales {
    pub peso: Tensor,
    pub sesgo: Tensor,
    pub x: Tensor, // Gradiente para pasar a la capa anterior
}