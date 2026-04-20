//! # La Transformación Lineal — La Lanza del Algebra
//!
//! La capa lineal es la espada de combate de toda red neuronal: una transformación
//! afín `y = x @ W + b` que proyecta los datos de un espacio a otro.
//!
//! Como Don Quijote viendo gigantes donde hay molinos, esta capa transforma
//! la perspectiva de los datos: un vector de embedding puede convertirse en
//! puntuaciones de atención, en logits del vocabulario, o en representaciones ocultas.
//!
//! ## Paso Hacia Adelante
//!
//! ```text
//! Entrada:  x [secuencia, dim_entrada]
//! Peso:     W [dim_entrada, dim_salida]
//! Sesgo:    b [dim_salida]
//! Salida:   y = x @ W + b [secuencia, dim_salida]
//! ```
//!
//! ## Paso Hacia Atrás (Regla de la Cadena)
//!
//! ```text
//! grad_W = xᵀ @ grad_y         ← cuánto cambia la pérdida por cada peso
//! grad_b = Σ grad_y             ← suma sobre todas las posiciones
//! grad_x = grad_y @ Wᵀ         ← para propagar el error hacia atrás
//! ```

use crate::tensor::Tensor;

/// Inicialización aleatoria con semilla reproducible
///
/// Usa un LCG (Generador Congruencial Lineal) — un escudero sencillo pero predecible.
/// La escala de He (√(2/n)) previene que los gradientes se desvanezcan o exploten.
pub fn inicializacion_aleatoria(tamano: usize, semilla: u64, escala: f32) -> Vec<f32> {
    let mut rng = semilla;
    (0..tamano)
        .map(|_| {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let valor = ((rng / 65536) % 32768) as f32 / 32768.0;
            (valor - 0.5) * 2.0 * escala // centrado en cero, escalado
        })
        .collect()
}

/// Capa lineal — la transformación noble de los datos
pub struct LinealEntrenable {
    pub peso: Tensor, // W: la matriz de transformación [dim_entrada, dim_salida]
    pub sesgo: Tensor, // b: el desplazamiento aprendible [dim_salida]
}

impl LinealEntrenable {
    /// Forja una nueva capa lineal con inicialización de He
    ///
    /// He scale = √(2/dim_entrada) — equilibra la varianza de las activaciones
    /// para que no se disparen ni se desvanezan en redes profundas.
    pub fn new(caract_entrada: usize, caract_salida: usize, semilla: u64) -> Self {
        let escala = (2.0 / caract_entrada as f32).sqrt();
        Self {
            peso: Tensor::new(
                inicializacion_aleatoria(caract_entrada * caract_salida, semilla, escala),
                vec![caract_entrada, caract_salida],
            ),
            // El sesgo empieza en cero — sin prejuicios iniciales
            sesgo: Tensor::new(vec![0.0; caract_salida], vec![caract_salida]),
        }
    }

    /// Paso hacia adelante: y = x @ W + b
    ///
    /// Multiplica la entrada por los pesos y suma el sesgo.
    /// Guarda x en caché para el paso hacia atrás.
    pub fn forward(&self, x: &Tensor) -> (Tensor, CacheLineal) {
        let y = x.matmul(&self.peso).add(&self.sesgo);
        let cache = CacheLineal { x: x.clone() };
        (y, cache)
    }

    /// Paso hacia atrás — la regla de la cadena en acción
    ///
    /// Calcula los gradientes para los pesos, el sesgo y la entrada,
    /// permitiendo que el error fluya hacia las capas anteriores.
    pub fn backward(&self, grad_salida: &Tensor, cache: &CacheLineal) -> GradientesLineales {
        // grad_W = xᵀ @ grad_salida: cada peso responde por su influencia en la salida
        let grad_peso = cache.x.transpose(-2, -1).matmul(grad_salida);

        // grad_b = Σ grad_salida: el sesgo acumula el error de todas las posiciones
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

        // grad_x = grad_salida @ Wᵀ: el error se propaga hacia la capa anterior
        let grad_x = grad_salida.matmul(&self.peso.transpose(-2, -1));

        GradientesLineales { peso: grad_peso, sesgo: grad_sesgo, x: grad_x }
    }
}

/// Tesoro del forward — guarda x para el cálculo del gradiente
pub struct CacheLineal {
    pub x: Tensor,
}

/// Los tres gradientes de la capa lineal
pub struct GradientesLineales {
    pub peso: Tensor,  // ∂L/∂W
    pub sesgo: Tensor, // ∂L/∂b
    pub x: Tensor,     // ∂L/∂x — para propagar hacia atrás
}
