//! Funciones de Activación
//!
//! Este módulo proporciona funciones de activación y sus derivadas para
//! la retropropagación (backpropagation).
//!
//! ## GELU (Unidad Lineal de Error Gaussiano)
//!
//! GELU se usa en transformers en lugar de ReLU porque proporciona gradientes 
//! más suaves y a menudo funciona mejor en la práctica.
//!
//! ### Fórmula
//!
//! ```text
//! GELU(x) = x × Φ(x)
//! ```
//!
//! donde Φ(x) es la función de distribución acumulativa de la distribución normal estándar.
//!
//! ### Aproximación
//!
//! Usamos la aproximación con tanh para mayor eficiencia:
//!
//! ```text
//! GELU(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
//! ```
//!
//! Esto es más rápido que calcular la función de distribución acumulativa exacta y es lo 
//! suficientemente preciso para redes neuronales.



//! ### ¿Por qué GELU?
//!
//! - **Gradientes suaves**: A diferencia de ReLU (que tiene gradiente cero para x<0), GELU tiene
//!   gradientes distintos de cero en todas partes.
//! - **Mejor rendimiento empírico**: Especialmente en grandes transformers como GPT-2/BERT.
//! - **Interpretación probabilística**: GELU puede verse como una aproximación suave al
//!   dropout a nivel de neurona.

use crate::tensor::Tensor;
use rayon::prelude::*;

/// Activación GELU 
///
/// Calcula la activación GELU usando la aproximación con tanh.
///
/// # Argumentos
///
/// * `x` - Tensor de entrada
///
/// # Retorna
///
/// Tensor con la activación GELU aplicada elemento a elemento
///
/// # Rendimiento
///
/// Utiliza computación paralela vía Rayon para un mejor rendimiento en CPUs multinúcleo.
pub fn gelu_forward(x: &Tensor) -> Tensor {
    let resultado = x
        .datos
        .par_iter()
        .map(|&valor| {
            0.5 * valor
                * (1.0
                    + ((2.0 / std::f32::consts::PI).sqrt() * (valor + 0.044715 * valor.powi(3))).tanh())
        })
        .collect();
    Tensor::new(resultado, x.forma.clone())
}

/// Derivada de la activación GELU 
///
/// Calcula el gradiente de GELU con respecto a su entrada.
///
/// # Argumentos
///
/// * `grad_salida` - Gradiente de la siguiente capa
/// * `x` - Entrada original a GELU (del paso hacia adelante)
///
/// # Retorna
///
/// Gradiente con respecto a la entrada: grad_x = grad_salida * GELU'(x)
///
/// # Derivación Matemática
///
/// La derivada involucra:
/// 1. Derivada de tanh (término sech²)
/// 2. Derivada del polinomio interno
/// 3. Aplicación de la regla del producto
///
/// Esto nos da un gradiente complejo pero suave que ayuda al entrenamiento.
pub fn gelu_backward(grad_salida: &Tensor, x: &Tensor) -> Tensor {
    let datos_grad: Vec<f32> = x
        .datos
        .par_iter()
        .zip(&grad_salida.datos)
        .map(|(&valor_x, &valor_grad)| {
            let raiz_2_pi = (2.0 / std::f32::consts::PI).sqrt();
            let interno = raiz_2_pi * (valor_x + 0.044715 * valor_x.powi(3));
            let tanh_interno = interno.tanh();
            let sech_cuadrado = 1.0 - tanh_interno * tanh_interno;

            let grad_gelu = 0.5 * (1.0 + tanh_interno)
                + 0.5 * valor_x * sech_cuadrado * raiz_2_pi * (1.0 + 3.0 * 0.044715 * valor_x.powi(2));

            valor_grad * grad_gelu
        })
        .collect();

    Tensor::new(datos_grad, x.forma.clone())
}