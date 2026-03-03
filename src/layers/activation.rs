//! Funciones de Activación
//!
//! Este módulo proporciona funciones de activación y sus derivadas para
//! la retropropagación (backpropagation).
//!
//! ## GELU (Unidad Lineal de Error Gaussiano)
//!
//! GELU se utiliza en transformers en lugar de ReLU porque proporciona
//! gradientes más suaves y generalmente mejor rendimiento en la práctica.
//!
//! ### Fórmula
//!
//! ```text
//! GELU(x) = x × Φ(x)
//! ```
//!
//! donde Φ(x) es la función de distribución acumulada de la distribución
//! normal estándar.
//!
//! ### Aproximación
//!
//! Usamos la aproximación con tanh por eficiencia:
//!
//! ```text
//! GELU(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
//! ```
//!
//! Esto es más rápido que calcular la CDF exacta y es suficientemente
//! preciso para redes neuronales.
//!
//! ### ¿Por qué GELU?
//!
//! - **Gradientes suaves**: A diferencia de ReLU (que tiene gradiente cero para x<0),
//!   GELU tiene gradientes distintos de cero en todo el dominio
//! - **Mejor rendimiento empírico**: Especialmente en transformers grandes como GPT-2/BERT
//! - **Interpretación probabilística**: GELU puede verse como una aproximación
//!   suave al dropout a nivel de neurona

use crate::tensor::Tensor;
use rayon::prelude::*;

/// Activación GELU (forward pass)
///
/// Calcula la activación GELU usando la aproximación con tanh.
///
/// # Argumentos
///
/// * `x` - Tensor de entrada
///
/// # Retorna
///
/// Tensor con la activación GELU aplicada elemento por elemento
///
/// # Rendimiento
///
/// Usa computación paralela mediante Rayon para mejor rendimiento
/// en CPUs multinúcleo.
pub fn gelu_forward(x: &Tensor) -> Tensor {
    let result = x
        .data
        .par_iter()
        .map(|&val| {
            0.5 * val
                * (1.0
                    + ((2.0 / std::f32::consts::PI).sqrt() * (val + 0.044715 * val.powi(3))).tanh())
        })
        .collect();
    Tensor::new(result, x.shape.clone())
}

/// Derivada de la activación GELU (backward pass)
///
/// Calcula el gradiente de GELU con respecto a su entrada.
///
/// # Argumentos
///
/// * `grad_out` - Gradiente proveniente de la siguiente capa
/// * `x` - Entrada original a GELU (del forward pass)
///
/// # Retorna
///
/// Gradiente con respecto a la entrada: grad_x = grad_out * GELU'(x)
///
/// # Derivación Matemática
///
/// La derivada involucra:
/// 1. Derivada de tanh (término sech²)
/// 2. Derivada del polinomio interno
/// 3. Aplicación de la regla del producto
///
/// Esto nos da un gradiente complejo pero suave que facilita el entrenamiento.
pub fn gelu_backward(grad_out: &Tensor, x: &Tensor) -> Tensor {
    let grad_data: Vec<f32> = x
        .data
        .par_iter()
        .zip(&grad_out.data)
        .map(|(&x_val, &grad_val)| {
            let sqrt_2_pi = (2.0 / std::f32::consts::PI).sqrt();
            let inner = sqrt_2_pi * (x_val + 0.044715 * x_val.powi(3));
            let tanh_inner = inner.tanh();
            let sech_sq = 1.0 - tanh_inner * tanh_inner;

            let grad_gelu = 0.5 * (1.0 + tanh_inner)
                + 0.5 * x_val * sech_sq * sqrt_2_pi * (1.0 + 3.0 * 0.044715 * x_val.powi(2));

            grad_val * grad_gelu
        })
        .collect();

    Tensor::new(grad_data, x.shape.clone())
}