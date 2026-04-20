//! # La Llama de la Activación — GELU
//!
//! Así como Don Quijote no reacciona de igual manera ante todos los estímulos
//! —con valentía ante los gigantes y ternura ante Dulcinea— las neuronas de
//! Molinete no amplifican linealmente todos los valores. La activación GELU
//! decide cuánto "encender" cada neurona según su valor de entrada.
//!
//! ## ¿Por qué GELU y no ReLU?
//!
//! ReLU apaga brutalmente todo valor negativo (gradiente cero para x < 0).
//! GELU es más compasiva: permite que los valores negativos pequeños pasen
//! con una chispa de gradiente, facilitando el aprendizaje en redes profundas.
//!
//! ## La Fórmula del Alquimista
//!
//! ```text
//! GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
//! ```
//!
//! Esta aproximación con tanh es más rápida que calcular la función de
//! distribución acumulativa exacta, y suficientemente precisa para el entrenamiento.

use crate::tensor::Tensor;
use rayon::prelude::*;

/// Activación GELU — el temple del caballero aplicado elemento a elemento
///
/// Calcula GELU usando la aproximación con tanh. Rayon paraliza el cálculo
/// entre los núcleos de la CPU como escuderos trabajando al unísono.
pub fn gelu_forward(x: &Tensor) -> Tensor {
    let resultado = x
        .datos
        .par_iter()
        .map(|&valor| {
            // La fórmula del alquimista: suave, continua, diferenciable en todas partes
            0.5 * valor
                * (1.0
                    + ((2.0 / std::f32::consts::PI).sqrt()
                        * (valor + 0.044715 * valor.powi(3)))
                    .tanh())
        })
        .collect();
    Tensor::new(resultado, x.forma.clone())
}

/// Derivada de GELU — el camino de vuelta por la regla de la cadena
///
/// Para entrenar el modelo necesitamos saber cómo fluye el error hacia atrás.
/// La derivada involucra la regla del producto y la derivada de tanh (sech²).
///
/// # Argumentos
///
/// * `grad_salida` - El error que llega de la siguiente capa
/// * `x` - La entrada original a GELU guardada durante el forward
pub fn gelu_backward(grad_salida: &Tensor, x: &Tensor) -> Tensor {
    let datos_grad: Vec<f32> = x
        .datos
        .par_iter()
        .zip(&grad_salida.datos)
        .map(|(&valor_x, &valor_grad)| {
            let raiz_2_pi = (2.0 / std::f32::consts::PI).sqrt();
            let interno = raiz_2_pi * (valor_x + 0.044715 * valor_x.powi(3));
            let tanh_interno = interno.tanh();
            // sech²(u) = 1 - tanh²(u) — la curvatura de la activación
            let sech_cuadrado = 1.0 - tanh_interno * tanh_interno;

            let grad_gelu = 0.5 * (1.0 + tanh_interno)
                + 0.5 * valor_x * sech_cuadrado * raiz_2_pi
                    * (1.0 + 3.0 * 0.044715 * valor_x.powi(2));

            valor_grad * grad_gelu
        })
        .collect();

    Tensor::new(datos_grad, x.forma.clone())
}
