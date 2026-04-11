//! Utilidades de Gradientes
//!
//! Este módulo proporciona utilidades para trabajar con gradientes durante el entrenamiento.
//! Estas operaciones son esenciales para la estabilidad del entrenamiento y su monitoreo.
//!
//! ## Componentes
//!
//! - **Cálculo de la Norma del Gradiente**: Mide la magnitud de los gradientes
//! - **Recorte de Gradientes (Gradient Clipping)**: Previene la explosión de gradientes mediante el escalado
//!
//! ## ¿Por qué recortar los gradientes?
//!
//! Durante el entrenamiento, algunos lotes (batches) ocasionales pueden producir gradientes muy grandes que
//! desestabilizan el modelo. El recorte de gradientes previene esto reduciendo a escala
//! los gradientes cuando su norma excede un umbral.
//!
//! Sin recorte:
//! ```text
//! Paso 1000: Pérdida = 3.2
//! Paso 1001: Pérdida = 287.5  (¡explosión de gradiente!)
//! Paso 1002: Pérdida = NaN    (entrenamiento fallido)
//! ```
//!
//! Con recorte:
//! ```text
//! Paso 1000: Pérdida = 3.2
//! Paso 1001: Pérdida = 3.3  (el gradiente fue recortado)
//! Paso 1002: Pérdida = 3.1  (recuperado)
//! ```
//!
//! ## Algoritmo
//!
//! ```text
//! norma = √(Σ gradiente²)  // Calcular la norma L2
//! si norma > norma_max:
//!     gradientes *= (norma_max / norma)  // Escalar proporcionalmente
//! ```
//!
//! Esto asegura que todos los gradientes se escalen por el mismo factor, preservando
//! sus magnitudes relativas mientras se limita la magnitud total de la actualización.
//!
//! ## Ejemplo
//!
//! ```rust,no_run
//! use molineteai::gradients::{calcular_norma_grad, recortar_gradientes};
//! # use molineteai::gpt2_trainable::GradientesGPT2;
//!
//! # let grads: GradientesGPT2 = todo!();
//! // Calcular la norma del gradiente para monitoreo
//! let norma = calcular_norma_grad(&grads);
//! println!("Norma del gradiente: {:.4}", norma);
//!
//! // Recortar si es demasiado grande
//! let mut grads_recortados = grads;
//! recortar_gradientes(&mut grads_recortados, 1.0);
//! ```

use crate::gpt2_entrenable::GradientesGPT2;
use rayon::prelude::*;

/// Calcula la norma L2 de todos los gradientes
///
/// La norma del gradiente es la raíz cuadrada de la suma de todos los valores de gradientes al cuadrado
/// en todos los parámetros del modelo. Esto da un único número que representa
/// la magnitud general de la actualización del gradiente.
///
/// # Argumentos
///
/// * `grads` - Gradientes para todos los parámetros del modelo
///
/// # Retorna
///
/// La norma L2: √(Σ g²) para todos los valores de gradiente g
///
/// # Rendimiento
///
/// Utiliza computación paralela a través de Rayon para un mejor rendimiento en CPUs multinúcleo.
/// La computación se paraleliza dentro de cada tensor para maximizar el rendimiento.
///
/// # Ejemplo
///
/// ```rust,no_run
/// # use molineteai::gradients::calcular_norma_grad;
/// # use molineteai::gpt2_trainable::GradientesGPT2;
/// # let grads: GradientesGPT2 = todo!();
/// let norma = calcular_norma_grad(&grads);
/// if norma > 5.0 {
///     println!("Advertencia: Norma del gradiente muy grande: {:.2}", norma);
/// }
/// ```
pub fn calcular_norma_grad(grads: &GradientesGPT2) -> f32 {
    // Función auxiliar para calcular la suma de cuadrados en paralelo
    let suma_cuadrados_paralelo = |datos: &Vec<f32>| -> f32 { datos.par_iter().map(|&val| val * val).sum() };

    let mut suma_cuadrados = 0.0;

    // Embeddings de tokens y posiciones
    suma_cuadrados += suma_cuadrados_paralelo(&grads.embedding_tokens.datos);
    suma_cuadrados += suma_cuadrados_paralelo(&grads.embedding_posiciones.datos);

    // Todos los bloques transformer
    for grad_bloque in &grads.grads_bloques {
        // NormCapa 1
        suma_cuadrados += suma_cuadrados_paralelo(&grad_bloque.ln1_gamma.datos);
        suma_cuadrados += suma_cuadrados_paralelo(&grad_bloque.ln1_beta.datos);

        // Atención
        suma_cuadrados += suma_cuadrados_paralelo(&grad_bloque.atencion.peso_q.datos);
        suma_cuadrados += suma_cuadrados_paralelo(&grad_bloque.atencion.sesgo_q.datos);
        suma_cuadrados += suma_cuadrados_paralelo(&grad_bloque.atencion.peso_k.datos);
        suma_cuadrados += suma_cuadrados_paralelo(&grad_bloque.atencion.sesgo_k.datos);
        suma_cuadrados += suma_cuadrados_paralelo(&grad_bloque.atencion.peso_v.datos);
        suma_cuadrados += suma_cuadrados_paralelo(&grad_bloque.atencion.sesgo_v.datos);
        suma_cuadrados += suma_cuadrados_paralelo(&grad_bloque.atencion.peso_salida.datos);
        suma_cuadrados += suma_cuadrados_paralelo(&grad_bloque.atencion.sesgo_salida.datos);

        // NormCapa 2
        suma_cuadrados += suma_cuadrados_paralelo(&grad_bloque.ln2_gamma.datos);
        suma_cuadrados += suma_cuadrados_paralelo(&grad_bloque.ln2_beta.datos);

        // MLP (red prealimentada)
        suma_cuadrados += suma_cuadrados_paralelo(&grad_bloque.mlp.peso_fc1.datos);
        suma_cuadrados += suma_cuadrados_paralelo(&grad_bloque.mlp.sesgo_fc1.datos);
        suma_cuadrados += suma_cuadrados_paralelo(&grad_bloque.mlp.peso_fc2.datos);
        suma_cuadrados += suma_cuadrados_paralelo(&grad_bloque.mlp.sesgo_fc2.datos);
    }

    // Normalización de capa final
    suma_cuadrados += suma_cuadrados_paralelo(&grads.ln_final_gamma.datos);
    suma_cuadrados += suma_cuadrados_paralelo(&grads.ln_final_beta.datos);

    // Peso de proyección de salida
    suma_cuadrados += suma_cuadrados_paralelo(&grads.peso_salida.datos);

    suma_cuadrados.sqrt()
}

/// Recorta gradientes a una norma máxima
///
/// Cuando la norma del gradiente excede `norma_max`, todos los gradientes se escalan
/// proporcionalmente para llevar la norma exactamente a `norma_max`. Esto previene
/// la explosión de gradientes mientras preserva la dirección de la actualización del gradiente.
///
/// # Argumentos
///
/// * `grads` - Gradientes a recortar (modificados en el lugar/in-place)
/// * `norma_max` - Norma máxima de gradiente permitida (típicamente 1.0)
///
/// # Algoritmo
///
/// ```text
/// norma = calcular_norma_grad(grads)
/// si norma > norma_max:
///     escala = norma_max / norma
///     para todos los gradientes g:
///         g *= escala
/// ```
///
/// # Rendimiento
///
/// Solo realiza el escalado si el recorte es necesario. Utiliza computación paralela
/// a través de Rayon para un mejor rendimiento en CPUs multinúcleo.
///
/// # Ejemplo
///
/// ```rust,no_run
/// # use molineteai::gradients::recortar_gradientes;
/// # use molineteai::gpt2_trainable::GradientesGPT2;
/// # let mut grads: GradientesGPT2 = todo!();
/// // Recortar gradientes a una norma de 1.0 (práctica estándar)
/// recortar_gradientes(&mut grads, 1.0);
/// ```
pub fn recortar_gradientes(grads: &mut GradientesGPT2, norma_max: f32) {
    let norma = calcular_norma_grad(grads);

    // Solo recortar si la norma excede el umbral
    if norma > norma_max {
        let escala = norma_max / norma;

        // Función auxiliar para escalar los datos del tensor en paralelo
        let escalar_paralelo = |datos: &mut Vec<f32>| {
            datos.par_iter_mut().for_each(|val| *val *= escala);
        };

        // Escalar todos los gradientes por el mismo factor

        // Embeddings de tokens y posiciones
        escalar_paralelo(&mut grads.embedding_tokens.datos);
        escalar_paralelo(&mut grads.embedding_posiciones.datos);

        // Todos los bloques transformer
        for grad_bloque in &mut grads.grads_bloques {
            // NormCapa 1
            escalar_paralelo(&mut grad_bloque.ln1_gamma.datos);
            escalar_paralelo(&mut grad_bloque.ln1_beta.datos);

            // Atención
            escalar_paralelo(&mut grad_bloque.atencion.peso_q.datos);
            escalar_paralelo(&mut grad_bloque.atencion.sesgo_q.datos);
            escalar_paralelo(&mut grad_bloque.atencion.peso_k.datos);
            escalar_paralelo(&mut grad_bloque.atencion.sesgo_k.datos);
            escalar_paralelo(&mut grad_bloque.atencion.peso_v.datos);
            escalar_paralelo(&mut grad_bloque.atencion.sesgo_v.datos);
            escalar_paralelo(&mut grad_bloque.atencion.peso_salida.datos);
            escalar_paralelo(&mut grad_bloque.atencion.sesgo_salida.datos);

            // NormCapa 2
            escalar_paralelo(&mut grad_bloque.ln2_gamma.datos);
            escalar_paralelo(&mut grad_bloque.ln2_beta.datos);

            // MLP (red prealimentada)
            escalar_paralelo(&mut grad_bloque.mlp.peso_fc1.datos);
            escalar_paralelo(&mut grad_bloque.mlp.sesgo_fc1.datos);
            escalar_paralelo(&mut grad_bloque.mlp.peso_fc2.datos);
            escalar_paralelo(&mut grad_bloque.mlp.sesgo_fc2.datos);
        }

        // Normalización de capa final
        escalar_paralelo(&mut grads.ln_final_gamma.datos);
        escalar_paralelo(&mut grads.ln_final_beta.datos);

        // Peso de proyección de salida
        escalar_paralelo(&mut grads.peso_salida.datos);
    }
}