//! Molinete AI: Implementación Educativa de Modelo de Lenguaje
//!
//! Un transformer estilo GPT-2 completo implementado desde cero en Rust
//! con fines educativos. Desarrollado íntegramente en español para aprender
//! inteligencia artificial desde los fundamentos.
//!
//! # Módulos
//!
//! - [`tokenizador`] - Tokenización por Codificación de Pares de Bytes (BPE)
//! - [`tensor`] - Arreglos multidimensionales y operaciones
//! - [`modelo`] - Arquitectura del modelo GPT-2 (solo propagación hacia adelante / forward pass)
//! - [`gpt2_entrenable`] - GPT-2 entrenable con propagación hacia atrás (backward pass)
//! - [`entrenamiento`] - Carga de datos para el entrenamiento
//! - [`registrador_entrenamiento`] - Métricas de entrenamiento y registro (logging)
//!
//! # Ejemplo: Tokenización
//!
//! ```rust,no_run
//! use molineteai::TokenizadorBPE;
//!
//! // Entrenar un tokenizador
//! let text = std::fs::read_to_string("corpus.txt").unwrap();
//! let mut tokenizer = TokenizadorBPE::new(1024);
//! tokenizer.train(&text, 1024);
//!
//! // Codificar y decodificar
//! let ids = tokenizer.codificar("¡Hola, mundo!");
//! let decoded = tokenizer.decodificar(&ids);
//! assert_eq!(decoded, "¡Hola, mundo!");
//! ```
//!
//! # Ejemplo: Operaciones con Tensores
//!
//! ```rust
//! use molineteai::Tensor;
//!
//! // Crear una matriz 2x2
//! let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
//! let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
//!
//! // Multiplicación de matrices
//! let c = a.matmul(&b);
//! assert_eq!(c.shape, vec![2, 2]);
//! ```
//!
//! # Ejemplo: Arquitectura del Modelo
//!
//! ```rust
//! use molineteai::{GPT2, Config};
//!
//! // Crear un modelo diminuto
//! let config = Config::tiny(512); // tamaño de vocabulario de 512
//! let model = GPT2::new(&config);
//!
//! // Propagación hacia adelante (forward pass): tokens → logits
//! let tokens = vec![vec![1, 2, 3, 4]]; // tamaño_lote=1, long_secuencia=4
//! let logits = model.forward(&tokens);
//! assert_eq!(logits.shape, vec![1, 4, 512]); // [lote, secuencia, vocabulario]
//! ```

pub mod gpt2_entrenable;
pub mod gradientes;
pub mod layers;
pub mod modelo;
pub mod optimizador;
pub mod tensor;
pub mod tokenizador; 
pub mod entrenamiento;
pub mod registrador_entrenamiento;

pub use gradientes::{recortar_gradientes, calcular_norma_grad};
pub use modelo::{gelu, Config, GPT2};
pub use optimizador::{actualizar_adamw, OptimizadorAdamW};
pub use tensor::Tensor;
pub use tokenizador::{TokenizadorBPE, EstadisticasTokenizador};
pub use entrenamiento::{Lote, CargadorDatosTexto, ConfigEntrenamiento};
pub use registrador_entrenamiento::{calcular_perdida_dataset, dividir_entrenamiento_validacion, RegistradorEntrenamiento};

#[cfg(feature = "python")]
pub mod python_bindings;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn molineteai(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python_bindings::molineteai(m)
}

