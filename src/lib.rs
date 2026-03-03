//! Molinete AI: Implementación Educativa de Modelo de Lenguaje
//!
//! Un transformer estilo GPT-2 implementado completamente desde cero en Rust
//! con fines educativos. Nombrado en honor al “molinete”, símbolo de movimiento,
//! dinámica y transformación — como el flujo de información en un transformer.
//!
//! # Módulos
//!
//! - [`tokenizer`] - Tokenización Byte Pair Encoding (BPE)
//! - [`tensor`] - Arreglos multidimensionales y operaciones
//! - [`model`] - Arquitectura del modelo GPT-2 (solo forward pass)
//! - [`gpt2_trainable`] - GPT-2 entrenable con backward pass
//! - [`train`] - Carga de datos para entrenamiento
//! - [`training_logger`] - Métricas y registro de entrenamiento
//!
//! # Ejemplo: Tokenización
//!
//! ```rust,no_run
//! use molinete_ai::BPETokenizer;
//!
//! // Entrenar un tokenizador
//! let text = std::fs::read_to_string("corpus.txt").unwrap();
//! let mut tokenizer = BPETokenizer::new(1024);
//! tokenizer.train(&text, 1024);
//!
//! // Codificar y decodificar
//! let ids = tokenizer.encode("Hola, mundo!");
//! let decoded = tokenizer.decode(&ids);
//! assert_eq!(decoded, "Hola, mundo!");
//! ```
//!
//! # Ejemplo: Operaciones con Tensores
//!
//! ```rust
//! use molinete_ai::Tensor;
//!
//! // Crear una matriz 2x2
//! let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
//! let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
//!
//! // Multiplicación matricial
//! let c = a.matmul(&b);
//! assert_eq!(c.shape, vec![2, 2]);
//! ```
//!
//! # Ejemplo: Arquitectura del Modelo
//!
//! ```rust
//! use molinete_ai::{GPT2, Config};
//!
//! // Crear un modelo pequeño
//! let config = Config::tiny(512); // vocabulario de tamaño 512
//! let model = GPT2::new(&config);
//!
//! // Forward pass: tokens → logits
//! let tokens = vec![vec![1, 2, 3, 4]]; // batch_size=1, seq_len=4
//! let logits = model.forward(&tokens);
//! assert_eq!(logits.shape, vec![1, 4, 512]); // [batch, seq, vocab]
//! ```

pub mod gpt2_trainable;
pub mod gradients;
pub mod layers;
pub mod model;
pub mod optimizer;
pub mod tensor;
pub mod tokenizer;
pub mod train;
pub mod training_logger;

// Re-exportar tipos principales para mayor comodidad
pub use gradients::{clip_gradients, compute_grad_norm};
pub use model::{gelu, Config, GPT2};
pub use optimizer::{adamw_update, AdamWOptimizer};
pub use tensor::Tensor;
pub use tokenizer::{BPETokenizer, TokenizerStats};
pub use train::{Batch, TextDataLoader, TrainingConfig};
pub use training_logger::{compute_dataset_loss, train_val_split, TrainingLogger};