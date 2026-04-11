//! Capas de Redes Neuronales
//!
//! Este módulo contiene todas las implementaciones de capas para el modelo GPT-2 entrenable.
//! Cada capa proporciona tanto el paso hacia adelante (forward) como hacia atrás (backward) para el entrenamiento.
//!
//! ## Capas
//!
//! - **activation**: Función de activación GELU (paso hacia adelante y hacia atrás)
//! - **linear**: Capa totalmente conectada (fully connected)
//! - **layer_norm**: Normalización de capa
//! - **dropout**: Regularización por dropout
//! - **mlp**: Perceptrón multicapa (red prealimentada / feedforward)
//! - **attention**: Mecanismo de autoatención
//! - **block**: Bloque transformer completo
//!
//! ## Patrón de Diseño
//!
//! Cada capa entrenable sigue un patrón consistente:
//!
//! ```rust,ignore
//! pub struct CapaEntrenable {
//!     // Parámetros (pesos, sesgos, etc.)
//! }
//!
//! impl CapaEntrenable {
//!     pub fn new(...) -> Self { }
//!     pub fn forward(&self, x: &Tensor) -> (Tensor, Cache) { }
//!     pub fn backward(&self, grad: &Tensor, cache: &Cache) -> Gradientes { }
//! }
//!
//! pub struct Cache {
//!     // Valores necesarios para el paso hacia atrás
//! }
//!
//! pub struct Gradientes {
//!     // Gradientes para los parámetros y la entrada
//! }
//! ```
//!
//! Este patrón hace que la retropropagación sea explícita y educativa.

pub mod activation;
pub mod attention;
pub mod block;
pub mod dropout;
pub mod layer_norm;
pub mod linear;
pub mod mlp;

// Reexportar los tipos principales por conveniencia
pub use activation::{gelu_backward, gelu_forward};
pub use attention::{CacheAtencion, GradientesAtencion, AtencionUnaCabezaEntrenable};
pub use block::{CacheBloque, GradientesBloque, BloqueTransformerEntrenable};
pub use dropout::{CacheDropout, DropoutEntrenable};
pub use layer_norm::{CacheNormCapa, GradientesNormCapa, NormCapaEntrenable};
pub use linear::{CacheLineal, GradientesLineales, LinealEntrenable, inicializacion_aleatoria as random_init};
pub use mlp::{CacheMLP, GradientesMLP, MLPEntrenable};