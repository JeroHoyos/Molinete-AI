//! Capas de la Red Neuronal
//!
//! Este módulo contiene todas las implementaciones de capas para el modelo GPT-2 entrenable.
//! Cada capa proporciona tanto el paso forward como el backward para entrenamiento.
//!
//! ## Capas
//!
//! - **activation**: Función de activación GELU (forward y backward)
//! - **linear**: Capa totalmente conectada
//! - **layer_norm**: Normalización por capa
//! - **dropout**: Regularización con dropout
//! - **mlp**: Perceptrón multicapa (red feedforward)
//! - **attention**: Mecanismo de auto-atención
//! - **block**: Bloque transformer completo
//!
//! ## Patrón de Diseño
//!
//! Cada capa entrenable sigue un patrón consistente:
//!
//! ```rust,ignore
//! pub struct TrainableLayer {
//!     // Parámetros (pesos, biases, etc.)
//! }
//!
//! impl TrainableLayer {
//!     pub fn new(...) -> Self { }
//!     pub fn forward(&self, x: &Tensor) -> (Tensor, Cache) { }
//!     pub fn backward(&self, grad: &Tensor, cache: &Cache) -> Gradients { }
//! }
//!
//! pub struct Cache {
//!     // Valores necesarios para el paso backward
//! }
//!
//! pub struct Gradients {
//!     // Gradientes para parámetros y entrada
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

// Re-exportar tipos principales para mayor comodidad
pub use activation::{gelu_backward, gelu_forward};
pub use attention::{AttentionCache, AttentionGradients, TrainableSingleHeadAttention};
pub use block::{BlockCache, BlockGradients, TrainableTransformerBlock};
pub use dropout::{DropoutCache, TrainableDropout};
pub use layer_norm::{LayerNormCache, LayerNormGradients, TrainableLayerNorm};
pub use linear::{random_init, LinearCache, LinearGradients, TrainableLinear};
pub use mlp::{MLPCache, MLPGradients, TrainableMLP};