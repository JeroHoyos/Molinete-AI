//! Implementación del Optimizador AdamW
//!
//! Este módulo implementa el optimizador AdamW (Adam con decaimiento de pesos desacoplado / decoupled Weight decay),
//! el optimizador estándar para entrenar modelos transformer como GPT-2.
//!
//! ## ¿Qué es AdamW?
//!
//! AdamW mejora a Adam al desacoplar el decaimiento de pesos de la optimización
//! basada en gradientes. Combina:
//! - **Momento (Momentum)**: Suaviza las actualizaciones de los gradientes usando un promedio móvil exponencial
//! - **RMSProp**: Adapta la tasa de aprendizaje por parámetro basándose en el historial de gradientes
//! - **Decaimiento de pesos desacoplado**: Regularización L2 aplicada directamente a los pesos
//!
//! ## Algoritmo
//!
//! Para cada parámetro θ con gradiente g:
//!
//! ```text
//! # Actualización AdamW:
//! θ = θ * (1 - α * λ)              # Decaimiento de pesos (si aplica)
//! m = β₁ * m + (1 - β₁) * g        # Primer momento (momento)
//! v = β₂ * v + (1 - β₂) * g²       # Segundo momento (varianza)
//! m_hat = m / (1 - β₁^t)           # Corrección de sesgo
//! v_hat = v / (1 - β₂^t)           # Corrección de sesgo
//! θ = θ - α * m_hat / (√v_hat + ε) # Actualización del parámetro
//! ```
//!
//! donde:
//! - α (alfa/lr) = tasa de aprendizaje (típicamente 3e-4 para transformers)
//! - λ (lambda/weight_decay) = 0.1 (típico para transformers)
//! - β₁ (beta1) = 0.9 (tasa de decaimiento del momento)
//! - β₂ (beta2) = 0.95 (tasa de decaimiento de la varianza, menor que el 0.999 de Adam)
//! - ε (épsilon) = 1e-8 (estabilidad numérica)
//! - t = número de paso de entrenamiento
//!
//! ## ¿Por qué AdamW en lugar de Adam?
//!
//! **vs Adam**:
//! - Mejor generalización a través de una regularización L2 adecuada
//! - El decaimiento de pesos no interactúa con las tasas de aprendizaje adaptativas
//! - Elección estándar para el entrenamiento de transformers modernos (GPT, BERT, etc.)
//!
//! **vs SGD**:
//! - Convergencia más rápida (menos pasos de entrenamiento)
//! - Menos sensible a la elección de la tasa de aprendizaje
//! - Tasas de aprendizaje adaptativas por parámetro
//!
//! ## Decaimiento de Pesos Selectivo
//!
//! Siguiendo las mejores prácticas del entrenamiento de transformers modernos, el decaimiento de pesos
//! se aplica **solo a tensores 2D (matrices de pesos)**, y no a:
//! - Sesgos / Biases (tensores 1D)
//! - Parámetros de LayerNorm (tensores 1D)
//! - Incrustaciones / Embeddings (opcionalmente, configurable)
//!
//! Esta aplicación selectiva evita la sobreregularización de los parámetros de escala/desplazamiento (scale/shift)
//! mientras sigue proporcionando los beneficios de la regularización L2 para las matrices de pesos.
//!
//! ## Corrección de Sesgo
//!
//! Los términos de corrección de sesgo `(1 - β^t)` son críticos. Sin ellos, m y v
//! están sesgados hacia cero durante los primeros pasos del entrenamiento. La corrección asegura
//! que el optimizador funcione bien desde el primer paso.
//!
//! ## Notas de Implementación
//!
//! Esta implementación refleja exactamente la estructura del modelo GPT-2:
//! - Vectores de momento separados para cada parámetro
//! - Actualizaciones en paralelo usando Rayon para mayor rendimiento
//! - Alternativa (fallback) automática a secuencial para tensores pequeños
//!
//! ## Ejemplo
//!
//! ```rust,no_run
//! use molineteai::optimizer::{OptimizadorAdamW, actualizar_adamw};
//! use molineteai::gpt2_entrenable::GPT2Entrenable;
//! # use molineteai::Config;
//!
//! # let config = Config::tiny(512);
//! # let mut model = GPT2Entrenable::new(&config);
//! // Inicializar el optimizador
//! let mut optimizer = OptimizadorAdamW::new(&model);
//!
//! // Bucle de entrenamiento
//! # let num_steps = 100;
//! for step in 0..num_steps {
//!     // ... paso hacia adelante (forward pass), calcular gradientes ...
//!     # let grads = todo!();
//!
//!     // Actualizar parámetros con decaimiento de pesos
//!     actualizar_adamw(&mut model, &grads, &mut optimizer, 3e-4, 0.1);
//! }
//! ```
//!
//! ## Referencias
//!
//! - Loshchilov & Hutter (2019): "Decoupled Weight Decay Regularization"
//!   https://arxiv.org/abs/1711.05101
//! - Kingma & Ba (2014): "Adam: A Method for Stochastic Optimization"
//!   https://arxiv.org/abs/1412.6980
use crate::gpt2_entrenable::{GradientesGPT2, GPT2Entrenable};
use crate::tensor::Tensor;
use rayon::prelude::*;

/// AdamW optimizer state
///
/// Maintains first and second moment estimates (m and v) for all model parameters.
/// The structure mirrors `GradientesGPT2` to ensure every parameter has corresponding
/// optimizer state.
///
/// # Fields
///
/// - **m_* (first moment)**: Exponential moving average of gradients (momentum)
/// - **v_* (second moment)**: Exponential moving average of squared gradients (variance)
/// - **beta1**: Momentum decay rate (default: 0.9)
/// - **beta2**: Variance decay rate (default: 0.95, lower than Adam's 0.999)
/// - **epsilon**: Numerical stability constant (default: 1e-8)
/// - **step**: Training step count (for bias correction)
pub struct OptimizadorAdamW {
    // First moment (momentum) - matches GradientesGPT2 structure
    pub m_embedding_tokens: Tensor,
    pub m_embedding_posiciones: Tensor,
    pub estados_m_bloques: Vec<EstadoAdamBloque>,
    pub m_ln_final_gamma: Tensor,
    pub m_ln_final_beta: Tensor,
    pub m_peso_salida: Tensor,

    // Second moment (variance) - matches GradientesGPT2 structure
    pub v_embedding_tokens: Tensor,
    pub v_embedding_posiciones: Tensor,
    pub estados_v_bloques: Vec<EstadoAdamBloque>,
    pub v_ln_final_gamma: Tensor,
    pub v_ln_final_beta: Tensor,
    pub v_peso_salida: Tensor,

    // Hyperparameters
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub step: usize,
}

/// Optimizer state for a single transformer block
///
/// Stores moment estimates for all parameters in a transformer block:
/// - Two LayerNorm layers (scale and shift parameters)
/// - Self-attention mechanism (Q, K, V, output projections)
/// - MLP feedforward network (two linear layers)
pub struct EstadoAdamBloque {
    pub ln1_gamma: Tensor,
    pub ln1_beta: Tensor,
    pub atencion: EstadoAdamAtencion,
    pub ln2_gamma: Tensor,
    pub ln2_beta: Tensor,
    pub mlp: EstadoAdamMLP,
}

/// Optimizer state for attention mechanism
///
/// Stores moment estimates for all attention parameters:
/// - Q, K, V projection matrices and biases
/// - Output projection matrix and bias
pub struct EstadoAdamAtencion {
    pub peso_q: Tensor,
    pub sesgo_q: Tensor,
    pub peso_k: Tensor,
    pub sesgo_k: Tensor,
    pub peso_v: Tensor,
    pub sesgo_v: Tensor,
    pub peso_salida: Tensor,
    pub sesgo_salida: Tensor,
}

/// Optimizer state for MLP (feedforward network)
///
/// Stores moment estimates for the two linear layers:
/// - fc1: First projection (n_embd → 4*n_embd)
/// - fc2: Second projection (4*n_embd → n_embd)
pub struct EstadoAdamMLP {
    pub peso_fc1: Tensor,
    pub sesgo_fc1: Tensor,
    pub peso_fc2: Tensor,
    pub sesgo_fc2: Tensor,
}

impl OptimizadorAdamW {
    /// Create a new AdamW optimizer for the given model
    ///
    /// Initializes all moment estimates to zero. The optimizer state mirrors
    /// the model structure exactly, ensuring every parameter has optimizer state.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to optimize
    ///
    /// # Returns
    ///
    /// Optimizer with:
    /// - All moment estimates initialized to zero
    /// - Standard hyperparameters (β₁=0.9, β₂=0.95, ε=1e-8)
    /// - Step counter at 0
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use molineteai::optimizer::OptimizadorAdamW;
    /// # use molineteai::gpt2_entrenable::GPT2Entrenable;
    /// # use molineteai::Config;
    /// let config = Config::tiny(512);
    /// let model = GPT2Entrenable::new(&config);
    /// let optimizer = OptimizadorAdamW::new(&model);
    /// ```
    pub fn new(model: &GPT2Entrenable) -> Self {
        // Initialize all momentum and variance tensors to zero
        let m_embedding_tokens = Tensor::ceros(model.embedding_tokens.forma.clone());
        let m_embedding_posiciones = Tensor::ceros(model.embedding_posiciones.forma.clone());
        let m_ln_final_gamma = Tensor::ceros(model.ln_final.gamma.forma.clone());
        let m_ln_final_beta = Tensor::ceros(model.ln_final.beta.forma.clone());
        let m_peso_salida = Tensor::ceros(model.peso_salida.forma.clone());

        let v_embedding_tokens = Tensor::ceros(model.embedding_tokens.forma.clone());
        let v_embedding_posiciones = Tensor::ceros(model.embedding_posiciones.forma.clone());
        let v_ln_final_gamma = Tensor::ceros(model.ln_final.gamma.forma.clone());
        let v_ln_final_beta = Tensor::ceros(model.ln_final.beta.forma.clone());
        let v_peso_salida = Tensor::ceros(model.peso_salida.forma.clone());

        let mut estados_m_bloques = Vec::new();
        let mut estados_v_bloques = Vec::new();

        for block in &model.bloques {
            let m_block = EstadoAdamBloque {
                ln1_gamma: Tensor::ceros(block.ln1.gamma.forma.clone()),
                ln1_beta: Tensor::ceros(block.ln1.beta.forma.clone()),
                atencion: EstadoAdamAtencion {
                    peso_q: Tensor::ceros(block.atencion.proy_q.peso.forma.clone()),
                    sesgo_q: Tensor::ceros(block.atencion.proy_q.sesgo.forma.clone()),
                    peso_k: Tensor::ceros(block.atencion.proy_k.peso.forma.clone()),
                    sesgo_k: Tensor::ceros(block.atencion.proy_k.sesgo.forma.clone()),
                    peso_v: Tensor::ceros(block.atencion.proy_v.peso.forma.clone()),
                    sesgo_v: Tensor::ceros(block.atencion.proy_v.sesgo.forma.clone()),
                    peso_salida: Tensor::ceros(block.atencion.proy_salida.peso.forma.clone()),
                    sesgo_salida: Tensor::ceros(block.atencion.proy_salida.sesgo.forma.clone()),
                },
                ln2_gamma: Tensor::ceros(block.ln2.gamma.forma.clone()),
                ln2_beta: Tensor::ceros(block.ln2.beta.forma.clone()),
                mlp: EstadoAdamMLP {
                    peso_fc1: Tensor::ceros(block.mlp.fc1.peso.forma.clone()),
                    sesgo_fc1: Tensor::ceros(block.mlp.fc1.sesgo.forma.clone()),
                    peso_fc2: Tensor::ceros(block.mlp.fc2.peso.forma.clone()),
                    sesgo_fc2: Tensor::ceros(block.mlp.fc2.sesgo.forma.clone()),
                },
            };

            let v_block = EstadoAdamBloque {
                ln1_gamma: Tensor::ceros(block.ln1.gamma.forma.clone()),
                ln1_beta: Tensor::ceros(block.ln1.beta.forma.clone()),
                atencion: EstadoAdamAtencion {
                    peso_q: Tensor::ceros(block.atencion.proy_q.peso.forma.clone()),
                    sesgo_q: Tensor::ceros(block.atencion.proy_q.sesgo.forma.clone()),
                    peso_k: Tensor::ceros(block.atencion.proy_k.peso.forma.clone()),
                    sesgo_k: Tensor::ceros(block.atencion.proy_k.sesgo.forma.clone()),
                    peso_v: Tensor::ceros(block.atencion.proy_v.peso.forma.clone()),
                    sesgo_v: Tensor::ceros(block.atencion.proy_v.sesgo.forma.clone()),
                    peso_salida: Tensor::ceros(block.atencion.proy_salida.peso.forma.clone()),
                    sesgo_salida: Tensor::ceros(block.atencion.proy_salida.sesgo.forma.clone()),
                },
                ln2_gamma: Tensor::ceros(block.ln2.gamma.forma.clone()),
                ln2_beta: Tensor::ceros(block.ln2.beta.forma.clone()),
                mlp: EstadoAdamMLP {
                    peso_fc1: Tensor::ceros(block.mlp.fc1.peso.forma.clone()),
                    sesgo_fc1: Tensor::ceros(block.mlp.fc1.sesgo.forma.clone()),
                    peso_fc2: Tensor::ceros(block.mlp.fc2.peso.forma.clone()),
                    sesgo_fc2: Tensor::ceros(block.mlp.fc2.sesgo.forma.clone()),
                },
            };

            estados_m_bloques.push(m_block);
            estados_v_bloques.push(v_block);
        }

        Self {
            m_embedding_tokens,
            m_embedding_posiciones,
            estados_m_bloques,
            m_ln_final_gamma,
            m_ln_final_beta,
            m_peso_salida,
            v_embedding_tokens,
            v_embedding_posiciones,
            estados_v_bloques,
            v_ln_final_gamma,
            v_ln_final_beta,
            v_peso_salida,
            beta1: 0.9,
            beta2: 0.95, // Lower than Adam's 0.999, standard for transformers
            epsilon: 1e-8,
            step: 0,
        }
    }

    /// Create a shallow copy for checkpointing
    ///
    /// Clones all tensors to create an independent copy of the optimizer state.
    /// Used when saving checkpoints to disk.
    ///
    /// # Returns
    ///
    /// New optimizer with cloned state (safe to save/serialize)
    pub fn clonar_superficialmente(&self) -> Self {
        Self {
            m_embedding_tokens: self.m_embedding_tokens.clone(),
            m_embedding_posiciones: self.m_embedding_posiciones.clone(),
            estados_m_bloques: self
                .estados_m_bloques
                .iter()
                .map(|b| EstadoAdamBloque {
                    ln1_gamma: b.ln1_gamma.clone(),
                    ln1_beta: b.ln1_beta.clone(),
                    atencion: EstadoAdamAtencion {
                        peso_q: b.atencion.peso_q.clone(),
                        sesgo_q: b.atencion.sesgo_q.clone(),
                        peso_k: b.atencion.peso_k.clone(),
                        sesgo_k: b.atencion.sesgo_k.clone(),
                        peso_v: b.atencion.peso_v.clone(),
                        sesgo_v: b.atencion.sesgo_v.clone(),
                        peso_salida: b.atencion.peso_salida.clone(),
                        sesgo_salida: b.atencion.sesgo_salida.clone(),
                    },
                    ln2_gamma: b.ln2_gamma.clone(),
                    ln2_beta: b.ln2_beta.clone(),
                    mlp: EstadoAdamMLP {
                        peso_fc1: b.mlp.peso_fc1.clone(),
                        sesgo_fc1: b.mlp.sesgo_fc1.clone(),
                        peso_fc2: b.mlp.peso_fc2.clone(),
                        sesgo_fc2: b.mlp.sesgo_fc2.clone(),
                    },
                })
                .collect(),
            m_ln_final_gamma: self.m_ln_final_gamma.clone(),
            m_ln_final_beta: self.m_ln_final_beta.clone(),
            m_peso_salida: self.m_peso_salida.clone(),
            v_embedding_tokens: self.v_embedding_tokens.clone(),
            v_embedding_posiciones: self.v_embedding_posiciones.clone(),
            estados_v_bloques: self
                .estados_v_bloques
                .iter()
                .map(|b| EstadoAdamBloque {
                    ln1_gamma: b.ln1_gamma.clone(),
                    ln1_beta: b.ln1_beta.clone(),
                    atencion: EstadoAdamAtencion {
                        peso_q: b.atencion.peso_q.clone(),
                        sesgo_q: b.atencion.sesgo_q.clone(),
                        peso_k: b.atencion.peso_k.clone(),
                        sesgo_k: b.atencion.sesgo_k.clone(),
                        peso_v: b.atencion.peso_v.clone(),
                        sesgo_v: b.atencion.sesgo_v.clone(),
                        peso_salida: b.atencion.peso_salida.clone(),
                        sesgo_salida: b.atencion.sesgo_salida.clone(),
                    },
                    ln2_gamma: b.ln2_gamma.clone(),
                    ln2_beta: b.ln2_beta.clone(),
                    mlp: EstadoAdamMLP {
                        peso_fc1: b.mlp.peso_fc1.clone(),
                        sesgo_fc1: b.mlp.sesgo_fc1.clone(),
                        peso_fc2: b.mlp.peso_fc2.clone(),
                        sesgo_fc2: b.mlp.sesgo_fc2.clone(),
                    },
                })
                .collect(),
            v_ln_final_gamma: self.v_ln_final_gamma.clone(),
            v_ln_final_beta: self.v_ln_final_beta.clone(),
            v_peso_salida: self.v_peso_salida.clone(),
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            step: self.step,
        }
    }
}

/// AdamW optimizer parameter update
///
/// Implements the complete AdamW algorithm with decoupled weight decay:
/// 1. Applies weight decay directly to weight matrices (not biases/LayerNorm)
/// 2. Updates moment estimates (m and v)
/// 3. Applies bias correction
/// 4. Updates model parameters using adaptive learning rates
///
/// # Arguments
///
/// * `model` - Model to update (modified in place)
/// * `grads` - Gradients from backpropagation
/// * `optimizer` - Optimizer state (moment estimates updated in place)
/// * `lr` - Learning rate (alpha in the AdamW paper)
/// * `weight_decay` - Weight decay coefficient (lambda, typically 0.1)
///
/// # Algorithm
///
/// For each parameter θ with gradient g:
/// ```text
/// θ = θ * (1 - lr * weight_decay)  # Weight decay (2D tensors only)
/// m = β₁*m + (1-β₁)*g              # Update momentum
/// v = β₂*v + (1-β₂)*g²             # Update variance
/// m_hat = m / (1 - β₁^step)        # Bias correction
/// v_hat = v / (1 - β₂^step)        # Bias correction
/// θ -= lr * m_hat / (√v_hat + ε)   # Update parameter
/// ```
///
/// # Selective Weight Decay
///
/// Weight decay is applied only to 2D tensors (weight matrices), not to:
/// - 1D tensors (biases, LayerNorm parameters)
/// - Embeddings (following common practice)
///
/// # Performance
///
/// Uses parallel computation via Rayon for tensors with >1000 elements.
/// Small tensors use sequential updates to avoid parallelization overhead.
///
/// # Example
///
/// ```rust,no_run
/// # use molineteai::optimizer::{OptimizadorAdamW, actualizar_adamw};
/// # use molineteai::gpt2_entrenable::GPT2Entrenable;
/// # use molineteai::Config;
/// # let config = Config::tiny(512);
/// # let mut model = GPT2Entrenable::new(&config);
/// # let grads = todo!();
/// let mut optimizer = OptimizadorAdamW::new(&model);
///
/// // Training step with weight decay
/// actualizar_adamw(&mut model, &grads, &mut optimizer, 3e-4, 0.1);
/// ```
pub fn actualizar_adamw(
    model: &mut GPT2Entrenable,
    grads: &GradientesGPT2,
    optimizer: &mut OptimizadorAdamW,
    lr: f32,
    weight_decay: f32,
) {
    optimizer.step += 1;
    let step = optimizer.step as f32;

    // Bias correction factors
    // These correct for initialization bias (m and v start at 0)
    let bias_correction1 = 1.0 - optimizer.beta1.powf(step);
    let bias_correction2 = 1.0 - optimizer.beta2.powf(step);

    let beta1 = optimizer.beta1;
    let beta2 = optimizer.beta2;
    let epsilon = optimizer.epsilon;

    // Helper macro to update a parameter with AdamW
    // Parallelizes for large tensors, sequential for small ones
    // apply_decay: whether to apply weight decay (only for 2D weight matrices)
    macro_rules! adamw_update_param {
        ($param:expr, $grad:expr, $m:expr, $v:expr, $apply_decay:expr) => {
            // Parallelize for large tensors (>1000 elements)
            if $param.datos.len() > 1000 {
                $param
                    .datos
                    .par_iter_mut()
                    .zip($grad.datos.par_iter())
                    .zip($m.datos.par_iter_mut().zip($v.datos.par_iter_mut()))
                    .for_each(|((param_val, &grad_val), (m_val, v_val))| {
                        // WEIGHT DECAY: Apply before Adam update (decoupled)
                        if $apply_decay {
                            *param_val *= 1.0 - lr * weight_decay;
                        }

                        // Update biased first moment estimate (momentum)
                        *m_val = beta1 * *m_val + (1.0 - beta1) * grad_val;

                        // Update biased second moment estimate (variance)
                        *v_val = beta2 * *v_val + (1.0 - beta2) * grad_val * grad_val;

                        // Compute bias-corrected first moment
                        let m_hat = *m_val / bias_correction1;

                        // Compute bias-corrected second moment
                        let v_hat = *v_val / bias_correction2;

                        // Update parameter (Adam step)
                        *param_val -= lr * m_hat / (v_hat.sqrt() + epsilon);
                    });
            } else {
                // Sequential for small tensors to avoid parallelization overhead
                for i in 0..$param.datos.len() {
                    // WEIGHT DECAY: Apply before Adam update (decoupled)
                    if $apply_decay {
                        $param.datos[i] *= 1.0 - lr * weight_decay;
                    }

                    let g = $grad.datos[i];

                    // Update biased first moment estimate (momentum)
                    $m.datos[i] = beta1 * $m.datos[i] + (1.0 - beta1) * g;

                    // Update biased second moment estimate (variance)
                    $v.datos[i] = beta2 * $v.datos[i] + (1.0 - beta2) * g * g;

                    // Compute bias-corrected first moment
                    let m_hat = $m.datos[i] / bias_correction1;

                    // Compute bias-corrected second moment
                    let v_hat = $v.datos[i] / bias_correction2;

                    // Update parameter (Adam step)
                    $param.datos[i] -= lr * m_hat / (v_hat.sqrt() + epsilon);
                }
            }
        };
    }

    // Update embeddings (no weight decay - common practice)
    adamw_update_param!(
        model.embedding_tokens,
        grads.embedding_tokens,
        optimizer.m_embedding_tokens,
        optimizer.v_embedding_tokens,
        false // No decay on embeddings
    );
    adamw_update_param!(
        model.embedding_posiciones,
        grads.embedding_posiciones,
        optimizer.m_embedding_posiciones,
        optimizer.v_embedding_posiciones,
        false // No decay on embeddings
    );

    // Update all transformer blocks
    for ((block, block_grads), (m_block, v_block)) in
        model.bloques.iter_mut().zip(&grads.grads_bloques).zip(
            optimizer
                .estados_m_bloques
                .iter_mut()
                .zip(optimizer.estados_v_bloques.iter_mut()),
        )
    {
        // LayerNorm 1 (no decay on 1D scale/shift parameters)
        adamw_update_param!(
            block.ln1.gamma,
            block_grads.ln1_gamma,
            m_block.ln1_gamma,
            v_block.ln1_gamma,
            false // No decay on LayerNorm
        );
        adamw_update_param!(
            block.ln1.beta,
            block_grads.ln1_beta,
            m_block.ln1_beta,
            v_block.ln1_beta,
            false // No decay on LayerNorm
        );

        // Self-attention (decay on 2D weight matrices, not on 1D biases)
        adamw_update_param!(
            block.atencion.proy_q.peso,
            block_grads.atencion.peso_q,
            m_block.atencion.peso_q,
            v_block.atencion.peso_q,
            true // Decay on weight matrix
        );
        adamw_update_param!(
            block.atencion.proy_q.sesgo,
            block_grads.atencion.sesgo_q,
            m_block.atencion.sesgo_q,
            v_block.atencion.sesgo_q,
            false // No decay on bias
        );
        adamw_update_param!(
            block.atencion.proy_k.peso,
            block_grads.atencion.peso_k,
            m_block.atencion.peso_k,
            v_block.atencion.peso_k,
            true // Decay on weight matrix
        );
        adamw_update_param!(
            block.atencion.proy_k.sesgo,
            block_grads.atencion.sesgo_k,
            m_block.atencion.sesgo_k,
            v_block.atencion.sesgo_k,
            false // No decay on bias
        );
        adamw_update_param!(
            block.atencion.proy_v.peso,
            block_grads.atencion.peso_v,
            m_block.atencion.peso_v,
            v_block.atencion.peso_v,
            true // Decay on weight matrix
        );
        adamw_update_param!(
            block.atencion.proy_v.sesgo,
            block_grads.atencion.sesgo_v,
            m_block.atencion.sesgo_v,
            v_block.atencion.sesgo_v,
            false // No decay on bias
        );
        adamw_update_param!(
            block.atencion.proy_salida.peso,
            block_grads.atencion.peso_salida,
            m_block.atencion.peso_salida,
            v_block.atencion.peso_salida,
            true // Decay on weight matrix
        );
        adamw_update_param!(
            block.atencion.proy_salida.sesgo,
            block_grads.atencion.sesgo_salida,
            m_block.atencion.sesgo_salida,
            v_block.atencion.sesgo_salida,
            false // No decay on bias
        );

        // LayerNorm 2 (no decay on 1D scale/shift parameters)
        adamw_update_param!(
            block.ln2.gamma,
            block_grads.ln2_gamma,
            m_block.ln2_gamma,
            v_block.ln2_gamma,
            false // No decay on LayerNorm
        );
        adamw_update_param!(
            block.ln2.beta,
            block_grads.ln2_beta,
            m_block.ln2_beta,
            v_block.ln2_beta,
            false // No decay on LayerNorm
        );

        // MLP (decay on 2D weight matrices, not on 1D biases)
        adamw_update_param!(
            block.mlp.fc1.peso,
            block_grads.mlp.peso_fc1,
            m_block.mlp.peso_fc1,
            v_block.mlp.peso_fc1,
            true // Decay on weight matrix
        );
        adamw_update_param!(
            block.mlp.fc1.sesgo,
            block_grads.mlp.sesgo_fc1,
            m_block.mlp.sesgo_fc1,
            v_block.mlp.sesgo_fc1,
            false // No decay on bias
        );
        adamw_update_param!(
            block.mlp.fc2.peso,
            block_grads.mlp.peso_fc2,
            m_block.mlp.peso_fc2,
            v_block.mlp.peso_fc2,
            true // Decay on weight matrix
        );
        adamw_update_param!(
            block.mlp.fc2.sesgo,
            block_grads.mlp.sesgo_fc2,
            m_block.mlp.sesgo_fc2,
            v_block.mlp.sesgo_fc2,
            false // No decay on bias
        );
    }

    // Final layer norm (no decay on 1D scale/shift parameters)
    adamw_update_param!(
        model.ln_final.gamma,
        grads.ln_final_gamma,
        optimizer.m_ln_final_gamma,
        optimizer.v_ln_final_gamma,
        false // No decay on LayerNorm
    );
    adamw_update_param!(
        model.ln_final.beta,
        grads.ln_final_beta,
        optimizer.m_ln_final_beta,
        optimizer.v_ln_final_beta,
        false // No decay on LayerNorm
    );

    // Output projection weight (decay on 2D weight matrix)
    adamw_update_param!(
        model.peso_salida,
        grads.peso_salida,
        optimizer.m_peso_salida,
        optimizer.v_peso_salida,
        true // Decay on weight matrix
    );
}
