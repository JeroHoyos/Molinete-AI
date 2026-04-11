//! Implementación de GPT-2 Entrenable
//!
//! Este módulo implementa la infraestructura de entrenamiento completa para
//! transformers estilo GPT-2, incluyendo los pasos hacia atrás (backward passes)
//! codificados a mano para todas las capas.
//!
//! ## Visión General
//!
//! Entrenar una red neuronal requiere tres cosas:
//! 1. **Paso hacia adelante (Forward pass)**: Calcular predicciones a partir de las entradas
//! 2. **Paso hacia atrás (Backward pass)**: Calcular gradientes usando la regla de la cadena
//! 3. **Optimización**: Actualizar los parámetros usando los gradientes
//!
//! Este módulo proporciona los tres para una arquitectura GPT-2.
//!
//! ## Componentes de la Arquitectura
//!
//! Cada componente tiene una versión entrenable con métodos "forward" y "backward":
//!
//! - **LinealEntrenable**: Capa totalmente conectada (y = x @ W + b)
//! - **NormCapaEntrenable**: Normalización de capa con escala y desplazamiento aprendibles
//! - **MLPEntrenable**: Red prealimentada (feedforward) de dos capas con activación GELU
//! - **AtencionUnaCabezaEntrenable**: Mecanismo de autoatención (self-attention)
//! - **BloqueTransformerEntrenable**: Bloque transformer completo
//! - **GPT2Entrenable**: Modelo completo que combina todos los componentes
//!
//! ## Retropropagación (Backpropagation)
//!
//! La retropropagación es la regla de la cadena aplicada de forma recursiva:
//!
//! ```text
//! Si y = f(g(x)), entonces dy/dx = (dy/df) * (df/dg) * (dg/dx)
//! ```
//!
//! Para cada capa, implementamos:
//! - `forward()`: Calcula la salida y almacena en caché los valores necesarios para el backward
//! - `backward()`: Calcula los gradientes a partir de los gradientes descendentes y la caché
//!
//! ## Ejemplo: Capa Lineal
//!
//! ```text
//! Forward:  y = x @ W + b
//! Backward:
//!   grad_W = x^T @ grad_y
//!   grad_b = sum(grad_y)
//!   grad_x = grad_y @ W^T
//! ```
//!
//! ## Optimización
//!
//! Implementamos el optimizador Adam, el cual mantiene:
//! - **Primer momento (momentum)**: Media móvil exponencial de los gradientes
//! - **Segundo momento (varianza)**: Media móvil exponencial de los gradientes al cuadrado
//! - **Corrección de sesgo**: Corrige el sesgo de inicialización en los primeros pasos
//!
//! Adam es más estable que SGD y requiere menos ajuste de hiperparámetros.
//!
//! ## Recorte de Gradientes (Gradient Clipping)
//!
//! Para prevenir la explosión de gradientes durante el entrenamiento, recortamos los gradientes a una
//! norma máxima. Esto es esencial para la estabilidad del entrenamiento:
//!
//! ```text
//! si ||grad|| > norma_max:
//!     grad = grad * (norma_max / ||grad||)
//! ```
//!
//! ## Puntos de Control del Modelo (Checkpointing)
//!
//! El módulo incluye funcionalidad completa de guardado/carga (save/load) para:
//! - Parámetros del modelo (pesos y sesgos)
//! - Estado del optimizador (momentum y varianza)
//! - Metadatos de entrenamiento (número de paso, configuración)
//!
//! Esto permite reanudar el entrenamiento desde los puntos de control.
//!
//! ## Optimizaciones de Rendimiento
//!
//! Varias optimizaciones mantienen el entrenamiento rápido en CPU:
//! - **Cálculo de gradientes en paralelo**: Usa Rayon para operaciones paralelas
//! - **Diseño de memoria eficiente**: Minimiza las asignaciones (allocations) durante el paso hacia atrás
//! - **Bloqueo de caché (Cache blocking)**: Reutiliza las optimizaciones de multiplicación de matrices del módulo de tensores
//!
//! ## Enfoque Educativo
//!
//! A diferencia del "autograd" de PyTorch, cada gradiente se calcula explícitamente. Esto hace que
//! el código sea más largo pero mucho más claro para el aprendizaje. Puedes ver exactamente cómo
//! fluyen los gradientes a través de cada capa.

use crate::tensor::Tensor;
use crate::tokenizador::TokenizadorBPE;
use crate::modelo::Config;

// Reexportar tipos de capas para compatibilidad hacia atrás
pub use crate::layers::{
    gelu_backward, gelu_forward, random_init, CacheAtencion, GradientesAtencion, CacheBloque,
    GradientesBloque, CacheNormCapa, CacheLineal, CacheMLP, GradientesMLP, DropoutEntrenable,
    NormCapaEntrenable, LinealEntrenable, MLPEntrenable, AtencionUnaCabezaEntrenable,
    BloqueTransformerEntrenable,
};

// Reexportar utilidades de optimizador y gradientes para compatibilidad hacia atrás
pub use crate::gradientes::{recortar_gradientes, calcular_norma_grad};
pub use crate::optimizador::{
    actualizar_adamw, OptimizadorAdamW, EstadoAdamAtencion, EstadoAdamBloque, EstadoAdamMLP,
};

pub struct GPT2Entrenable {
    pub(crate) embedding_tokens: Tensor,
    pub(crate) embedding_posiciones: Tensor,
    pub(crate) bloques: Vec<BloqueTransformerEntrenable>,
    pub(crate) ln_final: NormCapaEntrenable,
    pub(crate) peso_salida: Tensor,
    pub(crate) config: Config,
}

impl GPT2Entrenable {
    pub fn config(&self) -> &Config {
        &self.config  
    }
    /// Guarda los pesos del modelo en un archivo binario (solo para inferencia, por compatibilidad hacia atrás)
    /// Para guardar con el estado de entrenamiento, usa PuntoControl::guardar() en su lugar
    pub fn guardar_en_archivo(&self, ruta: &str) -> std::io::Result<()> {
        let punto_control = PuntoControl {
            modelo: self.clonar_superficialmente(),
            optimizador: None,
            tokenizador: None,
            paso: 0,
            mejor_perdida_val: f32::INFINITY,
            mejor_paso_val: 0,
        };
        punto_control.guardar(ruta)
    }

    /// Crea una copia superficial para guardar (solo hace referencia a los tensores)
    fn clonar_superficialmente(&self) -> GPT2Entrenable {
        GPT2Entrenable {
            embedding_tokens: self.embedding_tokens.clone(),
            embedding_posiciones: self.embedding_posiciones.clone(),
            bloques: self
                .bloques
                .iter()
                .map(|b| BloqueTransformerEntrenable {
                    ln1: NormCapaEntrenable {
                        gamma: b.ln1.gamma.clone(),
                        beta: b.ln1.beta.clone(),
                        eps: b.ln1.eps,
                    },
                    atencion: AtencionUnaCabezaEntrenable {
                        proy_q: LinealEntrenable {
                            peso: b.atencion.proy_q.peso.clone(),
                            sesgo: b.atencion.proy_q.sesgo.clone(),
                        },
                        proy_k: LinealEntrenable {
                            peso: b.atencion.proy_k.peso.clone(),
                            sesgo: b.atencion.proy_k.sesgo.clone(),
                        },
                        proy_v: LinealEntrenable {
                            peso: b.atencion.proy_v.peso.clone(),
                            sesgo: b.atencion.proy_v.sesgo.clone(),
                        },
                        proy_salida: LinealEntrenable {
                            peso: b.atencion.proy_salida.peso.clone(),
                            sesgo: b.atencion.proy_salida.sesgo.clone(),
                        },
                        dropout_atencion: DropoutEntrenable {
                            tasa: b.atencion.dropout_atencion.tasa,
                            entrenando: b.atencion.dropout_atencion.entrenando,
                        },
                        dropout_resid: DropoutEntrenable {
                            tasa: b.atencion.dropout_resid.tasa,
                            entrenando: b.atencion.dropout_resid.entrenando,
                        },
                        n_embd: b.atencion.n_embd,
                    },
                    ln2: NormCapaEntrenable {
                        gamma: b.ln2.gamma.clone(),
                        beta: b.ln2.beta.clone(),
                        eps: b.ln2.eps,
                    },
                    mlp: MLPEntrenable {
                        fc1: LinealEntrenable {
                            peso: b.mlp.fc1.peso.clone(),
                            sesgo: b.mlp.fc1.sesgo.clone(),
                        },
                        fc2: LinealEntrenable {
                            peso: b.mlp.fc2.peso.clone(),
                            sesgo: b.mlp.fc2.sesgo.clone(),
                        },
                        dropout_resid: DropoutEntrenable {
                            tasa: b.mlp.dropout_resid.tasa,
                            entrenando: b.mlp.dropout_resid.entrenando,
                        },
                    },
                })
                .collect(),
            ln_final: NormCapaEntrenable {
                gamma: self.ln_final.gamma.clone(),
                beta: self.ln_final.beta.clone(),
                eps: self.ln_final.eps,
            },
            peso_salida: self.peso_salida.clone(),
            config: self.config.clone(),
        }
    }

    /// Carga los pesos del modelo desde un archivo de punto de control
    pub fn cargar_desde_archivo(ruta: &str) -> std::io::Result<Self> {
        let punto_control = PuntoControl::cargar(ruta)?;
        Ok(punto_control.modelo)
    }

    pub fn new(config: &Config) -> Self {
        let tamano_vocab = config.vocab_size;
        let n_embd = config.n_embd;
        let tamano_bloque = config.block_size;
        let n_capas = config.n_layers;

        let escala_embedding = (1.0_f32 / (n_embd as f32)).sqrt();
        let embedding_tokens = Tensor::new(
            random_init(tamano_vocab * n_embd, 12345, escala_embedding),
            vec![tamano_vocab, n_embd],
        );
        let embedding_posiciones = Tensor::new(
            random_init(tamano_bloque * n_embd, 23456, escala_embedding),
            vec![tamano_bloque, n_embd],
        );

        let mut bloques = Vec::new();
        for i in 0..n_capas {
            bloques.push(BloqueTransformerEntrenable::new(
                n_embd,
                config.dropout_rate,
                10000 * (i as u64 + 1),
            ));
        }

        let escala_peso = (2.0_f32 / (n_embd as f32)).sqrt();
        let peso_salida = Tensor::new(
            random_init(n_embd * tamano_vocab, 78901, escala_peso),
            vec![n_embd, tamano_vocab],
        );

        Self {
            embedding_tokens,
            embedding_posiciones,
            bloques,
            ln_final: NormCapaEntrenable::new(n_embd),
            peso_salida,
            config: config.clone(),
        }
    }

    /// Cuenta el total de parámetros en el modelo
    ///
    /// Retorna el número total de parámetros entrenables, lo cual determina:
    /// - La capacidad del modelo (habilidad para aprender patrones complejos)
    /// - El uso de memoria (aproximadamente 4 bytes por parámetro para f32)
    /// - El tiempo de entrenamiento (más parámetros = entrenamiento más lento)
    pub fn num_parametros(&self) -> usize {
        let mut total = 0;

        // Embeddings de tokens y posiciones
        total += self.embedding_tokens.datos.len();
        total += self.embedding_posiciones.datos.len();

        // Todos los bloques transformer
        for bloque in &self.bloques {
            // NormCapa 1
            total += bloque.ln1.gamma.datos.len();
            total += bloque.ln1.beta.datos.len();

            // Proyecciones de atención
            total += bloque.atencion.proy_q.peso.datos.len();
            total += bloque.atencion.proy_q.sesgo.datos.len();
            total += bloque.atencion.proy_k.peso.datos.len();
            total += bloque.atencion.proy_k.sesgo.datos.len();
            total += bloque.atencion.proy_v.peso.datos.len();
            total += bloque.atencion.proy_v.sesgo.datos.len();
            total += bloque.atencion.proy_salida.peso.datos.len();
            total += bloque.atencion.proy_salida.sesgo.datos.len();

            // NormCapa 2
            total += bloque.ln2.gamma.datos.len();
            total += bloque.ln2.beta.datos.len();

            // MLP
            total += bloque.mlp.fc1.peso.datos.len();
            total += bloque.mlp.fc1.sesgo.datos.len();
            total += bloque.mlp.fc2.peso.datos.len();
            total += bloque.mlp.fc2.sesgo.datos.len();
        }

        // Normalización de capa final
        total += self.ln_final.gamma.datos.len();
        total += self.ln_final.beta.datos.len();

        // Proyección de salida
        total += self.peso_salida.datos.len();

        total
    }

    pub fn forward(&self, ids_entrada: &[usize]) -> (Tensor, CacheGPT2) {
        let long_sec = ids_entrada.len();
        let n_embd = self.config.n_embd;
        let tamano_vocab = self.config.vocab_size;

        // Embeber tokens y posiciones
        let mut embebidos = Vec::new();
        for (pos, &id_token) in ids_entrada.iter().enumerate() {
            let id_token = id_token.min(tamano_vocab - 1);
            let pos = pos.min(self.config.block_size - 1);

            let inicio_token = id_token * n_embd;
            let inicio_pos = pos * n_embd;

            for i in 0..n_embd {
                embebidos.push(
                    self.embedding_tokens.datos[inicio_token + i]
                        + self.embedding_posiciones.datos[inicio_pos + i],
                );
            }
        }

        let mut x = Tensor::new(embebidos, vec![long_sec, n_embd]);

        // Paso hacia adelante a través de todos los bloques transformer
        let mut caches_bloque = Vec::new();
        for bloque in &self.bloques {
            let (x_siguiente, cache) = bloque.forward(&x);
            caches_bloque.push(cache);
            x = x_siguiente;
        }

        // Normalización de capa final
        let (x_normalizado, cache_ln_final) = self.ln_final.forward(&x);

        // Proyectar al vocabulario
        let logits = x_normalizado.matmul(&self.peso_salida);

        let cache = CacheGPT2 {
            ids_entrada: ids_entrada.to_vec(),
            caches_bloque,
            cache_ln_final,
            x_antes_ln_final: x,
        };

        (logits, cache)
    }

    pub fn calcular_perdida(&self, logits: &Tensor, objetivos: &[usize]) -> f32 {
        let long_sec = objetivos.len();
        let tamano_vocab = self.config.vocab_size;
        let mut perdida_total = 0.0;

        for (i, &objetivo) in objetivos.iter().enumerate() {
            let inicio_logit = i * tamano_vocab;
            let slice_logits = &logits.datos[inicio_logit..inicio_logit + tamano_vocab];

            // Truco de estabilidad numérica: restar el máximo para evitar desbordamiento (overflow)
            let logit_maximo = slice_logits
                .iter()
                .fold(f32::NEG_INFINITY, |a: f32, &b| a.max(b));
            let suma_exp: f32 = slice_logits.iter().map(|&x| (x - logit_maximo).exp()).sum();

            let objetivo = objetivo.min(tamano_vocab - 1);
            let logit_objetivo = slice_logits[objetivo];
            let log_prob = (logit_objetivo - logit_maximo) - suma_exp.ln();
            perdida_total -= log_prob;
        }

        perdida_total / long_sec as f32
    }

    pub fn backward(&self, logits: &Tensor, objetivos: &[usize], cache: &CacheGPT2) -> GradientesGPT2 {
        let long_sec = objetivos.len();
        let tamano_vocab = self.config.vocab_size;
        let n_embd = self.config.n_embd;

        // 1. Gradiente de la pérdida con respecto a los logits
        let mut grad_logits = Vec::new();
        for (i, &id_objetivo) in objetivos.iter().enumerate().take(long_sec) {
            let inicio_logit = i * tamano_vocab;
            let slice_logits = &logits.datos[inicio_logit..inicio_logit + tamano_vocab];

            let logit_maximo = slice_logits
                .iter()
                .fold(f32::NEG_INFINITY, |a: f32, &b| a.max(b));
            let valores_exp: Vec<f32> = slice_logits
                .iter()
                .map(|&x| (x - logit_maximo).exp())
                .collect();
            let suma: f32 = valores_exp.iter().sum();
            let probs: Vec<f32> = valores_exp.iter().map(|&x| x / suma).collect();

            for (j, &prob) in probs.iter().enumerate() {
                let objetivo = id_objetivo.min(tamano_vocab - 1);
                let grad = if j == objetivo { prob - 1.0 } else { prob };
                grad_logits.push(grad / long_sec as f32);
            }
        }
        let grad_logits = Tensor::new(grad_logits, vec![long_sec, tamano_vocab]);

        // 2. Retropropagación a través de la proyección de salida
        let grad_peso_salida = cache
            .x_antes_ln_final
            .transpose(-2, -1)
            .matmul(&grad_logits);
        let mut grad_x = grad_logits.matmul(&self.peso_salida.transpose(-2, -1));

        // 3. Retropropagación a través de la normalización de capa final
        let grads_ln_final = self.ln_final.backward(&grad_x, &cache.cache_ln_final);
        grad_x = grads_ln_final.x;

        // 4. Retropropagación a través de los bloques transformer (en orden inverso)
        let mut grads_bloques = Vec::new();
        for (bloque, cache_bloque) in self.bloques.iter().zip(&cache.caches_bloque).rev() {
            let grads = bloque.backward(&grad_x, cache_bloque);
            grad_x = grads.x.clone();
            grads_bloques.push(grads);
        }
        grads_bloques.reverse(); // Poner de nuevo en orden hacia adelante

        // 5. Retropropagación a los embeddings
        let mut grad_embedding_tokens = vec![0.0; tamano_vocab * n_embd];
        let mut grad_embedding_posiciones = vec![0.0; self.config.block_size * n_embd];

        for (pos, &id_token) in cache.ids_entrada.iter().enumerate() {
            let id_token = id_token.min(tamano_vocab - 1);
            let pos = pos.min(self.config.block_size - 1);

            for i in 0..n_embd {
                let valor_grad = grad_x.datos[pos * n_embd + i];
                grad_embedding_tokens[id_token * n_embd + i] += valor_grad;
                grad_embedding_posiciones[pos * n_embd + i] += valor_grad;
            }
        }

        GradientesGPT2 {
            embedding_tokens: Tensor::new(grad_embedding_tokens, vec![tamano_vocab, n_embd]),
            embedding_posiciones: Tensor::new(
                grad_embedding_posiciones,
                vec![self.config.block_size, n_embd],
            ),
            grads_bloques,
            ln_final_gamma: grads_ln_final.gamma,
            ln_final_beta: grads_ln_final.beta,
            peso_salida: grad_peso_salida,
        }
    }

    /// Genera texto
    pub fn generar(&self, prompt: &[usize], max_tokens: usize, temperatura: f32) -> Vec<usize> {
        let mut tokens = prompt.to_vec();

        for _ in 0..max_tokens {
            let (logits, _) = self.forward(&tokens);

            // Obtener logits de la última posición
            let long_sec = tokens.len();
            let tamano_vocab = self.config.vocab_size;
            let inicio_ultima_pos = (long_sec - 1) * tamano_vocab;
            let ultimos_logits = &logits.datos[inicio_ultima_pos..inicio_ultima_pos + tamano_vocab];

            // Muestrear con temperatura
            let logits_escalados: Vec<f32> = ultimos_logits.iter().map(|&x| x / temperatura).collect();
            let logit_maximo = logits_escalados
                .iter()
                .fold(f32::NEG_INFINITY, |a: f32, &b| a.max(b));
            let valores_exp: Vec<f32> = logits_escalados
                .iter()
                .map(|&x| (x - logit_maximo).exp())
                .collect();
            let suma: f32 = valores_exp.iter().sum();
            let probs: Vec<f32> = valores_exp.iter().map(|&x| x / suma).collect();

            // Nota: asumiendo que sample_from_probs se traduce a muestrear_de_probs
            let siguiente_token = muestrear_de_probs(&probs);
            tokens.push(siguiente_token);

            if tokens.len() >= self.config.block_size {
                break;
            }
        }

        tokens
    }
}

pub struct CacheGPT2 {
    pub(crate) ids_entrada: Vec<usize>,
    pub(crate) caches_bloque: Vec<CacheBloque>,
    pub(crate) cache_ln_final: CacheNormCapa,
    pub(crate) x_antes_ln_final: Tensor,
}

pub struct GradientesGPT2 {
    pub embedding_tokens: Tensor,
    pub embedding_posiciones: Tensor,
    pub grads_bloques: Vec<GradientesBloque>,
    pub ln_final_gamma: Tensor,
    pub ln_final_beta: Tensor,
    pub peso_salida: Tensor,
}

//=============================================================================
// PUNTO DE CONTROL - Para guardar/cargar el estado de entrenamiento
//=============================================================================

#[derive(serde::Serialize, serde::Deserialize)]
pub struct MetadatosPuntoControl {
    pub paso: usize,
    pub mejor_perdida_val: f32,
    pub mejor_paso_val: usize,
}

pub struct PuntoControl {
    pub modelo: GPT2Entrenable,
    pub optimizador: Option<OptimizadorAdamW>,
    pub tokenizador: Option<TokenizadorBPE>,
    pub paso: usize,
    pub mejor_perdida_val: f32,
    pub mejor_paso_val: usize,
}

impl PuntoControl {
    /// Crea un punto de control solo para inferencia (sin estado del optimizador)
    pub fn solo_inferencia(modelo: GPT2Entrenable) -> Self {
        Self {
            modelo,
            optimizador: None,
            tokenizador: None,
            paso: 0,
            mejor_perdida_val: f32::INFINITY,
            mejor_paso_val: 0,
        }
    }

    /// Guarda el punto de control en un archivo
    pub fn guardar(&self, ruta: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;

        println!("💾 Guardando punto de control en {}...", ruta);

        let mut archivo = File::create(ruta)?;

        // Escribir encabezado y versión (mantenemos las firmas originales)
        archivo.write_all(b"FESTE_CKPT")?;
        archivo.write_all(&[1u8])?; // Versión 1

        // Escribir configuración del modelo
        let config_json = serde_json::to_string(&self.modelo.config)?;
        let bytes_config = config_json.as_bytes();
        archivo.write_all(&(bytes_config.len() as u32).to_le_bytes())?;
        archivo.write_all(bytes_config)?;

        // Función auxiliar para escribir un tensor
        let escribir_tensor = |archivo: &mut File, tensor: &Tensor| -> std::io::Result<()> {
            archivo.write_all(&(tensor.forma.len() as u32).to_le_bytes())?;
            for &dim in &tensor.forma {
                archivo.write_all(&(dim as u32).to_le_bytes())?;
            }
            archivo.write_all(&(tensor.datos.len() as u32).to_le_bytes())?;
            for &val in tensor.datos.iter() { let val: f32 = val;
                archivo.write_all(&val.to_le_bytes())?;
            }
            Ok(())
        };

        // Escribir pesos del modelo
        escribir_tensor(&mut archivo, &self.modelo.embedding_tokens)?;
        escribir_tensor(&mut archivo, &self.modelo.embedding_posiciones)?;

        archivo.write_all(&(self.modelo.bloques.len() as u32).to_le_bytes())?;
        for bloque in &self.modelo.bloques {
            escribir_tensor(&mut archivo, &bloque.ln1.gamma)?;
            escribir_tensor(&mut archivo, &bloque.ln1.beta)?;
            escribir_tensor(&mut archivo, &bloque.atencion.proy_q.peso)?;
            escribir_tensor(&mut archivo, &bloque.atencion.proy_q.sesgo)?;
            escribir_tensor(&mut archivo, &bloque.atencion.proy_k.peso)?;
            escribir_tensor(&mut archivo, &bloque.atencion.proy_k.sesgo)?;
            escribir_tensor(&mut archivo, &bloque.atencion.proy_v.peso)?;
            escribir_tensor(&mut archivo, &bloque.atencion.proy_v.sesgo)?;
            escribir_tensor(&mut archivo, &bloque.atencion.proy_salida.peso)?;
            escribir_tensor(&mut archivo, &bloque.atencion.proy_salida.sesgo)?;
            escribir_tensor(&mut archivo, &bloque.ln2.gamma)?;
            escribir_tensor(&mut archivo, &bloque.ln2.beta)?;
            escribir_tensor(&mut archivo, &bloque.mlp.fc1.peso)?;
            escribir_tensor(&mut archivo, &bloque.mlp.fc1.sesgo)?;
            escribir_tensor(&mut archivo, &bloque.mlp.fc2.peso)?;
            escribir_tensor(&mut archivo, &bloque.mlp.fc2.sesgo)?;
        }

        escribir_tensor(&mut archivo, &self.modelo.ln_final.gamma)?;
        escribir_tensor(&mut archivo, &self.modelo.ln_final.beta)?;
        escribir_tensor(&mut archivo, &self.modelo.peso_salida)?;

        // Escribir bandera de estado del optimizador
        let tiene_optimizador = self.optimizador.is_some();
        archivo.write_all(&[tiene_optimizador as u8])?;

        if let Some(opt) = &self.optimizador {
            // Escribir metadatos del optimizador
            archivo.write_all(&opt.step.to_le_bytes())?;
            archivo.write_all(&opt.beta1.to_le_bytes())?;
            archivo.write_all(&opt.beta2.to_le_bytes())?;
            archivo.write_all(&opt.epsilon.to_le_bytes())?;

            // Escribir tensores de momento (m) y varianza (v) del optimizador
            escribir_tensor(&mut archivo, &opt.m_embedding_tokens)?;
            escribir_tensor(&mut archivo, &opt.m_embedding_posiciones)?;
            escribir_tensor(&mut archivo, &opt.v_embedding_tokens)?;
            escribir_tensor(&mut archivo, &opt.v_embedding_posiciones)?;

            for (bloque_m, bloque_v) in opt.estados_m_bloques.iter().zip(&opt.estados_v_bloques) {
                escribir_tensor(&mut archivo, &bloque_m.ln1_gamma)?;
                escribir_tensor(&mut archivo, &bloque_m.ln1_beta)?;
                escribir_tensor(&mut archivo, &bloque_v.ln1_gamma)?;
                escribir_tensor(&mut archivo, &bloque_v.ln1_beta)?;

                escribir_tensor(&mut archivo, &bloque_m.atencion.peso_q)?;
                escribir_tensor(&mut archivo, &bloque_m.atencion.sesgo_q)?;
                escribir_tensor(&mut archivo, &bloque_m.atencion.peso_k)?;
                escribir_tensor(&mut archivo, &bloque_m.atencion.sesgo_k)?;
                escribir_tensor(&mut archivo, &bloque_m.atencion.peso_v)?;
                escribir_tensor(&mut archivo, &bloque_m.atencion.sesgo_v)?;
                escribir_tensor(&mut archivo, &bloque_m.atencion.peso_salida)?;
                escribir_tensor(&mut archivo, &bloque_m.atencion.sesgo_salida)?;

                escribir_tensor(&mut archivo, &bloque_v.atencion.peso_q)?;
                escribir_tensor(&mut archivo, &bloque_v.atencion.sesgo_q)?;
                escribir_tensor(&mut archivo, &bloque_v.atencion.peso_k)?;
                escribir_tensor(&mut archivo, &bloque_v.atencion.sesgo_k)?;
                escribir_tensor(&mut archivo, &bloque_v.atencion.peso_v)?;
                escribir_tensor(&mut archivo, &bloque_v.atencion.sesgo_v)?;
                escribir_tensor(&mut archivo, &bloque_v.atencion.peso_salida)?;
                escribir_tensor(&mut archivo, &bloque_v.atencion.sesgo_salida)?;

                escribir_tensor(&mut archivo, &bloque_m.ln2_gamma)?;
                escribir_tensor(&mut archivo, &bloque_m.ln2_beta)?;
                escribir_tensor(&mut archivo, &bloque_v.ln2_gamma)?;
                escribir_tensor(&mut archivo, &bloque_v.ln2_beta)?;

                escribir_tensor(&mut archivo, &bloque_m.mlp.peso_fc1)?;
                escribir_tensor(&mut archivo, &bloque_m.mlp.sesgo_fc1)?;
                escribir_tensor(&mut archivo, &bloque_m.mlp.peso_fc2)?;
                escribir_tensor(&mut archivo, &bloque_m.mlp.sesgo_fc2)?;

                escribir_tensor(&mut archivo, &bloque_v.mlp.peso_fc1)?;
                escribir_tensor(&mut archivo, &bloque_v.mlp.sesgo_fc1)?;
                escribir_tensor(&mut archivo, &bloque_v.mlp.peso_fc2)?;
                escribir_tensor(&mut archivo, &bloque_v.mlp.sesgo_fc2)?;
            }

            escribir_tensor(&mut archivo, &opt.m_ln_final_gamma)?;
            escribir_tensor(&mut archivo, &opt.m_ln_final_beta)?;
            escribir_tensor(&mut archivo, &opt.m_peso_salida)?;
            escribir_tensor(&mut archivo, &opt.v_ln_final_gamma)?;
            escribir_tensor(&mut archivo, &opt.v_ln_final_beta)?;
            escribir_tensor(&mut archivo, &opt.v_peso_salida)?;
        }

        // Escribir bandera de estado del tokenizador
        let tiene_tokenizador = self.tokenizador.is_some();
        archivo.write_all(&[tiene_tokenizador as u8])?;

        if let Some(tokenizador) = &self.tokenizador {
            // Serializar tokenizador a JSON
            let json_tokenizador = serde_json::to_string(tokenizador)?;
            let bytes_tokenizador = json_tokenizador.as_bytes();
            archivo.write_all(&(bytes_tokenizador.len() as u32).to_le_bytes())?;
            archivo.write_all(bytes_tokenizador)?;
        }

        // Escribir metadatos del punto de control
        let metadatos = MetadatosPuntoControl {
            paso: self.paso,
            mejor_perdida_val: self.mejor_perdida_val,
            mejor_paso_val: self.mejor_paso_val,
        };
        let json_metadatos = serde_json::to_string(&metadatos)?;
        let bytes_metadatos = json_metadatos.as_bytes();
        archivo.write_all(&(bytes_metadatos.len() as u32).to_le_bytes())?;
        archivo.write_all(bytes_metadatos)?;

        let tamano_archivo = archivo.metadata()?.len() as f64 / 1_000_000.0;
        println!("✅ ¡Punto de control guardado exitosamente!");
        println!("   Tamaño del archivo: {:.2} MB", tamano_archivo);
        let mut incluye = vec!["Pesos del modelo"];
        if self.optimizador.is_some() {
            incluye.push("Estado del optimizador");
        }
        if self.tokenizador.is_some() {
            incluye.push("Tokenizador");
        }
        incluye.push("Metadatos de entrenamiento");
        println!("   Incluye: {}", incluye.join(" + "));

        Ok(())
    }

    /// Carga un punto de control desde un archivo
    pub fn cargar(ruta: &str) -> std::io::Result<Self> {
        use std::fs::File;
        use std::io::Read;

        println!("📂 Cargando punto de control desde {}...", ruta);

        let mut archivo = File::open(ruta)?;

        // Leer y verificar el encabezado
        let mut encabezado = [0u8; 10];
        archivo.read_exact(&mut encabezado)?;
        if &encabezado != b"FESTE_CKPT" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Encabezado de punto de control inválido - se esperaba FESTE_CKPT",
            ));
        }

        // Leer la versión
        let mut version = [0u8; 1];
        archivo.read_exact(&mut version)?;
        if version[0] != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Versión de punto de control no soportada: {}", version[0]),
            ));
        }

        // Leer la configuración
        let mut bytes_long_config = [0u8; 4];
        archivo.read_exact(&mut bytes_long_config)?;
        let long_config = u32::from_le_bytes(bytes_long_config) as usize;

        let mut bytes_config = vec![0u8; long_config];
        archivo.read_exact(&mut bytes_config)?;
        let json_config = String::from_utf8(bytes_config)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let config: Config = serde_json::from_str(&json_config)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Función auxiliar para leer un tensor
        let leer_tensor = |archivo: &mut File| -> std::io::Result<Tensor> {
            let mut bytes_long_forma = [0u8; 4];
            archivo.read_exact(&mut bytes_long_forma)?;
            let long_forma = u32::from_le_bytes(bytes_long_forma) as usize;

            let mut forma = Vec::with_capacity(long_forma);
            for _ in 0..long_forma {
                let mut bytes_dim = [0u8; 4];
                archivo.read_exact(&mut bytes_dim)?;
                forma.push(u32::from_le_bytes(bytes_dim) as usize);
            }

            let mut bytes_long_datos = [0u8; 4];
            archivo.read_exact(&mut bytes_long_datos)?;
            let long_datos = u32::from_le_bytes(bytes_long_datos) as usize;

            let mut datos = Vec::with_capacity(long_datos);
            for _ in 0..long_datos {
                let mut bytes_val = [0u8; 4];
                archivo.read_exact(&mut bytes_val)?;
                datos.push(f32::from_le_bytes(bytes_val));
            }

            Ok(Tensor::new(datos, forma))
        };

        // Leer pesos del modelo
        let embedding_tokens = leer_tensor(&mut archivo)?;
        let embedding_posiciones = leer_tensor(&mut archivo)?;

        let mut bytes_num_bloques = [0u8; 4];
        archivo.read_exact(&mut bytes_num_bloques)?;
        let num_bloques = u32::from_le_bytes(bytes_num_bloques) as usize;

        let mut bloques = Vec::with_capacity(num_bloques);
        for _ in 0..num_bloques {
            let ln1_gamma = leer_tensor(&mut archivo)?;
            let ln1_beta = leer_tensor(&mut archivo)?;
            let ln1 = NormCapaEntrenable {
                gamma: ln1_gamma,
                beta: ln1_beta,
                eps: 1e-5,
            };

            let peso_q = leer_tensor(&mut archivo)?;
            let sesgo_q = leer_tensor(&mut archivo)?;
            let peso_k = leer_tensor(&mut archivo)?;
            let sesgo_k = leer_tensor(&mut archivo)?;
            let peso_v = leer_tensor(&mut archivo)?;
            let sesgo_v = leer_tensor(&mut archivo)?;
            let peso_salida_atn = leer_tensor(&mut archivo)?;
            let sesgo_salida_atn = leer_tensor(&mut archivo)?;

            let atencion = AtencionUnaCabezaEntrenable {
                proy_q: LinealEntrenable {
                    peso: peso_q,
                    sesgo: sesgo_q,
                },
                proy_k: LinealEntrenable {
                    peso: peso_k,
                    sesgo: sesgo_k,
                },
                proy_v: LinealEntrenable {
                    peso: peso_v,
                    sesgo: sesgo_v,
                },
                proy_salida: LinealEntrenable {
                    peso: peso_salida_atn,
                    sesgo: sesgo_salida_atn,
                },
                dropout_atencion: DropoutEntrenable::new(config.dropout_rate),
                dropout_resid: DropoutEntrenable::new(config.dropout_rate),
                n_embd: config.n_embd,
            };

            let ln2_gamma = leer_tensor(&mut archivo)?;
            let ln2_beta = leer_tensor(&mut archivo)?;
            let ln2 = NormCapaEntrenable {
                gamma: ln2_gamma,
                beta: ln2_beta,
                eps: 1e-5,
            };

            let peso_fc1 = leer_tensor(&mut archivo)?;
            let sesgo_fc1 = leer_tensor(&mut archivo)?;
            let peso_fc2 = leer_tensor(&mut archivo)?;
            let sesgo_fc2 = leer_tensor(&mut archivo)?;

            let mlp = MLPEntrenable {
                fc1: LinealEntrenable {
                    peso: peso_fc1,
                    sesgo: sesgo_fc1,
                },
                fc2: LinealEntrenable {
                    peso: peso_fc2,
                    sesgo: sesgo_fc2,
                },
                dropout_resid: DropoutEntrenable::new(config.dropout_rate),
            };

            bloques.push(BloqueTransformerEntrenable {
                ln1,
                atencion,
                ln2,
                mlp,
            });
        }

        let ln_final_gamma = leer_tensor(&mut archivo)?;
        let ln_final_beta = leer_tensor(&mut archivo)?;
        let ln_final = NormCapaEntrenable {
            gamma: ln_final_gamma,
            beta: ln_final_beta,
            eps: 1e-5,
        };

        let peso_salida = leer_tensor(&mut archivo)?;

        let modelo = GPT2Entrenable {
            embedding_tokens,
            embedding_posiciones,
            bloques,
            ln_final,
            peso_salida,
            config,
        };

        // Leer bandera de estado del optimizador
        let mut tiene_optimizador = [0u8; 1];
        archivo.read_exact(&mut tiene_optimizador)?;

        let optimizador = if tiene_optimizador[0] == 1 {
            // Leer metadatos del optimizador
            let mut bytes_paso = [0u8; 8];
            let mut bytes_beta1 = [0u8; 4];
            let mut bytes_beta2 = [0u8; 4];
            let mut bytes_epsilon = [0u8; 4];

            archivo.read_exact(&mut bytes_paso)?;
            archivo.read_exact(&mut bytes_beta1)?;
            archivo.read_exact(&mut bytes_beta2)?;
            archivo.read_exact(&mut bytes_epsilon)?;

            let paso = usize::from_le_bytes(bytes_paso);
            let beta1 = f32::from_le_bytes(bytes_beta1);
            let beta2 = f32::from_le_bytes(bytes_beta2);
            let epsilon = f32::from_le_bytes(bytes_epsilon);

            // Leer tensores del optimizador
            let m_embedding_tokens = leer_tensor(&mut archivo)?;
            let m_embedding_posiciones = leer_tensor(&mut archivo)?;
            let v_embedding_tokens = leer_tensor(&mut archivo)?;
            let v_embedding_posiciones = leer_tensor(&mut archivo)?;

            let mut estados_m_bloques = Vec::new();
            let mut estados_v_bloques = Vec::new();

            for _ in 0..modelo.bloques.len() {
                let m_ln1_gamma = leer_tensor(&mut archivo)?;
                let m_ln1_beta = leer_tensor(&mut archivo)?;
                let v_ln1_gamma = leer_tensor(&mut archivo)?;
                let v_ln1_beta = leer_tensor(&mut archivo)?;

                let m_peso_q = leer_tensor(&mut archivo)?;
                let m_sesgo_q = leer_tensor(&mut archivo)?;
                let m_peso_k = leer_tensor(&mut archivo)?;
                let m_sesgo_k = leer_tensor(&mut archivo)?;
                let m_peso_v = leer_tensor(&mut archivo)?;
                let m_sesgo_v = leer_tensor(&mut archivo)?;
                let m_peso_salida_atn = leer_tensor(&mut archivo)?;
                let m_sesgo_salida_atn = leer_tensor(&mut archivo)?;

                let v_peso_q = leer_tensor(&mut archivo)?;
                let v_sesgo_q = leer_tensor(&mut archivo)?;
                let v_peso_k = leer_tensor(&mut archivo)?;
                let v_sesgo_k = leer_tensor(&mut archivo)?;
                let v_peso_v = leer_tensor(&mut archivo)?;
                let v_sesgo_v = leer_tensor(&mut archivo)?;
                let v_peso_salida_atn = leer_tensor(&mut archivo)?;
                let v_sesgo_salida_atn = leer_tensor(&mut archivo)?;

                let m_ln2_gamma = leer_tensor(&mut archivo)?;
                let m_ln2_beta = leer_tensor(&mut archivo)?;
                let v_ln2_gamma = leer_tensor(&mut archivo)?;
                let v_ln2_beta = leer_tensor(&mut archivo)?;

                let m_peso_fc1 = leer_tensor(&mut archivo)?;
                let m_sesgo_fc1 = leer_tensor(&mut archivo)?;
                let m_peso_fc2 = leer_tensor(&mut archivo)?;
                let m_sesgo_fc2 = leer_tensor(&mut archivo)?;

                let v_peso_fc1 = leer_tensor(&mut archivo)?;
                let v_sesgo_fc1 = leer_tensor(&mut archivo)?;
                let v_peso_fc2 = leer_tensor(&mut archivo)?;
                let v_sesgo_fc2 = leer_tensor(&mut archivo)?;

                estados_m_bloques.push(EstadoAdamBloque {
                    ln1_gamma: m_ln1_gamma,
                    ln1_beta: m_ln1_beta,
                    atencion: EstadoAdamAtencion {
                        peso_q: m_peso_q,
                        sesgo_q: m_sesgo_q,
                        peso_k: m_peso_k,
                        sesgo_k: m_sesgo_k,
                        peso_v: m_peso_v,
                        sesgo_v: m_sesgo_v,
                        peso_salida: m_peso_salida_atn,
                        sesgo_salida: m_sesgo_salida_atn,
                    },
                    ln2_gamma: m_ln2_gamma,
                    ln2_beta: m_ln2_beta,
                    mlp: EstadoAdamMLP {
                        peso_fc1: m_peso_fc1,
                        sesgo_fc1: m_sesgo_fc1,
                        peso_fc2: m_peso_fc2,
                        sesgo_fc2: m_sesgo_fc2,
                    },
                });

                estados_v_bloques.push(EstadoAdamBloque {
                    ln1_gamma: v_ln1_gamma,
                    ln1_beta: v_ln1_beta,
                    atencion: EstadoAdamAtencion {
                        peso_q: v_peso_q,
                        sesgo_q: v_sesgo_q,
                        peso_k: v_peso_k,
                        sesgo_k: v_sesgo_k,
                        peso_v: v_peso_v,
                        sesgo_v: v_sesgo_v,
                        peso_salida: v_peso_salida_atn,
                        sesgo_salida: v_sesgo_salida_atn,
                    },
                    ln2_gamma: v_ln2_gamma,
                    ln2_beta: v_ln2_beta,
                    mlp: EstadoAdamMLP {
                        peso_fc1: v_peso_fc1,
                        sesgo_fc1: v_sesgo_fc1,
                        peso_fc2: v_peso_fc2,
                        sesgo_fc2: v_sesgo_fc2,
                    },
                });
            }
            
            let m_ln_final_gamma = leer_tensor(&mut archivo)?;
            let m_ln_final_beta = leer_tensor(&mut archivo)?;
            let m_peso_salida = leer_tensor(&mut archivo)?;
            let v_ln_final_gamma = leer_tensor(&mut archivo)?;
            let v_ln_final_beta = leer_tensor(&mut archivo)?;
            let v_peso_salida = leer_tensor(&mut archivo)?;

            Some(OptimizadorAdamW {
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
                beta1,
                beta2,
                epsilon,
                step: paso,
            })
        } else {
            None
        };

        // Leer bandera de estado del tokenizador
        let mut tiene_tokenizador = [0u8; 1];
        archivo.read_exact(&mut tiene_tokenizador)?;

        let tokenizador = if tiene_tokenizador[0] == 1 {
            // Leer JSON del tokenizador
            let mut bytes_long_tokenizador = [0u8; 4];
            archivo.read_exact(&mut bytes_long_tokenizador)?;
            let long_tokenizador = u32::from_le_bytes(bytes_long_tokenizador) as usize;

            let mut bytes_tokenizador = vec![0u8; long_tokenizador];
            archivo.read_exact(&mut bytes_tokenizador)?;
            let json_tokenizador = String::from_utf8(bytes_tokenizador)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            let tokenizador: TokenizadorBPE = serde_json::from_str(&json_tokenizador)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            Some(tokenizador)
        } else {
            None
        };

        // Leer metadatos del punto de control
        let mut bytes_long_metadatos = [0u8; 4];
        archivo.read_exact(&mut bytes_long_metadatos)?;
        let long_metadatos = u32::from_le_bytes(bytes_long_metadatos) as usize;

        let mut bytes_metadatos = vec![0u8; long_metadatos];
        archivo.read_exact(&mut bytes_metadatos)?;
        let json_metadatos = String::from_utf8(bytes_metadatos)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let metadatos: MetadatosPuntoControl = serde_json::from_str(&json_metadatos)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        println!("✅ ¡Punto de control cargado exitosamente!");
        if optimizador.is_some() {
            println!(
                "   Estado de entrenamiento: paso {}, mejor pérdida val: {:.4}",
                metadatos.paso, metadatos.mejor_perdida_val
            );
        } else {
            println!("   Modelo solo para inferencia (sin estado del optimizador)");
        }
        if tokenizador.is_some() {
            println!(
                "   Tokenizador incluido (tamaño del vocabulario: {})",
                tokenizador.as_ref().unwrap().tam_vocabulario()
            );
        }

        Ok(Self {
            modelo,
            optimizador,
            tokenizador,
            paso: metadatos.paso,
            mejor_perdida_val: metadatos.mejor_perdida_val,
            mejor_paso_val: metadatos.mejor_paso_val,
        })
    }

    /// Guarda el punto de control en un hilo en segundo plano (no bloqueante)
    /// Devuelve un JoinHandle que se puede usar para esperar a que termine
    pub fn guardar_en_segundo_plano(self, ruta: String) -> std::thread::JoinHandle<std::io::Result<()>> {
        std::thread::spawn(move || self.guardar(&ruta))
    }
}
/// Muestrea un índice a partir de un arreglo de probabilidades
fn muestrear_de_probs(probs: &[f32]) -> usize {
    use std::cell::Cell;
    thread_local! {
        static RNG: Cell<u64> = const { Cell::new(12345) };
    }

    RNG.with(|rng| {
        let mut estado = rng.get();
        estado = estado.wrapping_mul(1103515245).wrapping_add(12345);
        rng.set(estado);

        let val_aleatorio = ((estado / 65536) % 32768) as f32 / 32768.0;

        let mut suma_acumulada = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            suma_acumulada += p;
            if val_aleatorio < suma_acumulada {
                return i;
            }
        }
        probs.len() - 1
    })
}

//=============================================================================
// AYUDANTES DE ENTRENAMIENTO
//=============================================================================

/// Inicializa gradientes en cero coincidiendo con la estructura del modelo
fn inicializar_gradientes_cero(modelo: &GPT2Entrenable) -> GradientesGPT2 {
    let tamano_vocabulario = modelo.config.vocab_size;
    let n_embd = modelo.config.n_embd;
    let tamano_bloque = modelo.config.block_size;

    let mut grads_bloques = Vec::new();
    for bloque in &modelo.bloques {
        grads_bloques.push(GradientesBloque {
            x: Tensor::new(vec![0.0; n_embd], vec![1, n_embd]),
            ln1_gamma: Tensor::new(vec![0.0; n_embd], vec![n_embd]),
            ln1_beta: Tensor::new(vec![0.0; n_embd], vec![n_embd]),
            atencion: GradientesAtencion {
                peso_q: Tensor::new(
                    vec![0.0; bloque.atencion.proy_q.peso.datos.len()],
                    bloque.atencion.proy_q.peso.forma.clone(),
                ),
                sesgo_q: Tensor::new(vec![0.0; n_embd], vec![n_embd]),
                peso_k: Tensor::new(
                    vec![0.0; bloque.atencion.proy_k.peso.datos.len()],
                    bloque.atencion.proy_k.peso.forma.clone(),
                ),
                sesgo_k: Tensor::new(vec![0.0; n_embd], vec![n_embd]),
                peso_v: Tensor::new(
                    vec![0.0; bloque.atencion.proy_v.peso.datos.len()],
                    bloque.atencion.proy_v.peso.forma.clone(),
                ),
                sesgo_v: Tensor::new(vec![0.0; n_embd], vec![n_embd]),
                peso_salida: Tensor::new(
                    vec![0.0; bloque.atencion.proy_salida.peso.datos.len()],
                    bloque.atencion.proy_salida.peso.forma.clone(),
                ),
                sesgo_salida: Tensor::new(vec![0.0; n_embd], vec![n_embd]),
                x: Tensor::new(vec![0.0; n_embd], vec![1, n_embd]),
            },
            ln2_gamma: Tensor::new(vec![0.0; n_embd], vec![n_embd]),
            ln2_beta: Tensor::new(vec![0.0; n_embd], vec![n_embd]),
            mlp: GradientesMLP {
                peso_fc1: Tensor::new(
                    vec![0.0; bloque.mlp.fc1.peso.datos.len()],
                    bloque.mlp.fc1.peso.forma.clone(),
                ),
                sesgo_fc1: Tensor::new(
                    vec![0.0; bloque.mlp.fc1.sesgo.datos.len()],
                    bloque.mlp.fc1.sesgo.forma.clone(),
                ),
                peso_fc2: Tensor::new(
                    vec![0.0; bloque.mlp.fc2.peso.datos.len()],
                    bloque.mlp.fc2.peso.forma.clone(),
                ),
                sesgo_fc2: Tensor::new(
                    vec![0.0; bloque.mlp.fc2.sesgo.datos.len()],
                    bloque.mlp.fc2.sesgo.forma.clone(),
                ),
                x: Tensor::new(vec![0.0; n_embd], vec![1, n_embd]),
            },
        });
    }

    GradientesGPT2 {
        embedding_tokens: Tensor::new(vec![0.0; tamano_vocabulario * n_embd], vec![tamano_vocabulario, n_embd]),
        embedding_posiciones: Tensor::new(vec![0.0; tamano_bloque * n_embd], vec![tamano_bloque, n_embd]),
        grads_bloques,
        ln_final_gamma: Tensor::new(vec![0.0; n_embd], vec![n_embd]),
        ln_final_beta: Tensor::new(vec![0.0; n_embd], vec![n_embd]),
        peso_salida: Tensor::new(vec![0.0; n_embd * tamano_vocabulario], vec![n_embd, tamano_vocabulario]),
    }
}
/// Suma gradientes elemento por elemento (para la acumulación de gradientes)
fn sumar_gradientes(acumulados: &mut GradientesGPT2, nuevos: &GradientesGPT2) {
    // Embeddings
    for (a, b) in acumulados
        .embedding_tokens
        .datos
        .iter_mut()
        .zip(&nuevos.embedding_tokens.datos)
    {
        *a += b;
    }
    for (a, b) in acumulados
        .embedding_posiciones
        .datos
        .iter_mut()
        .zip(&nuevos.embedding_posiciones.datos)
    {
        *a += b;
    }

    // Gradientes de los bloques
    for (bloque_acu, bloque_nuevo) in acumulados.grads_bloques.iter_mut().zip(&nuevos.grads_bloques) {
        // Normalización de capa 1
        for (a, b) in bloque_acu
            .ln1_gamma
            .datos
            .iter_mut()
            .zip(&bloque_nuevo.ln1_gamma.datos)
        {
            *a += b;
        }
        for (a, b) in bloque_acu
            .ln1_beta
            .datos
            .iter_mut()
            .zip(&bloque_nuevo.ln1_beta.datos)
        {
            *a += b;
        }

        // Atención
        for (a, b) in bloque_acu
            .atencion
            .peso_q
            .datos
            .iter_mut()
            .zip(&bloque_nuevo.atencion.peso_q.datos)
        {
            *a += b;
        }
        for (a, b) in bloque_acu
            .atencion
            .sesgo_q
            .datos
            .iter_mut()
            .zip(&bloque_nuevo.atencion.sesgo_q.datos)
        {
            *a += b;
        }
        for (a, b) in bloque_acu
            .atencion
            .peso_k
            .datos
            .iter_mut()
            .zip(&bloque_nuevo.atencion.peso_k.datos)
        {
            *a += b;
        }
        for (a, b) in bloque_acu
            .atencion
            .sesgo_k
            .datos
            .iter_mut()
            .zip(&bloque_nuevo.atencion.sesgo_k.datos)
        {
            *a += b;
        }
        for (a, b) in bloque_acu
            .atencion
            .peso_v
            .datos
            .iter_mut()
            .zip(&bloque_nuevo.atencion.peso_v.datos)
        {
            *a += b;
        }
        for (a, b) in bloque_acu
            .atencion
            .sesgo_v
            .datos
            .iter_mut()
            .zip(&bloque_nuevo.atencion.sesgo_v.datos)
        {
            *a += b;
        }
        for (a, b) in bloque_acu
            .atencion
            .peso_salida
            .datos
            .iter_mut()
            .zip(&bloque_nuevo.atencion.peso_salida.datos)
        {
            *a += b;
        }
        for (a, b) in bloque_acu
            .atencion
            .sesgo_salida
            .datos
            .iter_mut()
            .zip(&bloque_nuevo.atencion.sesgo_salida.datos)
        {
            *a += b;
        }

        // Normalización de capa 2
        for (a, b) in bloque_acu
            .ln2_gamma
            .datos
            .iter_mut()
            .zip(&bloque_nuevo.ln2_gamma.datos)
        {
            *a += b;
        }
        for (a, b) in bloque_acu
            .ln2_beta
            .datos
            .iter_mut()
            .zip(&bloque_nuevo.ln2_beta.datos)
        {
            *a += b;
        }

        // MLP
        for (a, b) in bloque_acu
            .mlp
            .peso_fc1
            .datos
            .iter_mut()
            .zip(&bloque_nuevo.mlp.peso_fc1.datos)
        {
            *a += b;
        }
        for (a, b) in bloque_acu
            .mlp
            .sesgo_fc1
            .datos
            .iter_mut()
            .zip(&bloque_nuevo.mlp.sesgo_fc1.datos)
        {
            *a += b;
        }
        for (a, b) in bloque_acu
            .mlp
            .peso_fc2
            .datos
            .iter_mut()
            .zip(&bloque_nuevo.mlp.peso_fc2.datos)
        {
            *a += b;
        }
        for (a, b) in bloque_acu
            .mlp
            .sesgo_fc2
            .datos
            .iter_mut()
            .zip(&bloque_nuevo.mlp.sesgo_fc2.datos)
        {
            *a += b;
        }
    }

    // Normalización de capa final
    for (a, b) in acumulados
        .ln_final_gamma
        .datos
        .iter_mut()
        .zip(&nuevos.ln_final_gamma.datos)
    {
        *a += b;
    }
    for (a, b) in acumulados
        .ln_final_beta
        .datos
        .iter_mut()
        .zip(&nuevos.ln_final_beta.datos)
    {
        *a += b;
    }

    // Peso de salida
    for (a, b) in acumulados
        .peso_salida
        .datos
        .iter_mut()
        .zip(&nuevos.peso_salida.datos)
    {
        *a += b;
    }
}

/// Escala todos los gradientes por un factor constante
fn escalar_gradientes(gradientes: &mut GradientesGPT2, escala: f32) {
    for v in &mut gradientes.embedding_tokens.datos {
        *v *= escala;
    }
    for v in &mut gradientes.embedding_posiciones.datos {
        *v *= escala;
    }

    for bloque in &mut gradientes.grads_bloques {
        for v in &mut bloque.ln1_gamma.datos {
            *v *= escala;
        }
        for v in &mut bloque.ln1_beta.datos {
            *v *= escala;
        }
        for v in &mut bloque.atencion.peso_q.datos {
            *v *= escala;
        }
        for v in &mut bloque.atencion.sesgo_q.datos {
            *v *= escala;
        }
        for v in &mut bloque.atencion.peso_k.datos {
            *v *= escala;
        }
        for v in &mut bloque.atencion.sesgo_k.datos {
            *v *= escala;
        }
        for v in &mut bloque.atencion.peso_v.datos {
            *v *= escala;
        }
        for v in &mut bloque.atencion.sesgo_v.datos {
            *v *= escala;
        }
        for v in &mut bloque.atencion.peso_salida.datos {
            *v *= escala;
        }
        for v in &mut bloque.atencion.sesgo_salida.datos {
            *v *= escala;
        }
        for v in &mut bloque.ln2_gamma.datos {
            *v *= escala;
        }
        for v in &mut bloque.ln2_beta.datos {
            *v *= escala;
        }
        for v in &mut bloque.mlp.peso_fc1.datos {
            *v *= escala;
        }
        for v in &mut bloque.mlp.sesgo_fc1.datos {
            *v *= escala;
        }
        for v in &mut bloque.mlp.peso_fc2.datos {
            *v *= escala;
        }
        for v in &mut bloque.mlp.sesgo_fc2.datos {
            *v *= escala;
        }
    }

    for v in &mut gradientes.ln_final_gamma.datos {
        *v *= escala;
    }
    for v in &mut gradientes.ln_final_beta.datos {
        *v *= escala;
    }
    for v in &mut gradientes.peso_salida.datos {
        *v *= escala;
    }
}
/// Obtiene una posición de inicio aleatoria para un lote de entrenamiento
fn obtener_inicio_lote_aleatorio(longitud_tokens: usize, longitud_secuencia: usize) -> usize {
    use std::cell::Cell;
    thread_local! {
        static RNG: Cell<u64> = const { Cell::new(98765) };
    }

    let inicio_maximo = longitud_tokens.saturating_sub(longitud_secuencia + 1);
    if inicio_maximo == 0 {
        return 0;
    }

    RNG.with(|rng| {
        let mut estado = rng.get();
        estado = estado
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        rng.set(estado);
        (estado >> 33) as usize % inicio_maximo
    })
}

/// Obtiene la tasa de aprendizaje con programación de calentamiento (warmup) y decaimiento coseno
fn obtener_lr_con_programacion(
    paso: usize,
    pasos_calentamiento: usize,
    pasos_maximos: usize,
    lr_maximo: f32,
    lr_minimo: f32,
) -> f32 {
    if paso < pasos_calentamiento {
        // Calentamiento lineal: 0 -> lr_maximo
        lr_maximo * (paso as f32 / pasos_calentamiento as f32)
    } else {
        // Decaimiento coseno: lr_maximo -> lr_minimo
        let pasos_decaimiento = pasos_maximos - pasos_calentamiento;
        let paso_decaimiento = paso - pasos_calentamiento;
        let proporcion_decaimiento = paso_decaimiento as f32 / pasos_decaimiento as f32;

        // Fórmula de decaimiento coseno (cosine annealing)
        let decaimiento_coseno = 0.5 * (1.0 + (std::f32::consts::PI * proporcion_decaimiento).cos());
        lr_minimo + (lr_maximo - lr_minimo) * decaimiento_coseno
    }
}

/// Programador de tasa de aprendizaje adaptativo que responde a estancamientos en la pérdida de validación
///
/// Este programador funciona junto con la programación coseno base para reducir automáticamente
/// la tasa de aprendizaje cuando la pérdida de validación deja de mejorar.
pub struct ProgramadorLRAdaptativo {
    multiplicador_lr_actual: f32,
    paciencia: usize,
    factor_reduccion: f32,
    proporcion_lr_minima: f32,
    enfriamiento: usize,

    // Seguimiento de estado
    mejor_perdida_val: f32,
    pasos_sin_mejora: usize,
    enfriamiento_restante: usize,
}

impl ProgramadorLRAdaptativo {
    pub fn nuevo(_lr_base: f32, paciencia: usize) -> Self {
        Self {
            multiplicador_lr_actual: 1.0,
            paciencia,
            factor_reduccion: 0.5,
            proporcion_lr_minima: 0.1,
            enfriamiento: 500,
            mejor_perdida_val: f32::INFINITY,
            pasos_sin_mejora: 0,
            enfriamiento_restante: 0,
        }
    }

    /// Actualiza el programador con la última pérdida de validación
    /// Devuelve true si se redujo la tasa de aprendizaje
    pub fn paso(&mut self, perdida_val: f32, pasos_transcurridos: usize) -> bool {
        // Actualizar enfriamiento
        if self.enfriamiento_restante > 0 {
            self.enfriamiento_restante = self.enfriamiento_restante.saturating_sub(pasos_transcurridos);
        }

        // Comprobar si hay mejora
        if perdida_val < self.mejor_perdida_val {
            self.mejor_perdida_val = perdida_val;
            self.pasos_sin_mejora = 0;
            false
        } else {
            self.pasos_sin_mejora += pasos_transcurridos;

            // Comprobar si debemos reducir la tasa de aprendizaje
            if self.pasos_sin_mejora >= self.paciencia && self.enfriamiento_restante == 0 {
                // Reducir la tasa de aprendizaje
                self.multiplicador_lr_actual *= self.factor_reduccion;
                self.multiplicador_lr_actual = self.multiplicador_lr_actual.max(self.proporcion_lr_minima);
                self.enfriamiento_restante = self.enfriamiento;
                self.pasos_sin_mejora = 0;
                true
            } else {
                false
            }
        }
    }

    /// Obtiene el multiplicador de LR actual para aplicar al LR programado
    pub fn obtener_multiplicador(&self) -> f32 {
        self.multiplicador_lr_actual
    }
}

pub fn entrenar_gpt2(
    modelo: &mut GPT2Entrenable,
    tokenizador: &TokenizadorBPE,
    texto: &str,
    num_pasos: usize,
    tasa_aprendizaje: f32,
    longitud_secuencia: usize,
    dir_ejecucion: Option<&str>,
    paciencia: usize,
    fraccion_calentamiento: f32,
    norma_recorte_gradiente: f32,
    fraccion_validacion: f32,
    decaimiento_peso: f32,
) {
    use crate::registrador_entrenamiento::{calcular_perdida_dataset, dividir_entrenamiento_validacion, RegistradorEntrenamiento};

    // Crear directorio de ejecución para esta sesión de entrenamiento
    let dir_ejecucion = if let Some(dir) = dir_ejecucion {
        dir.to_string()
    } else {
        // Generar nombre de directorio con marca de tiempo Unix
        use std::time::{SystemTime, UNIX_EPOCH};
        let marca_tiempo = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        format!("ejecucion_{}", marca_tiempo)
    };

    // Crear el directorio
    std::fs::create_dir_all(&dir_ejecucion).expect("Error al crear el directorio de ejecución");
    println!("📁 Directorio de ejecución: {}/\n", dir_ejecucion);

    println!("\n=== Entrenando Transformer Estilo GPT-2 ===\n");

    let tokens = tokenizador.codificar(texto);

    // Dividir en entrenamiento/validación basado en la fracción proporcionada
    let (tokens_entrenamiento, tokens_val) = dividir_entrenamiento_validacion(&tokens, fraccion_validacion);
    println!(
        "Tokens de entrenamiento: {} ({:.1}%), Tokens de val: {} ({:.1}%)\n",
        tokens_entrenamiento.len(),
        (1.0 - fraccion_validacion) * 100.0,
        tokens_val.len(),
        fraccion_validacion * 100.0
    );

    // Inicializar registrador en el directorio de ejecución
    let ruta_registro = format!("{}/registro_entrenamiento.csv", dir_ejecucion);
    let mut registrador = RegistradorEntrenamiento::new(&ruta_registro).expect("Error al crear el registro de entrenamiento");

    // Inicializar optimizador AdamW
    println!("🚀 Inicializando optimizador AdamW...");
    let mut optimizador = OptimizadorAdamW::new(modelo);

    // Ajustar beta2 para un mejor seguimiento del segundo momento (experimental)
    optimizador.beta2 = 0.98;

    println!(
        "   Beta1: {}, Beta2: {}, Épsilon: {}",
        optimizador.beta1, optimizador.beta2, optimizador.epsilon
    );
    println!("   Decaimiento de peso: {}", decaimiento_peso);
    println!("   Recorte de gradientes: norma_max = {}", norma_recorte_gradiente);

    // Configuración de acumulación de gradientes
    let pasos_acumulacion = 1;
    println!(
        "   Acumulación de gradientes: {} mini-lotes (tamaño de lote efectivo = {})\n",
        pasos_acumulacion, pasos_acumulacion
    );

    // Configuración de la programación de la tasa de aprendizaje
    let pasos_calentamiento = (num_pasos as f32 * fraccion_calentamiento) as usize;
    let lr_minimo = tasa_aprendizaje * 0.1; // Decaer al 10% del LR máximo
    println!("📊 Programación de la Tasa de Aprendizaje (LR):");
    println!(
        "   Pasos de calentamiento: {} ({}% del entrenamiento)",
        pasos_calentamiento,
        (pasos_calentamiento as f32 / num_pasos as f32 * 100.0) as usize
    );
    println!("   Tasa de aprendizaje máxima: {}", tasa_aprendizaje);
    println!("   Tasa de aprendizaje mínima: {}", lr_minimo);
    println!("   Programación: Calentamiento lineal → Decaimiento coseno\n");

    // Inicializar programador adaptativo para la fase posterior al calentamiento
    let mut programador_adaptativo = ProgramadorLRAdaptativo::nuevo(tasa_aprendizaje, 500);
    println!("🎯 Programador de LR Adaptativo:");
    println!("   Paciencia: 500 pasos");
    println!("   Factor de reducción: 0.5x en estancamiento");
    println!("   Enfriamiento: 500 pasos entre reducciones");
    println!("   Proporción de LR mínima: 10% del LR base\n");

    // Configuración de detención anticipada (early stopping)
    println!("⏹️  Detención anticipada: paciencia = {} pasos\n", paciencia);
    let mut mejor_perdida_val = f32::INFINITY;
    let mut mejor_paso_val = 0;
    let mut pasos_sin_mejora = 0;

    // Rastrear hilos de guardado de puntos de control en segundo plano
    let mut handles_puntos_control: Vec<std::thread::JoinHandle<std::io::Result<()>>> = Vec::new();

    // Bucle de entrenamiento
    for paso in 0..num_pasos {
        // Calcular tasa de aprendizaje actual con programación adaptativa
        let lr_actual = if paso < pasos_calentamiento {
            // Fase de calentamiento estándar
            tasa_aprendizaje * (paso as f32 / pasos_calentamiento as f32)
        } else {
            // Fase adaptativa basada en rendimiento

            // Programación coseno base
            let lr_programado =
                obtener_lr_con_programacion(paso, pasos_calentamiento, num_pasos, tasa_aprendizaje, lr_minimo);

            // Aplicar multiplicador adaptativo
            let lr_adaptativo = lr_programado * programador_adaptativo.obtener_multiplicador();

            // Asegurar que no bajemos del mínimo
            lr_adaptativo.max(tasa_aprendizaje * programador_adaptativo.proporcion_lr_minima)
        };

        // Acumular gradientes sobre múltiples mini-lotes
        let mut gradientes_acumulados = inicializar_gradientes_cero(modelo);
        let mut perdida_total = 0.0;

        for _micro_paso in 0..pasos_acumulacion {
            // Muestreo aleatorio para mejor generalización
            let inicio = obtener_inicio_lote_aleatorio(tokens_entrenamiento.len(), longitud_secuencia);
            let secuencia_entrada = &tokens_entrenamiento[inicio..inicio + longitud_secuencia];
            let secuencia_objetivo = &tokens_entrenamiento[inicio + 1..inicio + longitud_secuencia + 1];

            // Propagación hacia adelante (Forward)
            let (logits, cache) = modelo.forward(secuencia_entrada);
            let perdida = modelo.calcular_perdida(&logits, secuencia_objetivo);
            perdida_total += perdida;

            // Propagación hacia atrás (Backward)
            let gradientes = modelo.backward(&logits, secuencia_objetivo, &cache);

            // Acumular gradientes
            sumar_gradientes(&mut gradientes_acumulados, &gradientes);
        }

        // Promediar los gradientes acumulados
        let perdida_entrenamiento = perdida_total / pasos_acumulacion as f32;
        escalar_gradientes(&mut gradientes_acumulados, 1.0 / pasos_acumulacion as f32);

        // Recorte de gradientes
        recortar_gradientes(&mut gradientes_acumulados, norma_recorte_gradiente);

        // Actualizar con AdamW (usando tasa de aprendizaje programada y decaimiento de peso)
        actualizar_adamw(
            modelo,
            &gradientes_acumulados,
            &mut optimizador,
            lr_actual,
            decaimiento_peso,
        );

        // Guardar punto de control cada 250 pasos (con estado de entrenamiento completo) - ¡en segundo plano!
        if paso > 0 && paso % 250 == 0 {
            let punto_control = PuntoControl {
                modelo: modelo.clonar_superficialmente(),
                optimizador: Some(optimizador.clonar_superficialmente()),
                tokenizador: Some((*tokenizador).clone()),
                paso,
                mejor_perdida_val,
                mejor_paso_val,
            };
            let ruta_punto_control = format!("{}/punto_control_paso_{}.bin", dir_ejecucion, paso);
            println!(
                "💾 Guardando punto de control en {} (segundo plano)...",
                ruta_punto_control
            );
            let handle = punto_control.guardar_en_segundo_plano(ruta_punto_control);
            handles_puntos_control.push(handle);
        }

        // Registrar progreso cada 50 pasos
        if paso % 50 == 0 || paso == num_pasos - 1 {
            // Calcular pérdida de validación
            let perdida_val = calcular_perdida_dataset(
                tokens_val,
                longitud_secuencia,
                10, // Usar 10 lotes de validación
                |entrada, objetivo| {
                    let (logits, _) = modelo.forward(entrada);
                    modelo.calcular_perdida(&logits, objetivo)
                },
            );

            // Comprobación de detención anticipada
            if perdida_val < mejor_perdida_val {
                mejor_perdida_val = perdida_val;
                mejor_paso_val = paso;
                pasos_sin_mejora = 0;

                // Guardar mejor punto de control del modelo (con estado de entrenamiento completo) - ¡en segundo plano!
                println!(
                    "📊 Nueva mejor pérdida de validación: {:.4} (perplejidad: {:.2})",
                    perdida_val,
                    perdida_val.exp()
                );
                let punto_control = PuntoControl {
                    modelo: modelo.clonar_superficialmente(),
                    optimizador: Some(optimizador.clonar_superficialmente()),
                    tokenizador: Some((*tokenizador).clone()),
                    paso,
                    mejor_perdida_val,
                    mejor_paso_val,
                };
                println!("💾 Guardando mejor punto de control (segundo plano)...");
                let handle = punto_control.guardar_en_segundo_plano(format!("{}/punto_control_mejor.bin", dir_ejecucion));
                handles_puntos_control.push(handle);
            } else {
                pasos_sin_mejora += 50; // Comprobamos cada 50 pasos
            }

            // Actualizar programador adaptativo (solo después del calentamiento)
            if paso > pasos_calentamiento {
                let lr_reducido = programador_adaptativo.paso(perdida_val, 50);
                if lr_reducido {
                    println!(
                        "🔧 Estancamiento de validación detectado, reduciendo multiplicador LR a {:.3}",
                        programador_adaptativo.obtener_multiplicador()
                    );
                }
            }

            // Generar muestra cada 200 pasos
            let muestra = if paso % 200 == 0 || paso == num_pasos - 1 {
                let prompt = "En un lugar de la Mancha, de cuyo nombre no quiero acordarme"; // Versión Quijotesca para la muestra
                let tokens_prompt = tokenizador.codificar(prompt);
                let generado = modelo.generar(&tokens_prompt, 30, 1.0);
                let texto_generado = tokenizador.decodificar(&generado[tokens_prompt.len()..]);
                let vista_previa: String = texto_generado.chars().take(50).collect();
                Some(format!("{}{}", prompt, vista_previa))
            } else {
                None
            };

            registrador
                .log(paso, lr_actual, perdida_entrenamiento, perdida_val, muestra.as_deref())
                .expect("Error al escribir el registro");

            // Interrumpir si se activa la detención anticipada
            if pasos_sin_mejora >= paciencia {
                println!("\n🛑 ¡Detención anticipada activada en el paso {}!", paso);
                println!(
                    "   Mejor pérdida de validación: {:.4} (perplejidad: {:.2}) en el paso {}",
                    mejor_perdida_val,
                    mejor_perdida_val.exp(),
                    mejor_paso_val
                );
                println!("   Sin mejoras durante {} pasos", pasos_sin_mejora);

                // Guardar punto de control de detención anticipada (con estado de entrenamiento completo) - ¡en segundo plano!
                let punto_control = PuntoControl {
                    modelo: modelo.clonar_superficialmente(),
                    optimizador: Some(optimizador.clonar_superficialmente()),
                    tokenizador: Some((*tokenizador).clone()),
                    paso,
                    mejor_perdida_val,
                    mejor_paso_val,
                };
                let ruta_detencion_anticipada =
                    format!("{}/punto_control_detencion_anticipada_paso_{}.bin", dir_ejecucion, paso);
                println!("💾 Guardando punto de control de detención anticipada (segundo plano)...");
                let handle = punto_control.guardar_en_segundo_plano(ruta_detencion_anticipada);
                handles_puntos_control.push(handle);

                break;
            }
        }
    }

    // Guardar punto de control final (si completamos todos los pasos sin detención anticipada) - ¡en segundo plano!
    println!("\n💾 Guardando punto de control final (segundo plano)...");
    let punto_control_final = PuntoControl {
        modelo: modelo.clonar_superficialmente(),
        optimizador: Some(optimizador.clonar_superficialmente()),
        tokenizador: Some((*tokenizador).clone()),
        paso: num_pasos - 1,
        mejor_perdida_val,
        mejor_paso_val,
    };
    let handle =
        punto_control_final.guardar_en_segundo_plano(format!("{}/punto_control_final.bin", dir_ejecucion).to_string());
    handles_puntos_control.push(handle);

    // Esperar a que se completen todos los guardados de puntos de control en segundo plano
    println!(
        "\n⏳ Esperando a que se completen {} guardados de puntos de control en segundo plano...",
        handles_puntos_control.len()
    );
    for (i, handle) in handles_puntos_control.into_iter().enumerate() {
        match handle.join() {
            Ok(Ok(())) => println!("   ✅ Punto de control {} guardado exitosamente", i + 1),
            Ok(Err(e)) => eprintln!("   ⚠️  Punto de control {} falló: {}", i + 1, e),
            Err(_) => eprintln!("   ⚠️  El hilo del punto de control {} entró en pánico", i + 1),
        }
    }
    println!("✅ ¡Todos los puntos de control guardados!");

    println!("\n=== Entrenamiento Completado ===\n");
    println!("Registro de entrenamiento guardado en: registro_entrenamiento.csv");
    println!(
        "Mejor pérdida de validación: {:.4} (perplejidad: {:.2}) en el paso {}",
        mejor_perdida_val,
        mejor_perdida_val.exp(),
        mejor_paso_val
    );
    println!("Mejor modelo guardado en: punto_control_mejor.bin");
}