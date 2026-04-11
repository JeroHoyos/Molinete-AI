//! Arquitectura del Modelo GPT-2
//!
//! Este módulo implementa un modelo transformer estilo GPT-2 completo desde cero.
//! La arquitectura consiste en:
//!
//! - **Embeddings de tokens y posiciones**: Convierte los IDs de los tokens en vectores
//! - **Bloques Transformer**: Pila de capas de atención y prealimentación (feedforward)
//! - **Normalización de capa (Layer normalization)**: Estabiliza las activaciones
//! - **Proyección lineal**: Capa final hacia los logits del vocabulario
//!
//! ## Descripción general de la arquitectura
//!
//! ```text
//! Tokens de entrada [lote, long_sec]
//!     ↓
//! embedding de tokens [lote, long_sec, dim_embd]
//!     + embedding de posiciones [long_sec, dim_embd]
//!     ↓
//! Bloque Transformer 1 (Atención + MLP)
//!     ↓
//! Bloque Transformer 2
//!     ↓
//!     ...
//!     ↓
//! Bloque Transformer N
//!     ↓
//! Normalización de capa (Layer Norm)
//!     ↓
//! Lineal → [lote, long_sec, tamano_vocab]
//! ```
//!
//! ## Solo paso hacia adelante (Forward Pass)
//!
//! Esta implementación proporciona el **paso hacia adelante** (forward pass) para inferencia y comprensión.
//! El entrenamiento (paso hacia atrás, gradientes, optimización) no está incluido en la Fase 3.
//!
//! ## Ejemplo
//!
//! ```rust,no_run
//! use molineteai::{GPT2, Config};
//!
//! // Crear una configuración de modelo miniatura (tiny)
//! let config = Config::tiny(512); // tamaño de vocabulario de 512
//! let model = GPT2::new(&config);
//!
//! // Paso hacia adelante: tokens → logits
//! // let logits = model.forward(&token_ids);
//! ```

// Módulo interno de nuestro proyecto que define la estructura base y las operaciones matemáticas de los tensores.
use crate::tensor::Tensor;

// Librería externa para generar números aleatorios siguiendo distribuciones estadísticas (como la Normal), muy útil para inicializar los pesos del modelo.
use rand_distr::{Distribution, Normal};

// Librería externa para guardar (serializar) y cargar (deserializar) estructuras de datos, facilitando su almacenamiento en disco o transferencia.


/// Configuración del modelo
///
/// Define los hiperparámetros de la arquitectura para un modelo estilo GPT-2.
///
/// # Campos
///
/// - `vocab_size`: Número de tokens en el vocabulario
/// - `n_embd`: Dimensión de las incrustaciones/embeddings (ancho del modelo)
/// - `n_heads`: Número de cabezales de atención por capa
/// - `n_layers`: Número de bloques transformer
/// - `block_size`: Longitud máxima de la secuencia (ventana de contexto)
/// - `dropout_rate`: Tasa de abandono (dropout) para la regularización
///
/// # Fórmula de recuento de parámetros
///
/// Parámetros aproximados:
/// ```text
/// incrustaciones ≈ vocab_size × n_embd × 2  (incrustaciones de tokens + posiciones)
/// por_capa ≈ 12 × n_embd²  (atención + MLP + normalizaciones de capa)
/// total ≈ incrustaciones + (n_layers × por_capa)
/// ```
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub block_size: usize,
    pub dropout_rate: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 50257, // Tamaño del vocabulario de GPT-2
            n_embd: 768,       // Dimensión de las incrustaciones (embeddings)
            n_heads: 12,       // Número de cabezales de atención
            n_layers: 12,      // Número de bloques transformer
            block_size: 1024,  // Longitud máxima de la secuencia
            dropout_rate: 0.1, // Probabilidad de abandono (dropout)
        }
    }
}

impl Config {
    /// Crea una configuración miniatura (tiny) para experimentos rápidos
    ///
    /// **~50K parámetros** - Muy rápido, ideal para pruebas (2-5 minutos de entrenamiento)
    ///
    /// # Argumentos
    ///
    /// * `vocab_size` - Tamaño del vocabulario (ej. proveniente del tokenizador)
    pub fn tiny(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            n_embd: 64,        // Incrustación (embedding) muy pequeña
            n_heads: 1,        // Atención de un solo cabezal
            n_layers: 2,       // Poco profundo (pocas capas)
            block_size: 64,    // Contexto corto
            dropout_rate: 0.1, // Probabilidad de abandono (dropout)
        }
    }

    /// Crea una configuración pequeña para experimentos
    ///
    /// **~200K parámetros** (con un vocabulario pequeño) - Buen equilibrio entre velocidad y capacidad
    ///
    /// # Argumentos
    ///
    /// * `vocab_size` - Tamaño del vocabulario
    pub fn small(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            n_embd: 128,       // Incrustación (embedding) pequeña
            n_heads: 1,        // Atención de un solo cabezal
            n_layers: 3,       // Profundidad media
            block_size: 128,   // Contexto medio
            dropout_rate: 0.1, // Probabilidad de abandono (dropout)
        }
    }

    /// Crea una configuración mediana
    ///
    /// **~4M parámetros** (con un vocabulario pequeño) - Capacidad sustancial
    ///
    /// # Argumentos
    ///
    /// * `vocab_size` - Tamaño del vocabulario
    pub fn medium(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            n_embd: 256,       // Incrustación (embedding) mediana
            n_heads: 4,        // Atención multicabezal (multi-head)
            n_layers: 4,       // Profundidad media
            block_size: 256,   // Contexto medio
            dropout_rate: 0.1, // Probabilidad de abandono (dropout)
        }
    }

    /// Crea la configuración de GPT-2 Small
    ///
    /// **~163M de parámetros** (con un vocabulario GPT-2 de 50257 tokens)
    /// **~86M de parámetros** (con un vocabulario de demostración más pequeño de 512 tokens)
    ///
    /// Esto coincide con la arquitectura GPT-2 Small de OpenAI:
    /// - Embeddings de 768 dimensiones
    /// - 12 capas transformer
    /// - 12 cabezales de atención
    /// - Ventana de contexto de 1024 tokens
    ///
    /// Nota: GPT-2 Small de OpenAI a menudo se cita con "117-124M de parámetros" porque
    /// utilizan vinculación de pesos (weight tying, la matriz de incrustación de tokens se reutiliza como la capa final del modelo de lenguaje).
    /// Nuestra implementación utiliza pesos separados por claridad, resultando en ~163M
    /// de parámetros con el vocabulario completo de GPT-2 (50257 tokens). Con la vinculación de pesos,
    /// esta configuración tendría ~124M de parámetros.
    ///
    /// # Argumentos
    ///
    /// * `vocab_size` - Tamaño del vocabulario (usa 50257 para el tokenizador de GPT-2)
    pub fn gpt2_small(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            n_embd: 768,       // GPT-2 Small
            n_heads: 12,       // GPT-2 Small
            n_layers: 12,      // GPT-2 Small
            block_size: 1024,  // Contexto estándar de GPT-2
            dropout_rate: 0.1, // Probabilidad de abandono (dropout)
        }
    }
}

//
// ============================================================================
// FUNCIONES DE ACTIVACIÓN
// ============================================================================
//

    /// Activación GELU (Unidad Lineal de Error Gaussiano / Gaussian Error Linear Unit)
    ///
    /// GELU se utiliza en transformers en lugar de ReLU porque proporciona gradientes
    /// más suaves y a menudo tiene un mejor rendimiento en la práctica.
    ///
    /// # Fórmula
    ///
    /// ```text
    /// GELU(x) = x × Φ(x)
    /// donde Φ(x) es la función de distribución acumulativa de la normal estándar
    /// ```
    ///
    /// # Aproximación
    ///
    /// Usamos la aproximación con tanh por eficiencia:
    ///
    /// ```text
    /// GELU(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
    /// ```
    ///
    /// Esto es más rápido que calcular la CDF (función de distribución acumulativa) exacta y es lo suficientemente preciso para las redes neuronales.
    ///
    /// # Argumentos
    ///
    /// * `x` - Tensor de entrada
    ///
    /// # Retorna
    ///
    /// Tensor con la activación GELU aplicada elemento por elemento
    pub fn gelu(x: &Tensor) -> Tensor {
        // Constantes para la aproximación
        let raiz_2_sobre_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
        let coef = 0.044715_f32;

        // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        let resultado: Vec<f32> = x
            .datos
            .iter()
            .map(|&val| {
                let x_cubo = val * val * val;
                let interno = raiz_2_sobre_pi * (val + coef * x_cubo);
                0.5 * val * (1.0 + interno.tanh())
            })
            .collect();

        Tensor::new(resultado, x.forma.clone())
    }

//
// ============================================================================
// CAPA EMBEDDING 
// ============================================================================
//

/// Capa de embedding de tokens (Token embedding)
///
/// Convierte los IDs de los tokens en vectores densos. Esta es una tabla de búsqueda
/// aprendible donde cada ID de token se mapea a un vector de embedding de tamaño fijo.
///
/// # Transformación de la forma (Shape Transformation)
///
/// ```text
/// Entrada: [lote, long_sec]  (IDs de tokens)
/// Salida:  [lote, long_sec, dim_embd]  (vectores de embedding)
/// ```
///
/// # Implementación
///
/// La tabla de embedding tiene la forma `[tamano_vocab, dim_embd]`. Para cada ID de token,
/// buscamos la fila correspondiente en esta tabla.
pub struct Embedding {
    /// Matriz de pesos de la embedding: [tamano_vocab, dim_embd]
    pub peso: Tensor,
}

impl Embedding {
    /// Create a new embedding layer with random initialization
    ///
    /// Weights are initialized from N(0, 0.02) following GPT-2.
    /// This uses a normal distribution with mean 0 and standard deviation 0.02,
    /// which helps with gradient flow during training.
    ///
    /// # Arguments
    ///
    /// * `vocab_size` - Number of tokens in vocabulary
    /// * `n_embd` - Embedding dimension
    pub fn new(vocab_size: usize, n_embd: usize) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.02).unwrap();
        let weight_data: Vec<f32> = (0..vocab_size * n_embd)
            .map(|_| normal.sample(&mut rng))
            .collect();

        Self {
            peso: Tensor::new(weight_data, vec![vocab_size, n_embd]),
        }
    }

    /// Forward pass: look up embeddings for token IDs
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Shape [batch, seq_len] with token indices
    ///
    /// # Returns
    ///
    /// Embedding vectors of shape [batch, seq_len, n_embd]
    pub fn forward(&self, token_ids: &[Vec<usize>]) -> Tensor {
        let batch_size = token_ids.len();
        let seq_len = token_ids[0].len();
        let n_embd = self.peso.forma[1];

        let mut output = Vec::with_capacity(batch_size * seq_len * n_embd);

        for batch in token_ids {
            for &token_id in batch {
                assert!(
                    token_id < self.peso.forma[0],
                    "Token ID {} out of vocab range (vocab_size = {})",
                    token_id,
                    self.peso.forma[0]
                );
                // Copy the embedding vector for this token
                let start = token_id * n_embd;
                let end = start + n_embd;
                output.extend_from_slice(&self.peso.datos[start..end]);
            }
        }

        Tensor::new(output, vec![batch_size, seq_len, n_embd])
    }
}

//
// ============================================================================
// LAYER NORMALIZATION
// ============================================================================
//

/// Capa de Normalización 
///
/// Normaliza las activaciones a lo largo de la última dimensión para tener media cero y varianza unitaria.
/// Esto estabiliza el entrenamiento y se aplica antes de cada subcapa (atención y MLP).
///
/// # Fórmula
///
/// ```text
/// salida = (entrada - media) / sqrt(varianza + eps) × gamma + beta
/// ```
///
/// donde `gamma` y `beta` son parámetros aprendibles.
///
/// # ¿Por qué Normalización de Capa (Layer Norm)?
///
/// A diferencia de la normalización por lotes (batch normalization, que normaliza a lo largo del lote), layer norm normaliza
/// a lo largo de las características para cada muestra de forma independiente. Esto funciona mejor para:
/// - Longitudes de secuencia variables
/// - Tamaños de lote pequeños
/// - Arquitecturas recurrentes/transformer
pub struct LayerNorm {
    /// Parámetro de escala (aprendible): [dim_embd]
    pub gamma: Tensor,
    /// Parámetro de desplazamiento (aprendible): [dim_embd]
    pub beta: Tensor,
    /// Constante pequeña para estabilidad numérica
    pub eps: f32,
}

impl LayerNorm {
    /// Crea una nueva capa de normalización
    ///
    /// # Argumentos
    ///
    /// * `n_embd` - Dimensión de las características sobre la cual normalizar
    /// * `eps` - Constante pequeña para prevenir la división por cero (por defecto: 1e-5)
    pub fn new(n_embd: usize, eps: f32) -> Self {
        // Inicializa gamma en 1 (sin escala inicialmente)
        let gamma = Tensor::new(vec![1.0; n_embd], vec![n_embd]);
        // Inicializa beta en 0 (sin desplazamiento inicialmente)
        let beta = Tensor::new(vec![0.0; n_embd], vec![n_embd]);

        Self { gamma, beta, eps }
    }

    /// Forward Pass: normaliza a lo largo de la última dimensión
    ///
    /// # Argumentos
    ///
    /// * `x` - Tensor de entrada, típicamente [lote, long_sec, dim_embd]
    ///
    /// # Retorna
    ///
    /// Tensor normalizado con la misma forma que la entrada
    pub fn forward(&self, entrada: &Tensor) -> Tensor {
        // Calcula la media y la varianza a lo largo de la última dimensión
        let media = entrada.mean(-1, true);
        let varianza = entrada.var(-1, true);

        // Normaliza: (entrada - media) / sqrt(varianza + epsilon)
        let normalizado = entrada
            .sub(&media)
            .div(&varianza.add_scalar(self.eps).sqrt());

        // Aplica escala y desplazamiento: normalizado * gamma + beta
        normalizado.mul(&self.gamma).add(&self.beta)
    }
}

//
// ============================================================================
// CAPA LINEAL
// ============================================================================
//

/// Capa lineal (completamente conectada)
///
/// Aplica una transformación afín: `y = x @ W + b`
///
/// # Transformación de la forma
///
/// ```text
/// Entrada: [*, caracteristicas_entrada]
/// Salida:  [*, caracteristicas_salida]
/// ```
///
/// donde `*` representa cualquier número de dimensiones iniciales (lote, secuencia, etc.)
pub struct Linear {
    /// Matriz de pesos: [caracteristicas_entrada, caracteristicas_salida]
    pub peso: Tensor,
    /// Vector de sesgo (bias): [caracteristicas_salida]
    pub sesgo: Tensor,
}

impl Linear {
    /// Crea una nueva capa lineal con inicialización aleatoria
    ///
    /// Los pesos se inicializan desde N(0, 0.02) siguiendo a GPT-2.
    /// Esto utiliza una distribución normal con media 0 y desviación estándar de 0.02,
    /// lo que ayuda con el flujo del gradiente durante el entrenamiento.
    /// El sesgo se inicializa en ceros.
    ///
    /// # Argumentos
    ///
    /// * `caracteristicas_entrada` - Dimensión de entrada
    /// * `caracteristicas_salida` - Dimensión de salida
    pub fn new(caracteristicas_entrada: usize, caracteristicas_salida: usize) -> Self {
        let mut gen_aleatorio = rand::thread_rng();
        let normal = Normal::new(0.0, 0.02).unwrap();

        let datos_peso: Vec<f32> = (0..caracteristicas_entrada * caracteristicas_salida)
            .map(|_| normal.sample(&mut gen_aleatorio))
            .collect();

        let peso = Tensor::new(datos_peso, vec![caracteristicas_entrada, caracteristicas_salida]);
        let sesgo = Tensor::ceros(vec![caracteristicas_salida]);

        Self { peso, sesgo }
    }

    /// Paso hacia adelante (forward pass): y = x @ W + b
    ///
    /// # Argumentos
    ///
    /// * `entrada` - Tensor de entrada [..., caracteristicas_entrada]
    ///
    /// # Retorna
    ///
    /// Tensor de salida [..., caracteristicas_salida]
    pub fn forward(&self, entrada: &Tensor) -> Tensor {
        // Por simplicidad, manejamos la entrada 3D [lote, sec, caracteristicas_entrada]
        // Cambiamos la forma a 2D, multiplicamos matrices, y luego volvemos a la forma original
        let tamano_lote = entrada.forma[0];
        let long_sec = entrada.forma[1];
        let carac_entrada = entrada.forma[2];

        // Cambiamos la forma a [lote * sec, caracteristicas_entrada]
        let entrada_2d = entrada.reshape(&[tamano_lote * long_sec, carac_entrada]);

        // Multiplicación de matrices
        let salida_2d = entrada_2d.matmul(&self.peso);

        // Volvemos a la forma [lote, sec, caracteristicas_salida]
        let carac_salida = self.peso.forma[1];
        let salida_3d = salida_2d.reshape(&[tamano_lote, long_sec, carac_salida]);

        // Sumamos el sesgo (se transmite / broadcast automáticamente)
        salida_3d.add(&self.sesgo)
    }
}

//
// ============================================================================
// MECANISMO DE ATENCIÓN
// ============================================================================
//

/// Autoatención multicabezal (Multi-head self-attention)
///
/// El mecanismo central que permite al modelo enfocarse en diferentes partes de
/// la secuencia de entrada al procesar cada posición.
///
/// # Arquitectura
///
/// 1. **Proyecciones lineales**: Q, K, V a partir de la entrada
/// 2. **División en cabezales**: Cambio de forma a [lote, num_cabezales, sec, dim_cabezal]
/// 3. **Atención de producto punto escalado**: puntuaciones = Q @ K^T / √dim_cabezal
/// 4. **Máscara causal**: Evita atender a posiciones futuras
/// 5. **Softmax**: Convierte las puntuaciones en pesos de atención
/// 6. **Suma ponderada**: salida = atencion @ V
/// 7. **Concatenar cabezales** y proyectar de vuelta
///
/// # Atención Multicabezal
///
/// En lugar de una sola operación de atención, utilizamos múltiples "cabezales" en paralelo.
/// Cada cabezal puede aprender a prestar atención a diferentes aspectos de la secuencia.
///
/// Por ejemplo, con dim_embd=256 y num_cabezales=4:
/// - Cada cabezal tiene dim_cabezal = 256 / 4 = 64
/// - Calculamos 4 operaciones de atención independientes
/// - Concatenamos los resultados de vuelta a 256 dimensiones
///
/// # Enmascaramiento Causal
///
/// En el modelado de lenguaje, predecimos el siguiente token, por lo que la posición `i` no puede
/// ver las posiciones `i+1, i+2, ...` (el futuro). Imponemos esto estableciendo las
/// puntuaciones de atención para posiciones futuras a -infinito antes de aplicar softmax.
pub struct Atencion {
    /// Proyección combinada de Q, K, V: [dim_embd, 3 * dim_embd]
    pub c_atencion: Linear,
    /// Proyección de salida: [dim_embd, dim_embd]
    pub c_proy: Linear,
    /// Número de cabezales de atención
    pub num_cabezales: usize,
    /// Dimensión por cabezal (dim_embd / num_cabezales)
    pub dim_cabezal: usize,
}

impl Atencion {
    /// Crea una nueva capa de atención
    ///
    /// # Argumentos
    ///
    /// * `dim_embd` - Dimensión de las incrustaciones (embeddings)
    /// * `num_cabezales` - Número de cabezales de atención
    pub fn new(dim_embd: usize, num_cabezales: usize) -> Self {
        assert_eq!(dim_embd % num_cabezales, 0, "dim_embd debe ser divisible por num_cabezales");

        let dim_cabezal = dim_embd / num_cabezales;

        // Una sola capa lineal calcula Q, K, V de una vez
        let c_atencion = Linear::new(dim_embd, 3 * dim_embd);
        // Proyección de salida después de concatenar los cabezales
        let c_proy = Linear::new(dim_embd, dim_embd);

        Self {
            c_atencion,
            c_proy,
            num_cabezales,
            dim_cabezal,
        }
    }

    /// Paso hacia adelante: calcula la autoatención multicabezal
    ///
    /// # Argumentos
    ///
    /// * `entrada` - Tensor de entrada [lote, long_sec, dim_embd]
    ///
    /// # Retorna
    ///
    /// Tensor de salida [lote, long_sec, dim_embd] después de la atención
    pub fn forward(&self, entrada: &Tensor) -> Tensor {
        let tamano_lote = entrada.forma[0];
        let long_sec = entrada.forma[1];
        let dim_embd = entrada.forma[2];

        // === 1. Calcular Q, K, V ===
        // c_atencion proyecta a 3*dim_embd (Q, K, V apilados)
        let qkv = self.c_atencion.forward(entrada); // [lote, sec, 3*dim_embd]

        // Dividir en Q, K, V
        let mut datos_q = Vec::with_capacity(tamano_lote * long_sec * dim_embd);
        let mut datos_k = Vec::with_capacity(tamano_lote * long_sec * dim_embd);
        let mut datos_v = Vec::with_capacity(tamano_lote * long_sec * dim_embd);

        for i in 0..tamano_lote * long_sec {
            let inicio = i * 3 * dim_embd;
            datos_q.extend_from_slice(&qkv.datos[inicio..inicio + dim_embd]);
            datos_k.extend_from_slice(&qkv.datos[inicio + dim_embd..inicio + 2 * dim_embd]);
            datos_v.extend_from_slice(&qkv.datos[inicio + 2 * dim_embd..inicio + 3 * dim_embd]);
        }

        let q = Tensor::new(datos_q, vec![tamano_lote, long_sec, dim_embd]);
        let k = Tensor::new(datos_k, vec![tamano_lote, long_sec, dim_embd]);
        let v = Tensor::new(datos_v, vec![tamano_lote, long_sec, dim_embd]);

        // === 2. Cambiar forma para atención multicabezal ===
        // [lote, sec, dim_embd] -> [lote, num_cabezales, sec, dim_cabezal]
        let q = self.dividir_cabezales(&q, tamano_lote, long_sec);
        let k = self.dividir_cabezales(&k, tamano_lote, long_sec);
        let v = self.dividir_cabezales(&v, tamano_lote, long_sec);

        // === 3. Transponer K para puntuaciones de atención ===
        // K: [lote, num_cabezales, sec, dim_cabezal] -> [lote, num_cabezales, dim_cabezal, sec]
        let k_t = k.transpose(2, 3);

        // === 4. Calcular puntuaciones de atención ===
        // Q @ K^T: [lote, num_cabezales, sec, dim_cabezal] @ [lote, num_cabezales, dim_cabezal, sec]
        //       -> [lote, num_cabezales, sec, sec]
        let puntuaciones = q.matmul(&k_t);

        // === 5. Escalar por sqrt(dim_cabezal) ===
        let escala = 1.0 / (self.dim_cabezal as f32).sqrt();
        let puntuaciones = puntuaciones.mul_scalar(escala);

        // === 6. Aplicar máscara causal ===
        let mascara = self.crear_mascara_causal(long_sec);
        let puntuaciones = puntuaciones.masked_fill(&mascara, f32::NEG_INFINITY);

        // === 7. Softmax para obtener los pesos de atención ===
        let atencion_pesos = puntuaciones.softmax(-1); // [lote, num_cabezales, sec, sec]

        // === 8. Aplicar atención a los valores ===
        // atencion_pesos @ V: [lote, num_cabezales, sec, sec] @ [lote, num_cabezales, sec, dim_cabezal]
        //        -> [lote, num_cabezales, sec, dim_cabezal]
        let salida = atencion_pesos.matmul(&v);

        // === 9. Concatenar cabezales ===
        let salida = self.unir_cabezales(&salida, tamano_lote, long_sec);

        // === 10. Proyección de salida ===
        self.c_proy.forward(&salida)
    }

    /// Divide en múltiples cabezales de atención
    ///
    /// [lote, sec, dim_embd] -> [lote, num_cabezales, sec, dim_cabezal]
    fn dividir_cabezales(&self, entrada: &Tensor, tamano_lote: usize, long_sec: usize) -> Tensor {
        // Cambiar forma y transponer
        let mut resultado = vec![0.0; tamano_lote * self.num_cabezales * long_sec * self.dim_cabezal];

        for l in 0..tamano_lote {
            for s in 0..long_sec {
                for c in 0..self.num_cabezales {
                    for d in 0..self.dim_cabezal {
                        let indice_origen = (l * long_sec + s) * (self.num_cabezales * self.dim_cabezal)
                            + c * self.dim_cabezal
                            + d;
                        let indice_destino = ((l * self.num_cabezales + c) * long_sec + s) * self.dim_cabezal + d;
                        resultado[indice_destino] = entrada.datos[indice_origen];
                    }
                }
            }
        }

        Tensor::new(
            resultado,
            vec![tamano_lote, self.num_cabezales, long_sec, self.dim_cabezal],
        )
    }

    /// Vuelve a unir (fusionar) los cabezales de atención
    ///
    /// [lote, num_cabezales, sec, dim_cabezal] -> [lote, sec, dim_embd]
    fn unir_cabezales(&self, entrada: &Tensor, tamano_lote: usize, long_sec: usize) -> Tensor {
        let dim_embd = self.num_cabezales * self.dim_cabezal;
        let mut resultado = vec![0.0; tamano_lote * long_sec * dim_embd];

        for l in 0..tamano_lote {
            for s in 0..long_sec {
                for c in 0..self.num_cabezales {
                    for d in 0..self.dim_cabezal {
                        let indice_origen = ((l * self.num_cabezales + c) * long_sec + s) * self.dim_cabezal + d;
                        let indice_destino = (l * long_sec + s) * dim_embd + c * self.dim_cabezal + d;
                        resultado[indice_destino] = entrada.datos[indice_origen];
                    }
                }
            }
        }

        Tensor::new(resultado, vec![tamano_lote, long_sec, dim_embd])
    }

    /// Crea la máscara de atención causal
    ///
    /// Retorna una máscara de [long_sec, long_sec] donde:
    /// - 0 = puede atender (actual o pasado)
    /// - 1 = no puede atender (futuro)
    ///
    /// Para long_sec=4, la máscara se ve así:
    /// ```text
    /// [0 1 1 1]  la posición 0 solo puede verse a sí misma
    /// [0 0 1 1]  la posición 1 puede ver 0,1
    /// [0 0 0 1]  la posición 2 puede ver 0,1,2
    /// [0 0 0 0]  la posición 3 puede ver todas
    /// ```
    fn crear_mascara_causal(&self, long_sec: usize) -> Tensor {
        let mut datos_mascara = vec![0.0; long_sec * long_sec];

        for i in 0..long_sec {
            for j in 0..long_sec {
                if j > i {
                    // j está en el futuro relativo a i
                    datos_mascara[i * long_sec + j] = 1.0;
                }
            }
        }

        Tensor::new(datos_mascara, vec![long_sec, long_sec])
    }
}

//
// ============================================================================
// CAPA MLP (FEEDFORWARD)
// ============================================================================
//

/// Perceptrón multicapa (red prealimentada / feedforward network)
///
/// Se aplica después de la atención en cada bloque transformer. Consiste en:
/// 1. Capa lineal que expande a 4×dim_embd (dimensión oculta)
/// 2. Activación GELU
/// 3. Capa lineal que proyecta de vuelta a dim_embd
///
/// # ¿Por qué una expansión de 4×?
///
/// GPT-2 utiliza 4×dim_embd como la dimensión oculta en la capa feedforward.
/// Esto proporciona suficiente capacidad para que la red aprenda transformaciones complejas
/// mientras mantiene el flujo residual (dim_embd) más pequeño por eficiencia.
pub struct MLP {
    /// Primera capa lineal: [dim_embd, 4*dim_embd]
    pub c_fc: Linear,
    /// Segunda capa lineal: [4*dim_embd, dim_embd]
    pub c_proy: Linear,
}

impl MLP {
    /// Crea una nueva capa MLP
    ///
    /// # Argumentos
    ///
    /// * `dim_embd` - Dimensión de las incrustaciones (embeddings)
    pub fn new(dim_embd: usize) -> Self {
        let dim_oculta = 4 * dim_embd;
        let c_fc = Linear::new(dim_embd, dim_oculta);
        let c_proy = Linear::new(dim_oculta, dim_embd);

        Self { c_fc, c_proy }
    }

    /// Paso hacia adelante: expandir → GELU → proyectar
    ///
    /// # Argumentos
    ///
    /// * `entrada` - Tensor de entrada [lote, long_sec, dim_embd]
    ///
    /// # Retorna
    ///
    /// Tensor de salida [lote, long_sec, dim_embd]
    pub fn forward(&self, entrada: &Tensor) -> Tensor {
        // Expandir a la dimensión oculta
        let oculta = self.c_fc.forward(entrada);
        // Aplicar activación GELU
        let oculta = gelu(&oculta);
        // Proyectar de vuelta a dim_embd
        self.c_proy.forward(&oculta)
    }
}

//
// ============================================================================
// BLOQUE TRANSFORMER
// ============================================================================
//

/// Bloque transformer individual
///
/// Combina las capas de atención y feedforward con conexiones residuales
/// y normalización de capa.
///
/// # Arquitectura
///
/// ```text
/// Entrada
///   ↓
///   ├─→ LayerNorm → Atención ──→ + (residual)
///   ↓                            ↓
///   ├─→ LayerNorm → MLP ───────→ + (residual)
///   ↓
/// Salida
/// ```
///
/// # Conexiones Residuales
///
/// Cada subcapa (atención y MLP) tiene una conexión residual que
/// suma la entrada de vuelta a la salida. Esto ayuda al flujo del gradiente durante
/// el entrenamiento y permite construir redes muy profundas.
///
/// # Pre-Norm vs Post-Norm
///
/// GPT-2 utiliza "pre-norm": la normalización de capa se aplica antes de cada subcapa.
/// Esto es más estable que "post-norm" (después de cada subcapa) para redes profundas.
pub struct Block {
    /// Normalización de capa antes de la atención
    pub ln_1: LayerNorm,
    /// Atención multicabezal
    pub atencion: Atencion,
    /// Normalización de capa antes del MLP
    pub ln_2: LayerNorm,
    /// Red prealimentada (feedforward)
    pub mlp: MLP,
}

impl Block {
    /// Crea un nuevo bloque transformer
    ///
    /// # Argumentos
    ///
    /// * `dim_embd` - Dimensión de las incrustaciones (embeddings)
    /// * `num_cabezales` - Número de cabezales de atención
    pub fn new(dim_embd: usize, num_cabezales: usize) -> Self {
        Self {
            ln_1: LayerNorm::new(dim_embd, 1e-5),
            atencion: Atencion::new(dim_embd, num_cabezales),
            ln_2: LayerNorm::new(dim_embd, 1e-5),
            mlp: MLP::new(dim_embd),
        }
    }

    /// Paso hacia adelante a través del bloque transformer
    ///
    /// # Argumentos
    ///
    /// * `entrada` - Tensor de entrada [lote, long_sec, dim_embd]
    ///
    /// # Retorna
    ///
    /// Tensor de salida [lote, long_sec, dim_embd]
    pub fn forward(&self, entrada: &Tensor) -> Tensor {
        // Bloque de atención con conexión residual
        let entrada_actualizada = entrada.add(&self.atencion.forward(&self.ln_1.forward(entrada)));

        // Bloque MLP con conexión residual
        entrada_actualizada.add(&self.mlp.forward(&self.ln_2.forward(&entrada_actualizada)))
    }
}

//
// ============================================================================
// MODELO GPT-2
// ============================================================================
//

/// Modelo GPT-2 completo
///
/// Combina todos los componentes en un modelo de lenguaje completo que puede:
/// - Tomar IDs de tokens como entrada
/// - Producir logits sobre el vocabulario como salida
/// - Ser utilizado para la generación de texto (con un muestreo apropiado)
///
/// # Resumen de la Arquitectura
///
/// - Embedding de tokens + embedding de posición
/// - N bloques transformer (atención + MLP)
/// - Normalización de capa final
/// - Proyección lineal al vocabulario
///
/// # Solo Paso Hacia Adelante (Forward Pass)
///
/// Esta implementación proporciona solo el paso hacia adelante (inferencia).
pub struct GPT2 {
    /// Configuración del modelo
    pub configuracion: Config,
    /// Capa de embedding de tokens
    pub embedding_tokens: Embedding,
    /// Capa de embedding de posición
    pub embedding_posicion: Embedding,
    /// Pila de bloques transformer
    pub bloques: Vec<Block>,
    /// Normalización de capa final
    pub ln_final: LayerNorm,
    /// Proyección de salida al vocabulario (unembedding)
    pub cabeza_lm: Linear,
}

impl GPT2 {
    /// Crea un nuevo modelo GPT-2 con inicialización aleatoria
    ///
    /// Todos los pesos se inicializan desde N(0, 0.02) siguiendo el paper de GPT-2.
    /// Los parámetros de normalización de capa (gamma, beta) se inicializan en 1 y 0 respectivamente.
    /// Los sesgos (biases) se inicializan en ceros.
    ///
    /// # Argumentos
    ///
    /// * `configuracion` - Configuración del modelo
    ///
    /// # Retorna
    ///
    /// Modelo inicializado listo para los pasos hacia adelante (forward passes)
    pub fn new(configuracion: &Config) -> Self {
        // Crear embeddings
        let embedding_tokens = Embedding::new(configuracion.vocab_size, configuracion.n_embd);
        let embedding_posicion = Embedding::new(configuracion.block_size, configuracion.n_embd);

        // Crear bloques transformer
        let bloques = (0..configuracion.n_layers)
            .map(|_| Block::new(configuracion.n_embd, configuracion.n_heads))
            .collect();

        // Normalización de capa final
        let ln_final = LayerNorm::new(configuracion.n_embd, 1e-5);

        // Proyección de salida al vocabulario
        let cabeza_lm = Linear::new(configuracion.n_embd, configuracion.vocab_size);

        Self {
            configuracion: configuracion.clone(),
            embedding_tokens,
            embedding_posicion,
            bloques,
            ln_final,
            cabeza_lm,
        }
    }

    /// Paso hacia adelante: tokens → logits
    ///
    /// # Argumentos
    ///
    /// * `ids_tokens` - IDs de los tokens de entrada [tamano_lote][long_sec]
    ///
    /// # Retorna
    ///
    /// Logits sobre el vocabulario: [lote, sec, tamano_vocabulario]
    ///
    /// # Ejemplo
    ///
    /// ```rust,no_run
    /// # use molineteai::{GPT2, Config};
    /// let configuracion = Config::tiny(512);
    /// let modelo = GPT2::new(&configuracion);
    ///
    /// let tokens = vec![vec![1, 2, 3, 4]]; // tamano_lote=1, long_sec=4
    /// let logits = modelo.forward(&tokens);
    /// // logits.forma = [1, 4, 512]
    /// ```
    pub fn forward(&self, ids_tokens: &[Vec<usize>]) -> Tensor {
        let tamano_lote = ids_tokens.len();
        let long_sec = ids_tokens[0].len();

        assert!(
            long_sec <= self.configuracion.block_size,
            "La longitud de secuencia {} excede el tamano_bloque {}",
            long_sec,
            self.configuracion.block_size
        );

        // === 1. Embeddings de tokens ===
        let mut x = self.embedding_tokens.forward(ids_tokens);

        // === 2. Embeddings de posición ===
        // Crear índices de posición [0, 1, 2, ..., long_sec-1]
        let posiciones: Vec<Vec<usize>> = vec![(0..long_sec).collect()];
        let emb_pos = self.embedding_posicion.forward(&posiciones);

        // Transmitir (broadcast) los embeddings de posición al tamaño del lote y sumar
        // emb_pos: [1, long_sec, dim_embd] -> transmitir a [lote, long_sec, dim_embd]
        for l in 0..tamano_lote {
            for s in 0..long_sec {
                for e in 0..self.configuracion.n_embd {
                    let indice = (l * long_sec + s) * self.configuracion.n_embd + e;
                    let indice_pos = s * self.configuracion.n_embd + e;
                    x.datos[indice] += emb_pos.datos[indice_pos];
                }
            }
        }

        // === 3. Pasar a través de los bloques transformer ===
        for bloque in &self.bloques {
            x = bloque.forward(&x);
        }

        // === 4. Normalización de capa final ===
        x = self.ln_final.forward(&x);

        // === 5. Proyectar al vocabulario ===
        self.cabeza_lm.forward(&x)
    }

    /// Contar el número total de parámetros
    ///
    /// Útil para entender el tamaño del modelo y los requisitos de memoria
    ///
    /// # Retorna
    ///
    /// Número total de parámetros entrenables
    pub fn contar_parametros(&self) -> usize {
        let mut total = 0;

        // Embeddings de tokens y posición
        total += self.embedding_tokens.peso.datos.len();
        total += self.embedding_posicion.peso.datos.len();

        // Bloques Transformer
        for bloque in &self.bloques {
            // Atención
            total += bloque.atencion.c_atencion.peso.datos.len();
            total += bloque.atencion.c_atencion.sesgo.datos.len();
            total += bloque.atencion.c_proy.peso.datos.len();
            total += bloque.atencion.c_proy.sesgo.datos.len();

            // MLP
            total += bloque.mlp.c_fc.peso.datos.len();
            total += bloque.mlp.c_fc.sesgo.datos.len();
            total += bloque.mlp.c_proy.peso.datos.len();
            total += bloque.mlp.c_proy.sesgo.datos.len();

            // Normalizaciones de capa
            total += bloque.ln_1.gamma.datos.len();
            total += bloque.ln_1.beta.datos.len();
            total += bloque.ln_2.gamma.datos.len();
            total += bloque.ln_2.beta.datos.len();
        }

        // Normalización de capa final
        total += self.ln_final.gamma.datos.len();
        total += self.ln_final.beta.datos.len();

        // Cabeza LM
        total += self.cabeza_lm.peso.datos.len();
        total += self.cabeza_lm.sesgo.datos.len();

        total
    }
}