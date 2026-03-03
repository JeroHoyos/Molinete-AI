//! Carga de Datos para Entrenamiento
//!
//! Este módulo proporciona un cargador de datos simple para entrenar modelos
//! de lenguaje sobre texto. Maneja la tokenización, el batching y la generación
//! de secuencias usando un enfoque de ventana deslizante.
//!
//! ## Cómo se generan las secuencias
//!
//! El cargador de datos usa una ventana deslizante para crear ejemplos de entrenamiento:
//!
//! ```text
//! Tokens: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
//! Seq length: 4
//! Batch size: 2
//!
//! Batch 1:
//!   Input:  [1, 2, 3, 4]  Target: [2, 3, 4, 5]
//!   Input:  [5, 6, 7, 8]  Target: [6, 7, 8, 9]
//!
//! Batch 2:
//!   Input:  [9, 10, 11, 12]  Target: [10, 11, 12, 13]  (si 13 existe)
//! ```
//!
//! El target siempre es el input desplazado una posición,
//! enseñando al modelo a predecir el siguiente token.
//!
//! ## Ejemplo
//!
//! ```rust,no_run
//! # use feste::{BPETokenizer, TextDataLoader};
//! # let tokenizer = BPETokenizer::new(512);
//! let text = std::fs::read_to_string("shakespeare.txt")?;
//!
//! let mut loader = TextDataLoader::new(
//!     &text,
//!     &tokenizer,
//!     128,  // longitud de secuencia
//!     4     // tamaño de batch
//! );
//!
//! while let Some((inputs, targets)) = loader.next_batch() {
//!     // Entrenar con este batch
//!     // inputs: Vec<Vec<usize>> con forma [batch_size, seq_len]
//!     // targets: Vec<Vec<usize>> con forma [batch_size, seq_len]
//! }
//! # Ok::<(), std::io::Error>(())
//! ```

use crate::tokenizer::BPETokenizer;
use std::fs;

/// Alias de tipo para un batch de secuencias input/target
/// Cada elemento es un Vec<Vec<usize>> con forma [batch_size][seq_len]
pub type Batch = (Vec<Vec<usize>>, Vec<Vec<usize>>);

/// Cargador de datos para conjuntos de texto
///
/// Carga texto, lo tokeniza y proporciona batches de pares (input, target)
/// para entrenar modelos de lenguaje.
///
/// # Campos
///
/// - `tokens`: Todos los datos tokenizados
/// - `seq_len`: Longitud de cada secuencia de entrenamiento
/// - `batch_size`: Número de secuencias por batch
/// - `position`: Posición actual en el dataset
pub struct TextDataLoader {
    tokens: Vec<usize>,
    seq_len: usize,
    batch_size: usize,
    position: usize,
}

impl TextDataLoader {
    /// Crear un cargador de datos a partir de texto
    ///
    /// Tokeniza el texto inmediatamente y almacena todos los tokens en memoria.
    ///
    /// # Argumentos
    ///
    /// * `text` - Texto crudo para entrenar
    /// * `tokenizer` - Tokenizador entrenado para codificar
    /// * `seq_len` - Longitud de cada secuencia de entrenamiento
    /// * `batch_size` - Número de secuencias por batch
    pub fn new(text: &str, tokenizer: &BPETokenizer, seq_len: usize, batch_size: usize) -> Self {
        let tokens = tokenizer.encode(text);
        println!("Se cargaron {} tokens del texto", tokens.len());

        Self {
            tokens,
            seq_len,
            batch_size,
            position: 0,
        }
    }

    /// Crear un cargador de datos a partir de un archivo
    ///
    /// Método de conveniencia que lee el archivo y crea el loader.
    ///
    /// # Argumentos
    ///
    /// * `path` - Ruta al archivo de texto
    /// * `tokenizer` - Tokenizador entrenado
    /// * `seq_len` - Longitud de secuencia
    /// * `batch_size` - Tamaño de batch
    ///
    /// # Retorna
    ///
    /// Result que contiene el loader o un error de IO
    pub fn from_file(
        path: &str,
        tokenizer: &BPETokenizer,
        seq_len: usize,
        batch_size: usize,
    ) -> std::io::Result<Self> {
        let text = fs::read_to_string(path)?;
        Ok(Self::new(&text, tokenizer, seq_len, batch_size))
    }

    /// Obtener el siguiente batch de datos de entrenamiento
    ///
    /// Devuelve un batch de pares (input, target). El target siempre es
    /// el input desplazado una posición (predicción del siguiente token).
    ///
    /// Cuando se alcanza el final del dataset, retorna `None` y reinicia
    /// al inicio para la siguiente época.
    ///
    /// # Retorna
    ///
    /// - `Some((inputs, targets))` si hay un batch disponible
    /// - `None` si la época terminó (reinicia position)
    ///
    /// # Forma
    ///
    /// Tanto inputs como targets tienen forma `[batch_size, seq_len]`
    pub fn next_batch(&mut self) -> Option<Batch> {
        // Verificar si quedan suficientes tokens para un batch completo
        if self.position + self.batch_size * (self.seq_len + 1) >= self.tokens.len() {
            // Reiniciar al comienzo (época completa)
            self.position = 0;
            return None;
        }

        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        // Construir el batch extrayendo secuencias
        for _ in 0..self.batch_size {
            // Verificar que haya suficientes tokens para esta secuencia
            if self.position + self.seq_len + 1 >= self.tokens.len() {
                break;
            }

            // Extraer secuencia de entrada
            let input_seq =
                self.tokens[self.position..self.position + self.seq_len].to_vec();

            // Extraer secuencia objetivo (input desplazado en 1)
            let target_seq =
                self.tokens[self.position + 1..self.position + self.seq_len + 1].to_vec();

            inputs.push(input_seq);
            targets.push(target_seq);

            // Avanzar seq_len (secuencias no superpuestas)
            self.position += self.seq_len;
        }

        if inputs.is_empty() {
            None
        } else {
            Some((inputs, targets))
        }
    }

    /// Reiniciar el cargador de datos al inicio
    ///
    /// Útil para comenzar una nueva época sin esperar a que `next_batch()`
    /// llegue al final.
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Obtener el número total de batches en una época
    ///
    /// Es una estimación basada en el tamaño del dataset y los parámetros del batch.
    pub fn num_batches(&self) -> usize {
        self.tokens.len() / (self.batch_size * self.seq_len)
    }
}

/// Configuración de entrenamiento
///
/// Hiperparámetros para entrenar un modelo de lenguaje.
///
/// # Configuraciones comunes
///
/// - **Tiny**: Experimentación rápida (minutos)
/// - **Small**: Entrenamientos medianos (horas)
/// - **Large**: Entrenamiento completo (toda la noche)
pub struct TrainingConfig {
    /// Tasa de aprendizaje del optimizador
    pub learning_rate: f32,
    /// Número de pasadas sobre el dataset
    pub num_epochs: usize,
    /// Número de secuencias por batch
    pub batch_size: usize,
    /// Longitud de cada secuencia de entrenamiento
    pub seq_len: usize,
    /// Imprimir métricas cada N pasos
    pub print_every: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            num_epochs: 1,
            batch_size: 4,
            seq_len: 64,
            print_every: 100,
        }
    }
}

impl TrainingConfig {
    /// Crear una configuración tiny para experimentos rápidos
    ///
    /// Buena para:
    /// - Probar cambios en el código
    /// - Iteraciones rápidas
    /// - Entornos con pocos recursos
    pub fn tiny() -> Self {
        Self {
            learning_rate: 3e-4,
            num_epochs: 3,
            batch_size: 8,
            seq_len: 64,
            print_every: 50,
        }
    }

    /// Crear una configuración small para experimentos medianos
    ///
    /// Buena para:
    /// - Prototipar cambios en el modelo
    /// - Entrenamientos nocturnos
    /// - Balancear velocidad y calidad
    pub fn small() -> Self {
        Self {
            learning_rate: 3e-4,
            num_epochs: 5,
            batch_size: 16,
            seq_len: 128,
            print_every: 100,
        }
    }
}