//! Logger de Entrenamiento y Utilidades
//!
//! Este módulo proporciona utilidades para rastrear y registrar métricas de entrenamiento.
//! Incluye un logger en CSV para seguimiento detallado y funciones auxiliares para
//! calcular pérdidas y dividir conjuntos de datos.
//!
//! ## Componentes
//!
//! - **TrainingLogger**: Registra métricas en CSV y consola con marcas de tiempo
//! - **train_val_split**: Divide datos tokenizados en conjuntos de entrenamiento y validación
//! - **compute_dataset_loss**: Calcula la pérdida promedio sobre un conjunto de datos
//!
//! ## Ejemplo
//!
//! ```rust,no_run
//! use molineteai::TrainingLogger;
//!
//! let mut logger = TrainingLogger::new("training_log.csv")
//!     .expect("Error al crear el logger");
//!
//! // Registrar paso de entrenamiento
//! logger.log(100, 0.001, 2.5, 2.8, Some("Ser o no ser"))
//!     .expect("Error al registrar");
//! ```
//!
//! ## Formato CSV
//!
//! El logger escribe archivos CSV con las siguientes columnas:
//! - `step`: Número de paso de entrenamiento
//! - `elapsed_seconds`: Tiempo desde que comenzó el entrenamiento
//! - `learning_rate`: Tasa de aprendizaje actual
//! - `train_loss`: Pérdida de entrenamiento (entropía cruzada)
//! - `val_loss`: Pérdida de validación
//! - `train_perplexity`: exp(train_loss) - métrica interpretable
//! - `val_perplexity`: exp(val_loss) - menor es mejor
//! - `sample`: Texto de muestra generado
//!
//! ## Perplejidad
//!
//! La perplejidad mide qué tan "sorprendido" está el modelo por los datos:
//!
//! ```text
//! perplejidad = exp(pérdida)
//! ```
//!
//! - **Modelo perfecto**: perplejidad = 1.0 (pérdida = 0)
//! - **Adivinanza aleatoria** (vocab=512): perplejidad ≈ 512 (pérdida ≈ 6.2)
//! - **Buen modelo**: perplejidad = 10-50 (pérdida = 2.3-3.9)
//!
//! Menor perplejidad significa que el modelo hace mejores predicciones.

use std::fs::File;
use std::io::Write;
use std::time::Instant;

/// Logger de entrenamiento para rastrear métricas en el tiempo.
///
/// Registra métricas de entrenamiento tanto en archivo CSV como en consola.
/// El archivo CSV puede analizarse posteriormente para visualización y comparación de modelos.
///
/// # Campos
///
/// - `log_file`: Archivo CSV de salida
/// - `start_time`: Momento en que comenzó el entrenamiento
/// - `last_log_time`: Última marca de tiempo de registro
pub struct TrainingLogger {
    log_file: File,
    start_time: Instant,
    last_log_time: Instant,
}

impl TrainingLogger {
    /// Crear un nuevo logger de entrenamiento.
    ///
    /// Crea un archivo CSV con encabezados e inicializa los tiempos.
    ///
    /// # Argumentos
    ///
    /// * `log_path` - Ruta al archivo CSV a crear
    ///
    /// # Retorna
    ///
    /// Resultado que contiene el logger o un error de IO
    pub fn new(log_path: &str) -> std::io::Result<Self> {
        let mut log_file = File::create(log_path)?;

        // Escribir encabezado del CSV
        writeln!(
            log_file,
            "step,elapsed_seconds,learning_rate,train_loss,val_loss,train_perplexity,val_perplexity,sample"
        )?;

        let now = Instant::now();

        Ok(Self {
            log_file,
            start_time: now,
            last_log_time: now,
        })
    }

    /// Registrar un paso de entrenamiento.
    ///
    /// Escribe métricas en el CSV e imprime en consola con información de tiempo.
    ///
    /// # Argumentos
    ///
    /// * `step` - Número de paso de entrenamiento
    /// * `learning_rate` - Tasa de aprendizaje actual
    /// * `train_loss` - Pérdida de entrenamiento
    /// * `val_loss` - Pérdida de validación
    /// * `sample` - Texto de muestra generado (opcional)
    ///
    /// # Retorna
    ///
    /// Resultado indicando éxito o error de IO
    pub fn log(
        &mut self,
        step: usize,
        learning_rate: f32,
        train_loss: f32,
        val_loss: f32,
        sample: Option<&str>,
    ) -> std::io::Result<()> {
        let elapsed = self.start_time.elapsed().as_secs_f32();

        // Perplejidad = exp(pérdida)
        // Es una métrica más interpretable que la pérdida cruda
        let train_perplexity = train_loss.exp();
        let val_perplexity = val_loss.exp();

        // Escapar comillas en el texto de muestra para formato CSV
        let sample_escaped = sample
            .map(|s| s.replace('"', "\"\""))
            .unwrap_or_default();

        // Escribir en archivo CSV
        writeln!(
            self.log_file,
            "{},{:.2},{:.6},{:.4},{:.4},{:.2},{:.2},\"{}\"",
            step,
            elapsed,
            learning_rate,
            train_loss,
            val_loss,
            train_perplexity,
            val_perplexity,
            sample_escaped
        )?;

        // Forzar escritura inmediata para no perder datos si el entrenamiento falla
        self.log_file.flush()?;

        // Imprimir en consola con información de tiempo
        let step_time = self.last_log_time.elapsed().as_secs_f32();

        println!(
            "Paso {:4} | Tiempo: {:7.1}s (+{:.1}s) | LR: {:.6} | Train: {:.4} | Val: {:.4} | Perplejidad: {:.2}",
            step, elapsed, step_time, learning_rate, train_loss, val_loss, val_perplexity
        );

        if let Some(text) = sample {
            println!("  Muestra: \"{}\"", text);
        }

        self.last_log_time = Instant::now();
        Ok(())
    }
}

/// Divide datos tokenizados en conjuntos de entrenamiento y validación.
///
/// Realiza una división simple usando una fracción fija.
/// El conjunto de validación se toma del final de los datos para asegurar
/// separación temporal en datos secuenciales.
///
/// # Argumentos
///
/// * `tokens` - Datos tokenizados completos
/// * `val_fraction` - Fracción para validación (por ejemplo, 0.1 para 10%)
///
/// # Retorna
///
/// Tupla (tokens_entrenamiento, tokens_validación)
pub fn train_val_split(tokens: &[usize], val_fraction: f32) -> (&[usize], &[usize]) {
    let split_idx = ((tokens.len() as f32) * (1.0 - val_fraction)) as usize;
    (&tokens[..split_idx], &tokens[split_idx..])
}

/// Calcula la pérdida promedio sobre un conjunto de datos.
///
/// Evalúa el modelo en múltiples batches del conjunto de datos y devuelve
/// la pérdida promedio. Esto se usa para calcular la pérdida de validación
/// durante el entrenamiento.
///
/// # Argumentos
///
/// * `tokens` - Conjunto de datos tokenizado
/// * `seq_len` - Longitud de secuencia por ejemplo
/// * `num_batches` - Número de batches a evaluar (limitado por el tamaño del dataset)
/// * `compute_loss_fn` - Función que calcula la pérdida para un batch
///
/// # Retorna
///
/// Pérdida promedio en todos los batches
pub fn compute_dataset_loss<F>(
    tokens: &[usize],
    seq_len: usize,
    num_batches: usize,
    mut compute_loss_fn: F,
) -> f32
where
    F: FnMut(&[usize], &[usize]) -> f32,
{
    // Se necesitan al menos seq_len + 1 tokens (entrada + objetivo)
    if tokens.len() < seq_len + 1 {
        return 0.0;
    }

    let mut total_loss = 0.0;

    // Limitar num_batches a lo realmente disponible en el dataset
    let max_batches = (tokens.len() - seq_len - 1) / seq_len;
    let num_batches = num_batches.min(max_batches);

    for batch_idx in 0..num_batches {
        // Extraer secuencias de entrada y objetivo
        // El objetivo está desplazado 1 posición (predicción del siguiente token)
        let start = (batch_idx * seq_len) % (tokens.len() - seq_len - 1);
        let input_seq = &tokens[start..start + seq_len];
        let target_seq = &tokens[start + 1..start + seq_len + 1];

        let loss = compute_loss_fn(input_seq, target_seq);
        total_loss += loss;
    }

    total_loss / num_batches as f32
}