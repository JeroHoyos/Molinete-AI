//! Registrador de Entrenamiento y Utilidades
//!
//! Este módulo proporciona utilidades para rastrear y registrar métricas de entrenamiento.
//! Incluye un registrador CSV para un seguimiento detallado y funciones auxiliares para
//! calcular las pérdidas y dividir los conjuntos de datos.
//!
//! ## Componentes
//!
//! - **RegistradorEntrenamiento**: Registra métricas en CSV y consola con marcas de tiempo
//! - **dividir_entrenamiento_validacion**: Divide los datos tokenizados en conjuntos de entrenamiento y validación
//! - **calcular_perdida_dataset**: Calcula la pérdida promedio en un conjunto de datos
//!
//! ## Ejemplo
//!
//! ```rust,no_run
//! use molineteai::RegistradorEntrenamiento;
//!
//! let mut registrador = RegistradorEntrenamiento::new("registro_entrenamiento.csv")
//!     .expect("Fallo al crear el registrador");
//!
//! // Registrar paso de entrenamiento
//! registrador.log(100, 0.001, 2.5, 2.8, Some("Ser o no ser"))
//!     .expect("Fallo al registrar");
//! ```
//!
//! ## Formato CSV
//!
//! El registrador escribe archivos CSV con las siguientes columnas:
//! - `paso`: Número del paso de entrenamiento
//! - `segundos_transcurridos`: Tiempo desde que inició el entrenamiento
//! - `tasa_aprendizaje`: Tasa de aprendizaje actual
//! - `perdida_entrenamiento`: Pérdida de entrenamiento (entropía cruzada)
//! - `perdida_validacion`: Pérdida de validación
//! - `perplejidad_entrenamiento`: exp(perdida_entrenamiento) - métrica interpretable
//! - `perplejidad_validacion`: exp(perdida_validacion) - menor es mejor
//! - `muestra`: Muestra de texto generado
//!
//! ## Perplejidad
//!
//! La perplejidad mide qué tan "sorprendido" está el modelo por los datos:
//! ```text
//! perplejidad = exp(pérdida)
//! ```
//!
//! - **Modelo perfecto**: perplejidad = 1.0 (pérdida = 0)
//! - **Adivinanza aleatoria** (vocab=512): perplejidad ≈ 512 (pérdida ≈ 6.2)
//! - **Buen modelo**: perplejidad = 10-50 (pérdida = 2.3-3.9)
//!
//! Una perplejidad más baja significa que el modelo hace mejores predicciones.

use std::fs::File;
use std::io::Write;
use std::time::Instant;

/// Registrador de entrenamiento para rastrear métricas a lo largo del tiempo
///
/// Registra métricas de entrenamiento tanto en un archivo CSV como en la consola. El archivo CSV
/// puede ser analizado más tarde para visualización y comparación de modelos.
///
/// # Campos
///
/// - `archivo_registro`: Archivo CSV de salida
/// - `tiempo_inicio`: Cuándo comenzó el entrenamiento (para el cálculo del tiempo transcurrido)
/// - `tiempo_ultimo_registro`: Marca de tiempo del último registro (para el tiempo del paso)
pub struct RegistradorEntrenamiento {
    archivo_registro: File,
    tiempo_inicio: Instant,
    tiempo_ultimo_registro: Instant,
}

impl RegistradorEntrenamiento {
    /// Crea un nuevo registrador de entrenamiento
    ///
    /// Crea un archivo CSV con encabezados e inicializa la temporización.
    ///
    /// # Argumentos
    ///
    /// * `ruta_registro` - Ruta al archivo CSV a crear
    ///
    /// # Retorna
    ///
    /// Resultado (Result) que contiene el registrador o un error de E/S
    ///
    /// # Ejemplo
    ///
    /// ```rust,no_run
    /// # use molineteai::RegistradorEntrenamiento;
    /// let registrador = RegistradorEntrenamiento::new("registro_entrenamiento.csv")?;
    /// # Ok::<(), std::io::Error>(())
    /// ```
    pub fn new(ruta_registro: &str) -> std::io::Result<Self> {
        let mut archivo_registro = File::create(ruta_registro)?;

        // Escribir encabezado CSV
        writeln!(
            archivo_registro,
            "paso,segundos_transcurridos,tasa_aprendizaje,perdida_entrenamiento,perdida_validacion,perplejidad_entrenamiento,perplejidad_validacion,muestra"
        )?;

        let ahora = Instant::now();
        Ok(Self {
            archivo_registro,
            tiempo_inicio: ahora,
            tiempo_ultimo_registro: ahora,
        })
    }

    /// Registra un paso de entrenamiento
    ///
    /// Escribe las métricas en el CSV e imprime en la consola con información de temporización.
    ///
    /// # Argumentos
    ///
    /// * `paso` - Número del paso de entrenamiento
    /// * `tasa_aprendizaje` - Tasa de aprendizaje actual
    /// * `perdida_entrenamiento` - Pérdida de entrenamiento
    /// * `perdida_validacion` - Pérdida de validación
    /// * `muestra` - Muestra de texto generado opcional
    ///
    /// # Retorna
    ///
    /// Resultado que indica éxito o un error de E/S
    ///
    /// # Ejemplo
    ///
    /// ```rust,no_run
    /// # use molineteai::RegistradorEntrenamiento;
    /// # let mut registrador = RegistradorEntrenamiento::new("registro.csv")?;
    /// registrador.log(100, 0.001, 2.5, 2.8, Some("Hola mundo"))?;
    /// # Ok::<(), std::io::Error>(())
    /// ```
    pub fn log(
        &mut self,
        paso: usize,
        tasa_aprendizaje: f32,
        perdida_entrenamiento: f32,
        perdida_validacion: f32,
        muestra: Option<&str>,
    ) -> std::io::Result<()> {
        let transcurrido = self.tiempo_inicio.elapsed().as_secs_f32();

        // Perplejidad = exp(pérdida)
        // Esta es una métrica más interpretable que la pérdida cruda
        let perplejidad_entrenamiento = perdida_entrenamiento.exp();
        let perplejidad_validacion = perdida_validacion.exp();

        // Escapar comillas en el texto de muestra para el formato CSV
        let muestra_escapada = muestra.map(|s| s.replace('"', "\"\"")).unwrap_or_default();

        // Escribir en el archivo CSV
        writeln!(
            self.archivo_registro,
            "{},{:.2},{:.6},{:.4},{:.4},{:.2},{:.2},\"{}\"",
            paso,
            transcurrido,
            tasa_aprendizaje,
            perdida_entrenamiento,
            perdida_validacion,
            perplejidad_entrenamiento,
            perplejidad_validacion,
            muestra_escapada
        )?;

        // Vaciar el búfer (flush) para asegurar que los datos se escriban inmediatamente
        // Esto es importante si el entrenamiento colapsa - así no perdemos datos
        self.archivo_registro.flush()?;

        // Imprimir en consola con información de tiempos
        let tiempo_paso = self.tiempo_ultimo_registro.elapsed().as_secs_f32();
        println!(
            "Paso {:4} | Tiempo: {:7.1}s (+{:.1}s) | TA: {:.6} | Entren: {:.4} | Val: {:.4} | Perplejidad: {:.2}",
            paso, transcurrido, tiempo_paso, tasa_aprendizaje, perdida_entrenamiento, perdida_validacion, perplejidad_validacion
        );

        if let Some(texto) = muestra {
            println!("  Muestra: \"{}\"", texto);
        }

        self.tiempo_ultimo_registro = Instant::now();
        Ok(())
    }
}

/// Divide los datos tokenizados en conjuntos de entrenamiento y validación
///
/// Realiza una división simple en una fracción fija. El conjunto de validación se toma
/// del final de los datos para asegurar la separación temporal en datos secuenciales.
///
/// # Argumentos
///
/// * `tokens` - Todos los datos tokenizados
/// * `fraccion_val` - Fracción a usar para validación (ej., 0.1 para 10%)
///
/// # Retorna
///
/// Tupla de (tokens_entrenamiento, tokens_validacion)
///
/// # Ejemplo
///
/// ```rust
/// # use molineteai::dividir_entrenamiento_validacion;
/// let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
/// let (entren, val) = dividir_entrenamiento_validacion(&tokens, 0.2);
/// assert_eq!(entren.len(), 8);  // 80% para entrenamiento
/// assert_eq!(val.len(), 2);    // 20% para validación
/// ```
pub fn dividir_entrenamiento_validacion(tokens: &[usize], fraccion_val: f32) -> (&[usize], &[usize]) {
    let indice_division = ((tokens.len() as f32) * (1.0 - fraccion_val)) as usize;
    (&tokens[..indice_division], &tokens[indice_division..])
}

/// Calcula la pérdida promedio sobre un conjunto de datos
///
/// Evalúa el modelo en múltiples lotes (batches) del conjunto de datos y retorna
/// la pérdida promedio. Esto se usa para calcular la pérdida de validación durante el entrenamiento.
///
/// # Argumentos
///
/// * `tokens` - Conjunto de datos tokenizado
/// * `long_sec` - Longitud de secuencia por ejemplo
/// * `num_lotes` - Número de lotes a evaluar (limitado por el tamaño del conjunto de datos)
/// * `fn_calcular_perdida` - Función que calcula la pérdida para un solo lote
///
/// # Retorna
///
/// Pérdida promedio a través de todos los lotes
///
/// # Ejemplo
///
/// ```rust,no_run
/// # use molineteai::calcular_perdida_dataset;
/// let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
/// let perdida_promedio = calcular_perdida_dataset(
///     &tokens,
///     4,    // long_sec
///     2,    // num_lotes
///     |entrada, objetivo| {
///         // Calcular pérdida para este lote
///         0.5  // marcador de posición
///     }
/// );
/// ```
pub fn calcular_perdida_dataset<F>(
    tokens: &[usize],
    long_sec: usize,
    num_lotes: usize,
    mut fn_calcular_perdida: F,
) -> f32
where
    F: FnMut(&[usize], &[usize]) -> f32,
{
    // Necesita al menos long_sec + 1 tokens (entrada + objetivo)
    if tokens.len() < long_sec + 1 {
        return 0.0;
    }

    let mut perdida_total = 0.0;

    // Limitar num_lotes a lo que realmente está disponible en el conjunto de datos
    let max_lotes = (tokens.len() - long_sec - 1) / long_sec;
    let num_lotes = num_lotes.min(max_lotes);

    for indice_lote in 0..num_lotes {
        // Extraer secuencias de entrada y objetivo
        // El objetivo está desplazado 1 posición (predicción del siguiente token)
        let inicio = (indice_lote * long_sec) % (tokens.len() - long_sec - 1);
        let sec_entrada = &tokens[inicio..inicio + long_sec];
        let sec_objetivo = &tokens[inicio + 1..inicio + long_sec + 1];

        let perdida = fn_calcular_perdida(sec_entrada, sec_objetivo);
        perdida_total += perdida;
    }

    perdida_total / num_lotes as f32
}