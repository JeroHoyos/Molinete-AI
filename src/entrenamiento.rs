//! Carga de Datos de Entrenamiento
//!
//! Este módulo proporciona un cargador de datos simple para entrenar modelos de lenguaje en texto.
//! Maneja la tokenización, la creación de lotes (batching) y la generación de secuencias con un 
//! enfoque de ventana deslizante.
//!
//! ## Cómo se Generan las Secuencias
//!
//! El cargador de datos utiliza una ventana deslizante para crear ejemplos de entrenamiento:
//!
//! ```text
//! Tokens: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
//! Longitud de sec: 4
//! Tamaño de lote: 2
//!
//! Lote 1:
//!   Entrada:  [1, 2, 3, 4]  Objetivo: [2, 3, 4, 5]
//!   Entrada:  [5, 6, 7, 8]  Objetivo: [6, 7, 8, 9]
//!
//! Lote 2:
//!   Entrada:  [9, 10, 11, 12]  Objetivo: [10, 11, 12, 13]  (si el 13 existe)
//! ```
//!
//! El objetivo es siempre la entrada desplazada en una posición, lo que le enseña al modelo
//! a predecir el siguiente token.
//!
//! ## Ejemplo
//!
//! ```rust,no_run
//! # use molineteai::{TokenizadorBPE, CargadorDatosTexto};
//! # let tokenizador = TokenizadorBPE::new(512);
//! let texto = std::fs::read_to_string("shakespeare.txt")?;
//!
//! let mut cargador = CargadorDatosTexto::new(
//!     &texto,
//!     &tokenizador,
//!     128,  // longitud de secuencia (long_sec)
//!     4     // tamaño de lote (tamano_lote)
//! );
//!
//! while let Some((entradas, objetivos)) = cargador.siguiente_lote() {
//!     // Entrenar con este lote
//!     // entradas: Vec<Vec<usize>> con forma [tamano_lote, long_sec]
//!     // objetivos: Vec<Vec<usize>> con forma [tamano_lote, long_sec]
//! }
//! # Ok::<(), std::io::Error>(())
//! ```

use crate::tokenizador::TokenizadorBPE;
use std::fs;

/// Alias de tipo para un lote de secuencias de entrada/objetivo
/// Cada elemento es un Vec<Vec<usize>> con forma [tamano_lote][long_sec]
pub type Lote = (Vec<Vec<usize>>, Vec<Vec<usize>>);

/// Cargador de datos para conjuntos de datos de texto
///
/// Carga texto, lo tokeniza y proporciona lotes de pares de secuencias (entrada, objetivo)
/// para entrenar modelos de lenguaje.
///
/// # Campos
///
/// - `tokens`: Todos los datos tokenizados
/// - `long_sec`: Longitud de cada secuencia de entrenamiento
/// - `tamano_lote`: Número de secuencias por lote
/// - `posicion`: Posición actual en el conjunto de datos
pub struct CargadorDatosTexto {
    tokens: Vec<usize>,
    long_sec: usize,
    tamano_lote: usize,
    posicion: usize,
}

impl CargadorDatosTexto {
    /// Crea un cargador de datos desde texto
    ///
    /// Tokeniza el texto inmediatamente y almacena todos los tokens en memoria.
    ///
    /// # Argumentos
    ///
    /// * `texto` - Texto sin procesar para entrenar
    /// * `tokenizador` - Tokenizador entrenado para la codificación
    /// * `long_sec` - Longitud de cada secuencia de entrenamiento
    /// * `tamano_lote` - Número de secuencias por lote
    ///
    /// # Ejemplo
    ///
    /// ```rust,no_run
    /// # use molineteai::{TokenizadorBPE, CargadorDatosTexto};
    /// # let tokenizador = TokenizadorBPE::new(512);
    /// let texto = "Ser o no ser, esa es la cuestión.";
    /// let cargador = CargadorDatosTexto::new(&texto, &tokenizador, 64, 4);
    /// ```
    pub fn new(texto: &str, tokenizador: &TokenizadorBPE, long_sec: usize, tamano_lote: usize) -> Self {
        let tokens = tokenizador.codificar(texto);
        println!("Cargados {} tokens desde el texto", tokens.len());

        Self {
            tokens,
            long_sec,
            tamano_lote,
            posicion: 0,
        }
    }

    /// Crea un cargador de datos desde un archivo
    ///
    /// Método de conveniencia que lee el archivo y crea el cargador.
    ///
    /// # Argumentos
    ///
    /// * `ruta` - Ruta al archivo de texto
    /// * `tokenizador` - Tokenizador entrenado
    /// * `long_sec` - Longitud de secuencia
    /// * `tamano_lote` - Tamaño del lote
    ///
    /// # Retorna
    ///
    /// Resultado (Result) que contiene el cargador o un error de E/S (IO)
    ///
    /// # Ejemplo
    ///
    /// ```rust,no_run
    /// # use molineteai::{TokenizadorBPE, CargadorDatosTexto};
    /// # let tokenizador = TokenizadorBPE::new(512);
    /// let cargador = CargadorDatosTexto::de_archivo(
    ///     "shakespeare.txt",
    ///     &tokenizador,
    ///     128,
    ///     4
    /// )?;
    /// # Ok::<(), std::io::Error>(())
    /// ```
    pub fn de_archivo(
        ruta: &str,
        tokenizador: &TokenizadorBPE,
        long_sec: usize,
        tamano_lote: usize,
    ) -> std::io::Result<Self> {
        let texto = fs::read_to_string(ruta)?;
        Ok(Self::new(&texto, tokenizador, long_sec, tamano_lote))
    }

    /// Obtiene el siguiente lote de datos de entrenamiento
    ///
    /// Retorna un lote de pares de secuencias (entrada, objetivo). El objetivo es siempre
    /// la entrada desplazada en una posición (predicción del siguiente token).
    ///
    /// Cuando se alcanza el final del conjunto de datos, retorna `None` y se reinicia al
    /// principio para la siguiente época (epoch).
    ///
    /// # Retorna
    ///
    /// - `Some((entradas, objetivos))` si hay un lote disponible
    /// - `None` si la época está completa (reinicia la posición)
    ///
    /// # Forma
    ///
    /// Tanto las entradas como los objetivos tienen la forma `[tamano_lote, long_sec]`
    ///
    /// # Ejemplo
    ///
    /// ```rust,no_run
    /// # use molineteai::{TokenizadorBPE, CargadorDatosTexto};
    /// # let tokenizador = TokenizadorBPE::new(512);
    /// # let mut cargador = CargadorDatosTexto::new("texto", &tokenizador, 64, 4);
    /// while let Some((entradas, objetivos)) = cargador.siguiente_lote() {
    ///     assert_eq!(entradas.len(), 4);     // tamano_lote
    ///     assert_eq!(entradas[0].len(), 64); // long_sec
    ///     // Entrenar con este lote...
    /// }
    /// ```
    pub fn siguiente_lote(&mut self) -> Option<Lote> {
        // Comprobar si nos quedan suficientes tokens para un lote completo
        if self.posicion + self.tamano_lote * (self.long_sec + 1) >= self.tokens.len() {
            // Reiniciar al principio (época completa)
            self.posicion = 0;
            return None;
        }

        let mut entradas = Vec::new();
        let mut objetivos = Vec::new();

        // Construir el lote extrayendo secuencias
        for _ in 0..self.tamano_lote {
            // Asegurar que tenemos suficientes tokens para esta secuencia
            if self.posicion + self.long_sec + 1 >= self.tokens.len() {
                break;
            }

            // Extraer la secuencia de entrada: tokens[posicion..posicion+long_sec]
            let sec_entrada = self.tokens[self.posicion..self.posicion + self.long_sec].to_vec();

            // Extraer la secuencia objetivo: tokens[posicion+1..posicion+long_sec+1]
            // Esta es la entrada desplazada por 1 (predicción del siguiente token)
            let sec_objetivo =
                self.tokens[self.posicion + 1..self.posicion + self.long_sec + 1].to_vec();

            entradas.push(sec_entrada);
            objetivos.push(sec_objetivo);

            // Avanzar por long_sec (secuencias sin superposición)
            self.posicion += self.long_sec;
        }

        if entradas.is_empty() {
            None
        } else {
            Some((entradas, objetivos))
        }
    }

    /// Reinicia el cargador de datos al principio
    ///
    /// Útil para comenzar una nueva época sin esperar a que `siguiente_lote()`
    /// llegue al final.
    pub fn reiniciar(&mut self) {
        self.posicion = 0;
    }

    /// Obtiene el número total de lotes en una época
    ///
    /// Esta es una estimación basada en el tamaño del conjunto de datos y los parámetros del lote.
    ///
    /// # Retorna
    ///
    /// Número de lotes por época
    pub fn num_lotes(&self) -> usize {
        self.tokens.len() / (self.tamano_lote * self.long_sec)
    }
}

/// Configuración de entrenamiento
///
/// Hiperparámetros para entrenar un modelo de lenguaje.
///
/// # Configuraciones Comunes
///
/// - **Diminuta (Tiny)**: Experimentación rápida (minutos)
/// - **Pequeña (Small)**: Ejecuciones de entrenamiento medias (horas)
/// - **Grande (Large)**: Entrenamiento completo (toda la noche)
pub struct ConfigEntrenamiento {
    /// Tasa de aprendizaje para el optimizador
    pub tasa_aprendizaje: f32,
    /// Número de pasadas a través del conjunto de datos
    pub num_epocas: usize,
    /// Número de secuencias por lote
    pub tamano_lote: usize,
    /// Longitud de cada secuencia de entrenamiento
    pub long_sec: usize,
    /// Imprimir métricas cada N pasos
    pub imprimir_cada: usize,
}

impl Default for ConfigEntrenamiento {
    fn default() -> Self {
        Self {
            tasa_aprendizaje: 1e-3,
            num_epocas: 1,
            tamano_lote: 4,
            long_sec: 64,
            imprimir_cada: 100,
        }
    }
}

impl ConfigEntrenamiento {
    /// Crea una configuración diminuta para experimentos rápidos
    ///
    /// Útil para:
    /// - Probar cambios en el código
    /// - Iteraciones rápidas
    /// - Entornos de bajos recursos
    ///
    /// # Retorna
    ///
    /// ConfigEntrenamiento con un tamaño de lote pequeño y secuencias cortas
    pub fn diminuta() -> Self {
        Self {
            tasa_aprendizaje: 3e-4,
            num_epocas: 3,
            tamano_lote: 8,
            long_sec: 64,
            imprimir_cada: 50,
        }
    }

    /// Crea una configuración pequeña para experimentos medianos
    ///
    /// Útil para:
    /// - Prototipar cambios en el modelo
    /// - Ejecuciones de entrenamiento que duran toda la noche
    /// - Equilibrar velocidad y calidad
    ///
    /// # Retorna
    ///
    /// ConfigEntrenamiento con ajustes moderados
    pub fn pequena() -> Self {
        Self {
            tasa_aprendizaje: 3e-4,
            num_epocas: 5,
            tamano_lote: 16,
            long_sec: 128,
            imprimir_cada: 100,
        }
    }
}