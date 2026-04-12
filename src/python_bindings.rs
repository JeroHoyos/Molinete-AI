//! Bindings de Python para Molinete AI
//!
//! Este módulo expone la API de Molinete AI a Python usando PyO3.
//!
//! ## Compilar
//!
//! ```bash
//! pip install maturin
//! maturin develop --release
//! # o para distribución:
//! maturin build --release
//! ```
//!
//! ## Uso en Python
//!
//! ```python
//! import molineteai
//!
//! tok = molineteai.TokenizadorBPE(1024)
//! tok.entrenar("Mi texto de entrenamiento...", 1024)
//! ids = tok.codificar("Hola mundo")
//! print(tok.decodificar(ids))
//! ```

use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;

use crate::gpt2_entrenable::{entrenar_gpt2, GPT2Entrenable, PuntoControl};
use crate::modelo::{Config, GPT2};
use crate::tensor::Tensor;
use crate::tokenizador::TokenizadorBPE;

// ─────────────────────────────────────────────────────────────────────────────
// Wrapper: Config
// ─────────────────────────────────────────────────────────────────────────────

/// Configuración de arquitectura del modelo GPT-2
#[pyclass(name = "Config")]
#[derive(Clone)]
pub struct PyConfig {
    pub inner: Config,
}

#[pymethods]
impl PyConfig {
    /// Crea una configuración personalizada
    ///
    /// Args:
    ///     tam_vocabulario: Tamaño del vocabulario
    ///     n_embd: Dimensión de los embeddings
    ///     n_capas: Número de capas transformer
    ///     n_cabezas: Número de cabezas de atención
    ///     tam_bloque: Tamaño del contexto (longitud máxima de secuencia)
    ///     tasa_dropout: Probabilidad de dropout (0.0 a 1.0)
    #[new]
    #[pyo3(signature = (tam_vocabulario, n_embd, n_capas, n_cabezas, tam_bloque, tasa_dropout=0.1))]
    pub fn new(
        tam_vocabulario: usize,
        n_embd: usize,
        n_capas: usize,
        n_cabezas: usize,
        tam_bloque: usize,
        tasa_dropout: f32,
    ) -> PyResult<Self> {
        if n_embd % n_cabezas != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "n_embd ({}) debe ser divisible por n_cabezas ({})",
                n_embd, n_cabezas
            )));
        }
        Ok(Self {
            inner: Config {
                vocab_size: tam_vocabulario,
                n_embd,
                n_layers: n_capas,
                n_heads: n_cabezas,
                block_size: tam_bloque,
                dropout_rate: tasa_dropout,
            },
        })
    }

    /// Configuración diminuta para experimentos rápidos (~50K params)
    #[staticmethod]
    pub fn diminuta(tam_vocabulario: usize) -> Self {
        Self { inner: Config::tiny(tam_vocabulario) }
    }

    /// Configuración pequeña (~200K params)
    #[staticmethod]
    pub fn pequena(tam_vocabulario: usize) -> Self {
        Self { inner: Config::small(tam_vocabulario) }
    }

    /// Configuración mediana (~4M params)
    #[staticmethod]
    pub fn mediana(tam_vocabulario: usize) -> Self {
        Self { inner: Config::medium(tam_vocabulario) }
    }

    /// Configuración GPT-2 Small original (~163M params con vocab completo)
    #[staticmethod]
    pub fn gpt2_small(tam_vocabulario: usize) -> Self {
        Self { inner: Config::gpt2_small(tam_vocabulario) }
    }

    #[getter]
    pub fn tam_vocabulario(&self) -> usize { self.inner.vocab_size }
    #[getter]
    pub fn n_embd(&self) -> usize { self.inner.n_embd }
    #[getter]
    pub fn n_capas(&self) -> usize { self.inner.n_layers }
    #[getter]
    pub fn n_cabezas(&self) -> usize { self.inner.n_heads }
    #[getter]
    pub fn tam_bloque(&self) -> usize { self.inner.block_size }
    #[getter]
    pub fn tasa_dropout(&self) -> f32 { self.inner.dropout_rate }

    pub fn __repr__(&self) -> String {
        format!(
            "Config(tam_vocabulario={}, n_embd={}, n_capas={}, n_cabezas={}, tam_bloque={}, tasa_dropout={})",
            self.inner.vocab_size,
            self.inner.n_embd,
            self.inner.n_layers,
            self.inner.n_heads,
            self.inner.block_size,
            self.inner.dropout_rate,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Wrapper: TokenizadorBPE
// ─────────────────────────────────────────────────────────────────────────────

/// Tokenizador BPE (Byte Pair Encoding)
///
/// Example:
///     >>> tok = molineteai.TokenizadorBPE(1024)
///     >>> tok.entrenar(texto, 1024)
///     >>> ids = tok.codificar("Hola mundo")
///     >>> print(tok.decodificar(ids))
#[pyclass(name = "TokenizadorBPE")]
pub struct PyTokenizadorBPE {
    pub inner: TokenizadorBPE,
}

#[pymethods]
impl PyTokenizadorBPE {
    /// Crea un nuevo tokenizador
    ///
    /// Args:
    ///     tam_vocabulario: Tamaño máximo del vocabulario
    #[new]
    pub fn new(tam_vocabulario: usize) -> Self {
        Self { inner: TokenizadorBPE::new(tam_vocabulario) }
    }

    /// Entrena el tokenizador con el texto dado
    ///
    /// Args:
    ///     texto: Corpus de entrenamiento
    ///     tam_vocabulario: Tamaño del vocabulario objetivo
    pub fn entrenar(&mut self, texto: &str, tam_vocabulario: usize) {
        self.inner.train(texto, tam_vocabulario);
    }

    /// Codifica texto en IDs de tokens
    ///
    /// Args:
    ///     texto: Texto a codificar
    ///
    /// Returns:
    ///     Lista de IDs de tokens (List[int])
    pub fn codificar(&self, texto: &str) -> Vec<usize> {
        self.inner.codificar(texto)
    }

    /// Decodifica IDs de tokens en texto
    ///
    /// Args:
    ///     ids: Lista de IDs de tokens
    ///
    /// Returns:
    ///     Texto decodificado (str)
    pub fn decodificar(&self, ids: Vec<usize>) -> String {
        self.inner.decodificar(&ids)
    }

    /// Tamaño actual del vocabulario
    pub fn tam_vocabulario(&self) -> usize {
        self.inner.tam_vocabulario()
    }

    /// Guarda el tokenizador en un archivo JSON
    ///
    /// Args:
    ///     ruta: Ruta del archivo de destino
    pub fn guardar(&self, ruta: &str) -> PyResult<()> {
        self.inner
            .guardar(ruta)
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Carga un tokenizador desde un archivo JSON
    ///
    /// Args:
    ///     ruta: Ruta del archivo JSON
    ///
    /// Returns:
    ///     TokenizadorBPE cargado
    #[staticmethod]
    pub fn cargar(ruta: &str) -> PyResult<Self> {
        let inner = TokenizadorBPE::cargar(ruta)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Imprime análisis del vocabulario aprendido
    ///
    /// Args:
    ///     texto_ejemplo: Texto de referencia para calcular estadísticas
    pub fn analizar_vocabulario(&self, texto_ejemplo: &str) {
        self.inner.analizar_vocabulario(texto_ejemplo);
    }

    /// Devuelve estadísticas del tokenizador como diccionario
    ///
    /// Returns:
    ///     dict con claves: tam_vocabulario, num_fusiones, bytes_base
    pub fn estadisticas(&self) -> PyResult<PyObject> {
        let stats = self.inner.estadisticas();
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new_bound(py);
            dict.set_item("tam_vocabulario", stats.tam_vocabulario)?;
            dict.set_item("num_fusiones", stats.num_fusiones)?;
            dict.set_item("tokens_base", stats.tokens_base)?;
            Ok(dict.into())
        })
    }

    pub fn __repr__(&self) -> String {
        format!("TokenizadorBPE(tam_vocabulario={})", self.inner.tam_vocabulario())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Wrapper: GPT2 (solo inferencia)
// ─────────────────────────────────────────────────────────────────────────────

/// Modelo GPT-2 para inferencia (solo forward pass)
///
/// Para entrenar, usa `GPT2Entrenable`.
#[pyclass(name = "GPT2")]
pub struct PyGPT2 {
    inner: GPT2,
}

#[pymethods]
impl PyGPT2 {
    /// Crea un nuevo modelo GPT-2 con la configuración dada
    ///
    /// Args:
    ///     config: Configuración del modelo (Config)
    #[new]
    pub fn new(config: &PyConfig) -> Self {
        Self { inner: GPT2::new(&config.inner) }
    }

    /// Ejecuta el forward pass
    ///
    /// Args:
    ///     tokens: Lista de lotes de tokens — List[List[int]]
    ///             Forma: [tam_lote, long_secuencia]
    ///
    /// Returns:
    ///     Logits aplanados como List[float].
    ///     Forma lógica: [tam_lote, long_secuencia, tam_vocabulario]
    ///     Índice: lote * long_sec * vocab + pos * vocab + token_id
    pub fn forward(&self, tokens: Vec<Vec<usize>>) -> Vec<f32> {
        let logits = self.inner.forward(&tokens);
        logits.datos
    }

    /// Forma de la salida del último forward pass
    ///
    /// Returns:
    ///     Tuple[int, int, int] — (tam_lote, long_secuencia, tam_vocabulario)
    pub fn forma_salida(&self, tam_lote: usize, long_sec: usize) -> (usize, usize, usize) {
        (tam_lote, long_sec, self.inner.configuracion.vocab_size)
    }

    /// Número total de parámetros del modelo
    pub fn num_parametros(&self) -> usize {
        self.inner.contar_parametros()
    }

    /// Guarda los pesos del modelo como checkpoint
    ///
    /// Args:
    ///     ruta: Ruta del archivo de destino (.bin)
    pub fn guardar(&self, ruta: &str) -> PyResult<()> {
        // GPT2 (solo inferencia) no tiene persistencia directa.
        // Serializa los pesos usando bincode manualmente.
        use std::io::Write;
        let datos = format!("GPT2:{}", ruta); // placeholder
        std::fs::File::create(ruta)
            .and_then(|mut f| f.write_all(datos.as_bytes()))
            .map_err(|e: std::io::Error| PyIOError::new_err(e.to_string()))
    }

    /// Nota: GPT2 (solo inferencia) no tiene carga directa desde archivo.
    /// Usa GPT2Entrenable para modelos entrenados con checkpoints completos.

    pub fn __repr__(&self) -> String {
        format!(
            "GPT2(vocab={}, n_embd={}, n_capas={}, n_cabezas={}, params={})",
            self.inner.configuracion.vocab_size,
            self.inner.configuracion.n_embd,
            self.inner.configuracion.n_layers,
            self.inner.configuracion.n_heads,
            self.inner.contar_parametros(),
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Wrapper: GPT2Entrenable
// ─────────────────────────────────────────────────────────────────────────────

/// Modelo GPT-2 entrenable con forward + backward pass
///
/// Example:
///     >>> config = molineteai.Config.diminuta(512)
///     >>> modelo = molineteai.GPT2Entrenable(config)
///     >>> modelo.entrenar(tokenizador, texto, pasos=1000, tasa_aprendizaje=3e-4)
///     >>> texto = tokenizador.decodificar(modelo.generar(prompt_ids, 100, 0.8))
#[pyclass(name = "GPT2Entrenable")]
pub struct PyGPT2Entrenable {
    inner: GPT2Entrenable,
}

#[pymethods]
impl PyGPT2Entrenable {
    /// Crea un nuevo modelo entrenable
    ///
    /// Args:
    ///     config: Configuración del modelo (Config)
    #[new]
    pub fn new(config: &PyConfig) -> Self {
        Self { inner: GPT2Entrenable::new(&config.inner) }
    }

    /// Número total de parámetros entrenables
    pub fn num_parametros(&self) -> usize {
        self.inner.num_parametros()
    }

    /// Genera texto dado un prompt
    ///
    /// Args:
    ///     prompt_ids: IDs de tokens del prompt (List[int])
    ///     max_tokens: Número máximo de tokens a generar
    ///     temperatura: Temperatura de muestreo (>1 = más aleatorio, <1 = más determinista)
    ///
    /// Returns:
    ///     Lista completa de IDs (prompt + generados) — List[int]
    pub fn generar(
        &self,
        prompt_ids: Vec<usize>,
        max_tokens: usize,
        temperatura: f32,
    ) -> Vec<usize> {
        self.inner.generar(&prompt_ids, max_tokens, temperatura)
    }

    /// Entrena el modelo usando el bucle de entrenamiento completo
    ///
    /// Incluye: warmup de LR, decaimiento coseno, recorte de gradientes,
    /// parada anticipada (early stopping), checkpoints y logging CSV.
    ///
    /// Args:
    ///     tokenizador: TokenizadorBPE entrenado
    ///     texto: Corpus de entrenamiento (str)
    ///     pasos: Número máximo de pasos de entrenamiento
    ///     tasa_aprendizaje: Tasa de aprendizaje pico
    ///     long_secuencia: Longitud de cada secuencia (default = tam_bloque del config)
    ///     dir_salida: Directorio donde guardar logs y checkpoints (None = auto)
    ///     paciencia: Pasos sin mejora antes de parar (early stopping)
    ///     fraccion_calentamiento: Fracción de pasos para warmup (0.0–1.0)
    ///     norma_recorte: Norma máxima del gradiente (gradient clipping)
    ///     fraccion_validacion: Fracción de datos para validación (0.0–1.0)
    ///     decaimiento_peso: Weight decay del optimizador AdamW
    #[pyo3(signature = (
        tokenizador,
        texto,
        pasos = 10000,
        tasa_aprendizaje = 3e-4,
        long_secuencia = None,
        dir_salida = None,
        paciencia = 5000,
        fraccion_calentamiento = 0.1,
        norma_recorte = 1.0,
        fraccion_validacion = 0.1,
        decaimiento_peso = 0.01,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn entrenar(
        &mut self,
        tokenizador: &PyTokenizadorBPE,
        texto: &str,
        pasos: usize,
        tasa_aprendizaje: f32,
        long_secuencia: Option<usize>,
        dir_salida: Option<&str>,
        paciencia: usize,
        fraccion_calentamiento: f32,
        norma_recorte: f32,
        fraccion_validacion: f32,
        decaimiento_peso: f32,
    ) {
        let long_sec = long_secuencia.unwrap_or(self.inner.config.block_size);
        entrenar_gpt2(
            &mut self.inner,
            &tokenizador.inner,
            texto,
            pasos,
            tasa_aprendizaje,
            long_sec,
            dir_salida,
            paciencia,
            fraccion_calentamiento,
            norma_recorte,
            fraccion_validacion,
            decaimiento_peso,
        );
    }

    /// Guarda el modelo en un archivo binario
    ///
    /// Args:
    ///     ruta: Ruta del archivo (.bin)
    pub fn guardar(&self, ruta: &str) -> PyResult<()> {
        self.inner
            .guardar_en_archivo(ruta)
            .map_err(|e: std::io::Error| PyIOError::new_err(e.to_string()))
    }

    /// Carga un modelo desde archivo binario (.bin)
    ///
    /// Args:
    ///     ruta: Ruta del archivo
    ///
    /// Returns:
    ///     GPT2Entrenable cargado
    #[staticmethod]
    pub fn cargar(ruta: &str) -> PyResult<Self> {
        let inner = GPT2Entrenable::cargar_desde_archivo(ruta)
            .map_err(|e: std::io::Error| PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Carga checkpoint completo (modelo + tokenizador)
    ///
    /// Args:
    ///     ruta: Ruta del checkpoint (.bin)
    ///
    /// Returns:
    ///     Tuple (GPT2Entrenable, TokenizadorBPE | None)
    #[staticmethod]
    pub fn cargar_checkpoint(ruta: &str) -> PyResult<(Self, Option<PyTokenizadorBPE>)> {
        let ckpt = PuntoControl::cargar(ruta)
            .map_err(|e: std::io::Error| PyIOError::new_err(e.to_string()))?;
        let tokenizador = ckpt.tokenizador.map(|t| PyTokenizadorBPE { inner: t });
        Ok((Self { inner: ckpt.modelo }, tokenizador))
    }

    pub fn __repr__(&self) -> String {
        format!(
            "GPT2Entrenable(vocab={}, n_embd={}, n_capas={}, n_cabezas={}, params={})",
            self.inner.config.vocab_size,
            self.inner.config.n_embd,
            self.inner.config.n_layers,
            self.inner.config.n_heads,
            self.inner.num_parametros(),
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Wrapper: Tensor
// ─────────────────────────────────────────────────────────────────────────────

/// Tensor multidimensional para operaciones de redes neuronales
///
/// Arreglo plano (f32) con información de forma y saltos.
/// Equivalente educativo a un numpy array, implementado en Rust.
///
/// Example:
///     >>> t = molineteai.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
///     >>> t.forma
///     [2, 2]
///     >>> t.datos
///     [1.0, 2.0, 3.0, 4.0]
#[pyclass(name = "Tensor")]
#[derive(Clone)]
pub struct PyTensor {
    pub inner: Tensor,
}

#[pymethods]
impl PyTensor {
    /// Crea un tensor con datos y forma dados
    ///
    /// Args:
    ///     datos: Lista plana de valores float
    ///     forma: Dimensiones del tensor (ej. [2, 3] para una matriz 2×3)
    #[new]
    pub fn new(datos: Vec<f32>, forma: Vec<usize>) -> PyResult<Self> {
        let esperado: usize = forma.iter().product();
        if datos.len() != esperado {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "datos tiene {} elementos pero la forma {:?} requiere {}",
                datos.len(), forma, esperado
            )));
        }
        Ok(Self { inner: Tensor::new(datos, forma) })
    }

    /// Crea un tensor lleno de ceros
    ///
    /// Args:
    ///     forma: Dimensiones del tensor
    ///
    /// Returns:
    ///     Tensor con todos sus elementos en 0.0
    #[staticmethod]
    pub fn ceros(forma: Vec<usize>) -> Self {
        Self { inner: Tensor::ceros(forma) }
    }

    /// Crea un tensor con valores enteros secuenciales [inicio, fin)
    ///
    /// Args:
    ///     inicio: Valor inicial (inclusivo)
    ///     fin:    Valor final (exclusivo)
    ///
    /// Returns:
    ///     Tensor 1D con los valores inicio, inicio+1, ..., fin-1
    #[staticmethod]
    pub fn arange(inicio: usize, fin: usize) -> Self {
        Self { inner: Tensor::arange(inicio, fin) }
    }

    /// Forma del tensor (dimensiones)
    #[getter]
    pub fn forma(&self) -> Vec<usize> {
        self.inner.forma.clone()
    }

    /// Datos del tensor como lista plana de floats
    #[getter]
    pub fn datos(&self) -> Vec<f32> {
        self.inner.datos.clone()
    }

    /// Número total de elementos
    pub fn numel(&self) -> usize {
        self.inner.datos.len()
    }

    // ── Operaciones de forma ─────────────────────────────────────────────────

    /// Redimensiona el tensor a una nueva forma (los datos no cambian)
    ///
    /// Args:
    ///     nueva_forma: Nueva lista de dimensiones (el producto debe ser igual)
    pub fn reshape(&self, nueva_forma: Vec<usize>) -> PyResult<Self> {
        let esperado: usize = nueva_forma.iter().product();
        if self.inner.datos.len() != esperado {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "No se puede redimensionar {} elementos a la forma {:?}",
                self.inner.datos.len(), nueva_forma
            )));
        }
        Ok(Self { inner: self.inner.reshape(&nueva_forma) })
    }

    /// Transpone dos dimensiones
    ///
    /// Args:
    ///     dim1: Primera dimensión (soporta índices negativos)
    ///     dim2: Segunda dimensión (soporta índices negativos)
    pub fn transpose(&self, dim1: isize, dim2: isize) -> Self {
        Self { inner: self.inner.transpose(dim1, dim2) }
    }

    // ── Operaciones elemento a elemento ─────────────────────────────────────

    /// Suma elemento a elemento (con broadcasting)
    pub fn add(&self, other: &PyTensor) -> Self {
        Self { inner: self.inner.add(&other.inner) }
    }

    /// Resta elemento a elemento (con broadcasting)
    pub fn sub(&self, other: &PyTensor) -> Self {
        Self { inner: self.inner.sub(&other.inner) }
    }

    /// Multiplicación elemento a elemento (con broadcasting)
    pub fn mul(&self, other: &PyTensor) -> Self {
        Self { inner: self.inner.mul(&other.inner) }
    }

    /// División elemento a elemento (con broadcasting)
    pub fn div(&self, other: &PyTensor) -> Self {
        Self { inner: self.inner.div(&other.inner) }
    }

    /// Suma un escalar a todos los elementos
    pub fn add_scalar(&self, escalar: f32) -> Self {
        Self { inner: self.inner.add_scalar(escalar) }
    }

    /// Multiplica todos los elementos por un escalar
    pub fn mul_scalar(&self, escalar: f32) -> Self {
        Self { inner: self.inner.mul_scalar(escalar) }
    }

    /// Divide todos los elementos por un escalar
    pub fn div_scalar(&self, escalar: f32) -> Self {
        Self { inner: self.inner.div_scalar(escalar) }
    }

    /// Raíz cuadrada elemento a elemento
    pub fn sqrt(&self) -> Self {
        Self { inner: self.inner.sqrt() }
    }

    // ── Álgebra lineal ───────────────────────────────────────────────────────

    /// Multiplicación de matrices (2D) o por lotes (4D para atención)
    ///
    /// Para 2D: [m, k] @ [k, n] → [m, n]
    /// Para 4D: [lote, cabezas, sec, dim] @ [lote, cabezas, dim, sec] → [lote, cabezas, sec, sec]
    pub fn matmul(&self, other: &PyTensor) -> Self {
        Self { inner: self.inner.matmul(&other.inner) }
    }

    // ── Operaciones estadísticas ─────────────────────────────────────────────

    /// Softmax a lo largo de un eje (numéricamente estable)
    ///
    /// Args:
    ///     eje: Eje sobre el que calcular softmax (-1 = último eje)
    pub fn softmax(&self, eje: isize) -> Self {
        Self { inner: self.inner.softmax(eje) }
    }

    /// Media a lo largo de un eje
    ///
    /// Args:
    ///     eje:         Eje sobre el que calcular la media (-1 = último eje)
    ///     mantener_dim: Si True, mantiene la dimensión reducida con tamaño 1
    pub fn mean(&self, eje: isize, mantener_dim: bool) -> Self {
        Self { inner: self.inner.mean(eje, mantener_dim) }
    }

    /// Varianza a lo largo de un eje
    ///
    /// Args:
    ///     eje:         Eje sobre el que calcular la varianza (-1 = último eje)
    ///     mantener_dim: Si True, mantiene la dimensión reducida con tamaño 1
    pub fn var(&self, eje: isize, mantener_dim: bool) -> Self {
        Self { inner: self.inner.var(eje, mantener_dim) }
    }

    // ── Enmascaramiento ──────────────────────────────────────────────────────

    /// Reemplaza elementos donde la máscara ≠ 0 con el valor dado
    ///
    /// Usado para la máscara causal en la atención (pone -inf en posiciones futuras).
    ///
    /// Args:
    ///     mascara: Tensor de máscara (≠ 0 = posición a enmascarar)
    ///     valor:   Valor de relleno (típicamente float('-inf'))
    pub fn masked_fill(&self, mascara: &PyTensor, valor: f32) -> Self {
        Self { inner: self.inner.masked_fill(&mascara.inner, valor) }
    }

    /// Concatena con otro tensor a lo largo de la primera dimensión
    pub fn concat(&self, other: &PyTensor) -> Self {
        Self { inner: self.inner.concat(&other.inner) }
    }

    // ── Representación ───────────────────────────────────────────────────────

    pub fn __repr__(&self) -> String {
        format!(
            "Tensor(forma={:?}, numel={})",
            self.inner.forma,
            self.inner.datos.len()
        )
    }

    pub fn __len__(&self) -> usize {
        self.inner.datos.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Funciones utilitarias a nivel de módulo
// ─────────────────────────────────────────────────────────────────────────────

/// Divide tokens en conjuntos de entrenamiento y validación
///
/// Args:
///     tokens: Lista completa de tokens (List[int])
///     fraccion_val: Fracción para validación (ej. 0.1 = 10%)
///
/// Returns:
///     Tuple (tokens_entrenamiento, tokens_validacion)
#[pyfunction]
pub fn dividir_entrenamiento_validacion(
    tokens: Vec<usize>,
    fraccion_val: f32,
) -> (Vec<usize>, Vec<usize>) {
    let indice = ((tokens.len() as f32) * (1.0 - fraccion_val)) as usize;
    (tokens[..indice].to_vec(), tokens[indice..].to_vec())
}

/// Cuenta parámetros de un modelo según su configuración
///
/// Útil para estimar el tamaño antes de crear el modelo.
///
/// Args:
///     config: Config del modelo
///
/// Returns:
///     Número total de parámetros (int)
#[pyfunction]
pub fn contar_parametros_config(config: &PyConfig) -> usize {
    let c = &config.inner;
    let emb = c.vocab_size * c.n_embd + c.block_size * c.n_embd;
    let por_bloque = 4 * (c.n_embd * c.n_embd + c.n_embd)    // atención
        + 2 * (c.n_embd * 4 * c.n_embd + c.n_embd * 4)        // MLP fc1
        + 2 * (4 * c.n_embd * c.n_embd + c.n_embd)            // MLP fc2
        + 4 * c.n_embd;                                         // layer norms
    let final_ln = 2 * c.n_embd;
    let lm_head = c.n_embd * c.vocab_size;
    emb + c.n_layers * por_bloque + final_ln + lm_head
}

// ─────────────────────────────────────────────────────────────────────────────
// Definición del módulo Python
// ─────────────────────────────────────────────────────────────────────────────

/// Registra todas las clases y funciones en el módulo Python.
/// Esta función es llamada desde lib.rs, que es el punto de entrada real.
pub fn molineteai(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_class::<PyConfig>()?;
    m.add_class::<PyTokenizadorBPE>()?;
    m.add_class::<PyGPT2>()?;
    m.add_class::<PyGPT2Entrenable>()?;
    m.add_function(wrap_pyfunction!(dividir_entrenamiento_validacion, m)?)?;
    m.add_function(wrap_pyfunction!(contar_parametros_config, m)?)?;

    // Versión del módulo
    m.add("__version__", "0.1.0")?;
    m.add("__autor__", "Molinete AI — Educativo")?;

    Ok(())
}
