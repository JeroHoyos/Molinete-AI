//! # El Puente entre Mundos — Enlace con Python
//!
//! Como el escudero Sancho Panza que actúa de mensajero entre Don Quijote y el mundo real,
//! este módulo hace de puente entre el alma veloz de Rust y la facilidad de uso de Python.
//!
//! Usando la biblioteca **PyO3**, cada clase de Rust se envuelve en un "wrapper" que
//! Python puede invocar como si fuera código Python nativo. El usuario de Python
//! no necesita saber nada de Rust — solo llama a `molineteai.GPT2Entrenable(config)`
//! y la magia sucede por debajo.
//!
//! ## Compilar el Puente
//!
//! ```bash
//! pip install maturin
//! maturin develop --release   # para desarrollo local
//! maturin build --release     # para distribución
//! ```
//!
//! ## Uso desde Python
//!
//! ```python
//! import molineteai
//!
//! # Crear y entrenar el tokenizador
//! tok = molineteai.TokenizadorBPE(1024)
//! tok.entrenar("Mi corpus de texto...", 1024)
//!
//! # Crear y entrenar el modelo
//! config = molineteai.Config.diminuta(tok.tam_vocabulario())
//! modelo = molineteai.GPT2Entrenable(config)
//! modelo.entrenar(tok, texto, pasos=5000, tasa_aprendizaje=3e-4)
//!
//! # Generar texto
//! prompt_ids = tok.codificar("En un lugar de la Mancha")
//! ids_generados = modelo.generar(prompt_ids, max_tokens=100, temperatura=0.8)
//! print(tok.decodificar(ids_generados))
//! ```

use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;

use crate::gpt2_entrenable::{entrenar_gpt2, GPT2Entrenable, PuntoControl};
use crate::modelo::{Config, GPT2};
use crate::tensor::Tensor;
use crate::tokenizador::TokenizadorBPE;

// ─── Wrapper: Config — la pergamino de configuración ─────────────────────────

/// Configuración de arquitectura del modelo GPT-2
///
/// Define todos los hiperparámetros de la arquitectura: tamaño del vocabulario,
/// dimensión de los embeddings, número de capas y cabezas de atención, etc.
#[pyclass(name = "Config")]
#[derive(Clone)]
pub struct PyConfig {
    pub inner: Config,
}

#[pymethods]
impl PyConfig {
    /// Forja una configuración personalizada
    #[new]
    #[pyo3(signature = (tam_vocabulario, n_embd, n_capas, n_cabezas, tam_bloque, tasa_dropout=0.1))]
    pub fn new(
        tam_vocabulario: usize, n_embd: usize, n_capas: usize,
        n_cabezas: usize, tam_bloque: usize, tasa_dropout: f32,
    ) -> PyResult<Self> {
        if n_embd % n_cabezas != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "n_embd ({}) debe ser divisible por n_cabezas ({})", n_embd, n_cabezas
            )));
        }
        Ok(Self { inner: Config { vocab_size: tam_vocabulario, n_embd, n_layers: n_capas,
            n_heads: n_cabezas, block_size: tam_bloque, dropout_rate: tasa_dropout } })
    }

    /// Configuración diminuta — para probar sin esperar (≈50K params)
    #[staticmethod]
    pub fn diminuta(tam_vocabulario: usize) -> Self {
        Self { inner: Config::tiny(tam_vocabulario) }
    }

    /// Configuración pequeña (≈200K params)
    #[staticmethod]
    pub fn pequena(tam_vocabulario: usize) -> Self {
        Self { inner: Config::small(tam_vocabulario) }
    }

    /// Configuración mediana (≈4M params)
    #[staticmethod]
    pub fn mediana(tam_vocabulario: usize) -> Self {
        Self { inner: Config::medium(tam_vocabulario) }
    }

    /// Configuración GPT-2 Small original (≈163M params)
    #[staticmethod]
    pub fn gpt2_small(tam_vocabulario: usize) -> Self {
        Self { inner: Config::gpt2_small(tam_vocabulario) }
    }

    #[getter] pub fn tam_vocabulario(&self) -> usize { self.inner.vocab_size }
    #[getter] pub fn n_embd(&self) -> usize { self.inner.n_embd }
    #[getter] pub fn n_capas(&self) -> usize { self.inner.n_layers }
    #[getter] pub fn n_cabezas(&self) -> usize { self.inner.n_heads }
    #[getter] pub fn tam_bloque(&self) -> usize { self.inner.block_size }
    #[getter] pub fn tasa_dropout(&self) -> f32 { self.inner.dropout_rate }

    pub fn __repr__(&self) -> String {
        format!("Config(vocab={}, n_embd={}, capas={}, cabezas={}, bloque={}, dropout={})",
            self.inner.vocab_size, self.inner.n_embd, self.inner.n_layers,
            self.inner.n_heads, self.inner.block_size, self.inner.dropout_rate)
    }
}

// ─── Wrapper: TokenizadorBPE — el escribano que codifica palabras ────────────

/// Tokenizador BPE (Codificación por Pares de Bytes)
///
/// Aprende a partir del corpus cómo fusionar caracteres frecuentes en tokens
/// más eficientes. GPT-2 original usa un vocabulario de 50.257 tokens.
///
/// Example:
///     >>> tok = molineteai.TokenizadorBPE(1024)
///     >>> tok.entrenar(texto, 1024)
///     >>> ids = tok.codificar("En un lugar de la Mancha")
///     >>> print(tok.decodificar(ids))
#[pyclass(name = "TokenizadorBPE")]
pub struct PyTokenizadorBPE {
    pub inner: TokenizadorBPE,
}

#[pymethods]
impl PyTokenizadorBPE {
    #[new]
    pub fn new(tam_vocabulario: usize) -> Self {
        Self { inner: TokenizadorBPE::new(tam_vocabulario) }
    }

    /// Entrena el tokenizador con el texto dado como corpus
    pub fn entrenar(&mut self, texto: &str, tam_vocabulario: usize) {
        self.inner.train(texto, tam_vocabulario);
    }

    /// Convierte texto en una lista de IDs de tokens
    pub fn codificar(&self, texto: &str) -> Vec<usize> { self.inner.codificar(texto) }

    /// Convierte una lista de IDs de tokens de vuelta en texto
    pub fn decodificar(&self, ids: Vec<usize>) -> String { self.inner.decodificar(&ids) }

    /// Tamaño actual del vocabulario aprendido
    pub fn tam_vocabulario(&self) -> usize { self.inner.tam_vocabulario() }

    /// Guarda el tokenizador en un archivo JSON
    pub fn guardar(&self, ruta: &str) -> PyResult<()> {
        self.inner.guardar(ruta).map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Carga un tokenizador desde un archivo JSON
    #[staticmethod]
    pub fn cargar(ruta: &str) -> PyResult<Self> {
        let inner = TokenizadorBPE::cargar(ruta).map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Imprime análisis del vocabulario aprendido
    pub fn analizar_vocabulario(&self, texto_ejemplo: &str) {
        self.inner.analizar_vocabulario(texto_ejemplo);
    }

    /// Estadísticas del tokenizador como diccionario Python
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

// ─── Wrapper: GPT2 — el modelo de solo inferencia ────────────────────────────

/// Modelo GPT-2 para inferencia pura (solo forward pass)
///
/// Para entrenar desde cero, usa `GPT2Entrenable`.
#[pyclass(name = "GPT2")]
pub struct PyGPT2 { inner: GPT2 }

#[pymethods]
impl PyGPT2 {
    #[new]
    pub fn new(config: &PyConfig) -> Self { Self { inner: GPT2::new(&config.inner) } }

    /// Ejecuta el forward pass y devuelve los logits aplanados
    pub fn forward(&self, tokens: Vec<Vec<usize>>) -> Vec<f32> {
        self.inner.forward(&tokens).datos
    }

    /// Forma lógica de la salida: (lote, secuencia, vocabulario)
    pub fn forma_salida(&self, tam_lote: usize, long_sec: usize) -> (usize, usize, usize) {
        (tam_lote, long_sec, self.inner.configuracion.vocab_size)
    }

    /// Número total de parámetros del modelo
    pub fn num_parametros(&self) -> usize { self.inner.contar_parametros() }

    pub fn __repr__(&self) -> String {
        format!("GPT2(vocab={}, n_embd={}, capas={}, cabezas={}, params={})",
            self.inner.configuracion.vocab_size, self.inner.configuracion.n_embd,
            self.inner.configuracion.n_layers, self.inner.configuracion.n_heads,
            self.inner.contar_parametros())
    }
}

// ─── Wrapper: GPT2Entrenable — el modelo completo con forward + backward ─────

/// Modelo GPT-2 entrenable con forward y backward pass completos
///
/// Incluye el bucle de entrenamiento con warmup de LR, decaimiento coseno,
/// recorte de gradientes, early stopping, checkpoints y logging CSV.
///
/// Example:
///     >>> config = molineteai.Config.diminuta(512)
///     >>> modelo = molineteai.GPT2Entrenable(config)
///     >>> modelo.entrenar(tokenizador, texto, pasos=1000, tasa_aprendizaje=3e-4)
///     >>> texto = tokenizador.decodificar(modelo.generar(prompt_ids, 100, 0.8))
#[pyclass(name = "GPT2Entrenable")]
pub struct PyGPT2Entrenable { inner: GPT2Entrenable }

#[pymethods]
impl PyGPT2Entrenable {
    #[new]
    pub fn new(config: &PyConfig) -> Self { Self { inner: GPT2Entrenable::new(&config.inner) } }

    /// Número total de parámetros entrenables
    pub fn num_parametros(&self) -> usize { self.inner.num_parametros() }

    /// Genera texto dado un prompt de tokens
    ///
    /// Args:
    ///     prompt_ids: IDs de tokens iniciales
    ///     max_tokens: Tokens nuevos a generar
    ///     temperatura: >1 = más creativo, <1 = más conservador
    pub fn generar(&self, prompt_ids: Vec<usize>, max_tokens: usize, temperatura: f32) -> Vec<usize> {
        self.inner.generar(&prompt_ids, max_tokens, temperatura)
    }

    /// Entrena el modelo con el corpus dado
    ///
    /// Incluye: warmup de LR, decaimiento coseno, recorte de gradientes,
    /// early stopping, checkpoints y logging CSV.
    #[pyo3(signature = (
        tokenizador, texto, pasos=10000, tasa_aprendizaje=3e-4, long_secuencia=None,
        dir_salida=None, paciencia=5000, fraccion_calentamiento=0.1,
        norma_recorte=1.0, fraccion_validacion=0.1, decaimiento_peso=0.01,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn entrenar(
        &mut self, tokenizador: &PyTokenizadorBPE, texto: &str,
        pasos: usize, tasa_aprendizaje: f32, long_secuencia: Option<usize>,
        dir_salida: Option<&str>, paciencia: usize, fraccion_calentamiento: f32,
        norma_recorte: f32, fraccion_validacion: f32, decaimiento_peso: f32,
    ) {
        let long_sec = long_secuencia.unwrap_or(self.inner.config.block_size);
        entrenar_gpt2(
            &mut self.inner, &tokenizador.inner, texto,
            pasos, tasa_aprendizaje, long_sec, dir_salida, paciencia,
            fraccion_calentamiento, norma_recorte, fraccion_validacion, decaimiento_peso,
        );
    }

    /// Guarda el modelo en un archivo binario
    pub fn guardar(&self, ruta: &str) -> PyResult<()> {
        self.inner.guardar_en_archivo(ruta).map_err(|e: std::io::Error| PyIOError::new_err(e.to_string()))
    }

    /// Carga un modelo desde archivo binario
    #[staticmethod]
    pub fn cargar(ruta: &str) -> PyResult<Self> {
        let inner = GPT2Entrenable::cargar_desde_archivo(ruta)
            .map_err(|e: std::io::Error| PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Carga checkpoint completo (modelo + tokenizador)
    #[staticmethod]
    pub fn cargar_checkpoint(ruta: &str) -> PyResult<(Self, Option<PyTokenizadorBPE>)> {
        let ckpt = PuntoControl::cargar(ruta)
            .map_err(|e: std::io::Error| PyIOError::new_err(e.to_string()))?;
        let tokenizador = ckpt.tokenizador.map(|t| PyTokenizadorBPE { inner: t });
        Ok((Self { inner: ckpt.modelo }, tokenizador))
    }

    pub fn __repr__(&self) -> String {
        format!("GPT2Entrenable(vocab={}, n_embd={}, capas={}, cabezas={}, params={})",
            self.inner.config.vocab_size, self.inner.config.n_embd,
            self.inner.config.n_layers, self.inner.config.n_heads,
            self.inner.num_parametros())
    }
}

// ─── Wrapper: Tensor — el pergamino numérico desde Python ────────────────────

/// Tensor multidimensional — el pergamino numérico accesible desde Python
///
/// Equivalente educativo de un numpy array, implementado en Rust para velocidad.
///
/// Example:
///     >>> t = molineteai.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
///     >>> t.forma  # [2, 2]
///     >>> t.datos  # [1.0, 2.0, 3.0, 4.0]
#[pyclass(name = "Tensor")]
#[derive(Clone)]
pub struct PyTensor { pub inner: Tensor }

#[pymethods]
impl PyTensor {
    #[new]
    pub fn new(datos: Vec<f32>, forma: Vec<usize>) -> PyResult<Self> {
        let esperado: usize = forma.iter().product();
        if datos.len() != esperado {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "datos tiene {} elementos pero la forma {:?} requiere {}", datos.len(), forma, esperado
            )));
        }
        Ok(Self { inner: Tensor::new(datos, forma) })
    }

    #[staticmethod] pub fn ceros(forma: Vec<usize>) -> Self { Self { inner: Tensor::ceros(forma) } }
    #[staticmethod] pub fn arange(inicio: usize, fin: usize) -> Self { Self { inner: Tensor::arange(inicio, fin) } }

    #[getter] pub fn forma(&self) -> Vec<usize> { self.inner.forma.clone() }
    #[getter] pub fn datos(&self) -> Vec<f32> { self.inner.datos.clone() }
    pub fn numel(&self) -> usize { self.inner.datos.len() }

    pub fn reshape(&self, nueva_forma: Vec<usize>) -> PyResult<Self> {
        let esperado: usize = nueva_forma.iter().product();
        if self.inner.datos.len() != esperado {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "No se puede redimensionar {} elementos a {:?}", self.inner.datos.len(), nueva_forma
            )));
        }
        Ok(Self { inner: self.inner.reshape(&nueva_forma) })
    }

    pub fn transpose(&self, dim1: isize, dim2: isize) -> Self { Self { inner: self.inner.transpose(dim1, dim2) } }
    pub fn add(&self, other: &PyTensor) -> Self { Self { inner: self.inner.add(&other.inner) } }
    pub fn sub(&self, other: &PyTensor) -> Self { Self { inner: self.inner.sub(&other.inner) } }
    pub fn mul(&self, other: &PyTensor) -> Self { Self { inner: self.inner.mul(&other.inner) } }
    pub fn div(&self, other: &PyTensor) -> Self { Self { inner: self.inner.div(&other.inner) } }
    pub fn add_scalar(&self, escalar: f32) -> Self { Self { inner: self.inner.add_scalar(escalar) } }
    pub fn mul_scalar(&self, escalar: f32) -> Self { Self { inner: self.inner.mul_scalar(escalar) } }
    pub fn div_scalar(&self, escalar: f32) -> Self { Self { inner: self.inner.div_scalar(escalar) } }
    pub fn sqrt(&self) -> Self { Self { inner: self.inner.sqrt() } }
    pub fn matmul(&self, other: &PyTensor) -> Self { Self { inner: self.inner.matmul(&other.inner) } }
    pub fn softmax(&self, eje: isize) -> Self { Self { inner: self.inner.softmax(eje) } }
    pub fn mean(&self, eje: isize, mantener_dim: bool) -> Self { Self { inner: self.inner.mean(eje, mantener_dim) } }
    pub fn var(&self, eje: isize, mantener_dim: bool) -> Self { Self { inner: self.inner.var(eje, mantener_dim) } }
    pub fn masked_fill(&self, mascara: &PyTensor, valor: f32) -> Self { Self { inner: self.inner.masked_fill(&mascara.inner, valor) } }
    pub fn concat(&self, other: &PyTensor) -> Self { Self { inner: self.inner.concat(&other.inner) } }

    pub fn __repr__(&self) -> String {
        format!("Tensor(forma={:?}, numel={})", self.inner.forma, self.inner.datos.len())
    }
    pub fn __len__(&self) -> usize { self.inner.datos.len() }
}

// ─── Funciones del módulo ────────────────────────────────────────────────────

/// Divide tokens en conjuntos de entrenamiento y validación
#[pyfunction]
pub fn dividir_entrenamiento_validacion(tokens: Vec<usize>, fraccion_val: f32) -> (Vec<usize>, Vec<usize>) {
    let indice = ((tokens.len() as f32) * (1.0 - fraccion_val)) as usize;
    (tokens[..indice].to_vec(), tokens[indice..].to_vec())
}

/// Cuenta parámetros de un modelo según su configuración
#[pyfunction]
pub fn contar_parametros_config(config: &PyConfig) -> usize {
    let c = &config.inner;
    let emb = c.vocab_size * c.n_embd + c.block_size * c.n_embd;
    let por_bloque = 4 * (c.n_embd * c.n_embd + c.n_embd)
        + 2 * (c.n_embd * 4 * c.n_embd + c.n_embd * 4)
        + 2 * (4 * c.n_embd * c.n_embd + c.n_embd)
        + 4 * c.n_embd;
    let final_ln = 2 * c.n_embd;
    let lm_head = c.n_embd * c.vocab_size;
    emb + c.n_layers * por_bloque + final_ln + lm_head
}

/// Registra todas las clases y funciones en el módulo Python
pub fn molineteai(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_class::<PyConfig>()?;
    m.add_class::<PyTokenizadorBPE>()?;
    m.add_class::<PyGPT2>()?;
    m.add_class::<PyGPT2Entrenable>()?;
    m.add_function(wrap_pyfunction!(dividir_entrenamiento_validacion, m)?)?;
    m.add_function(wrap_pyfunction!(contar_parametros_config, m)?)?;
    m.add("__version__", "0.1.0")?;
    m.add("__autor__", "Molinete AI — Educativo")?;
    Ok(())
}
