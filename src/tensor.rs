//! Operaciones con Tensores para Redes Neuronales
//!
//! Este módulo proporciona una biblioteca de tensores mínima optimizada para modelos transformer.
//! Los tensores almacenan arreglos multidimensionales con información de forma (shape) y saltos (stride)
//! para una indexación y disposición en memoria eficientes.
//!
//! ## Conceptos Principales
//!
//! - **Datos (Data)**: Un `Vec<f32>` plano que almacena todos los elementos en orden de fila mayor (row-major order).
//! - **Forma (Shape)**: Dimensiones del tensor (ej., `[lote, secuencia, dimension]`).
//! - **Saltos (Strides)**: Tamaños de paso para cada dimensión para calcular los índices planos.
//!
//! ## Ejemplo
//!
//! ```rust
//! use molineteai::Tensor;
//!
//! // Crear una matriz 2x3
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let tensor = Tensor::new(data, vec![2, 3]);
//!
//! // Multiplicación de matrices
//! let other = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2]);
//! let result = tensor.matmul(&other);
//! assert_eq!(result.shape, vec![2, 2]);
//! ```
//!
//! ## Optimizaciones de Rendimiento
//!
//! Varias operaciones utilizan procesamiento en paralelo a través de Rayon para mejorar el rendimiento:
//!
//! - **Multiplicación de matrices**: Algoritmo de bloqueo de caché (cache-blocked) con procesamiento de filas en paralelo.
//! - **Operaciones elemento a elemento (Element-wise)**: Iteración paralela sobre los datos.
//! - **Softmax**: Cálculo en paralelo por fila.
//!
//! Estas optimizaciones proporcionan un aumento de velocidad de 2 a 4 veces en CPUs multinúcleo, 
//! manteniendo el código educativo y comprensible. Todas las optimizaciones están claramente marcadas 
//! en el código con comentarios que explican el enfoque.

// Librería externa para procesar datos en paralelo (multihilo) y ganar velocidad.
use rayon::prelude::*;

/// Un arreglo multidimensional para cálculos de redes neuronales
///
/// Los tensores almacenan datos en un `Vec<f32>` contiguo con información de forma y saltos
/// para una indexación multidimensional eficiente. Todas las operaciones usan una disposición 
/// de memoria de fila mayor (estilo C, row-major layout).
///
/// # Campos
///
/// - `datos`: Arreglo plano de valores f32
/// - `forma`: Dimensiones (ej., `[2, 3]` para una matriz de 2x3)
/// - `saltos`: Tamaños de paso para cada dimensión (calculados a partir de la forma)
///
/// # Disposición en Memoria
///
/// Para la forma `[2, 3]`, los datos se almacenan como: `[fila0_col0, fila0_col1, fila0_col2, fila1_col0, fila1_col1, fila1_col2]`
///
/// Los saltos (strides) serían `[3, 1]`, lo que significa:
/// - Moverse un paso en la dimensión 0 (filas) avanza 3 posiciones en los datos
/// - Moverse un paso en la dimensión 1 (columnas) avanza 1 posición en los datos
#[derive(Clone, Debug)]
pub struct Tensor {
    /// Almacenamiento plano de todos los elementos del tensor
    pub datos: Vec<f32>,
    /// Forma del tensor (dimensiones)
    pub forma: Vec<usize>,
    /// Saltos para cada dimensión (calculados a partir de la forma)
    pub saltos: Vec<usize>,
}

impl Tensor {
    /// Crea un nuevo tensor con los datos y la forma dados
    ///
    /// # Argumentos
    ///
    /// * `data` - Vector plano de valores
    /// * `shape` - Dimensiones del tensor
    ///
    /// # Pánicos
    ///
    /// Entra en pánico si el producto de las dimensiones de la forma no es igual a la longitud de los datos
    ///
    /// # Ejemplo
    ///
    /// ```rust
    /// # use molineteai::Tensor;
    /// let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// assert_eq!(tensor.shape, vec![2, 2]);
    /// ```
    pub fn new(datos: Vec<f32>, forma: Vec<usize>) -> Self {
        let tamaño_esperado: usize = forma.iter().product();
        assert_eq!(
            datos.len(),
            tamaño_esperado,
            "La longitud de los datos ({}) no coincide con la forma {:?} (se esperaba {})",
            datos.len(),
            forma,
            tamaño_esperado
        );

        let saltos = Self::calcular_saltos(&forma);
        Self {
            datos,
            forma,
            saltos,
        }
    }

    /// Crea un tensor lleno de ceros
    ///
    /// # Argumentos
    ///
    /// * `forma` - Dimensiones del tensor
    ///
    /// # Ejemplo
    ///
    /// ```rust
    /// # use molineteai::Tensor;
    /// let tensor = Tensor::ceros(vec![3, 4]);
    /// assert_eq!(tensor.datos.len(), 12);
    /// assert!(tensor.datos.iter().all(|&x| x == 0.0));
    /// ```
    pub fn ceros(forma: Vec<usize>) -> Self {
        let tamaño: usize = forma.iter().product();
        let datos = vec![0.0; tamaño];
        Self::new(datos, forma)
    }

    /// Calcula los saltos a partir de la forma (disposición de fila mayor)
    ///
    /// Para la forma `[d0, d1, d2]`, los saltos son `[d1*d2, d2, 1]`
    ///
    /// # Argumentos
    ///
    /// * `forma` - Dimensiones del tensor
    ///
    /// # Retorna
    ///
    /// Vector con los valores de salto para cada dimensión
    fn calcular_saltos(forma: &[usize]) -> Vec<usize> {
        let mut saltos = vec![1; forma.len()];
        for i in (0..forma.len().saturating_sub(1)).rev() {
            saltos[i] = saltos[i + 1] * forma[i + 1];
        }
        saltos
    }

    /// Multiplicación de matrices
    ///
    /// Soporta:
    /// - 2D × 2D: Multiplicación de matrices estándar
    /// - 4D × 4D: Multiplicación de matrices por lotes para atención (ver más abajo)
    ///
    /// # Multiplicación de Matrices 2D
    ///
    /// Para `A @ B` donde `A` es `[m, k]` y `B` es `[k, n]`:
    /// - Forma del resultado: `[m, n]`
    /// - Cada elemento `C[i,j] = sum(A[i,k] * B[k,j])` para todo k
    ///
    /// # Rendimiento
    ///
    /// - **Matrices pequeñas** (< 1K ops): Cálculo secuencial
    /// - **Matrices grandes** (≥ 1K ops): Algoritmo en paralelo de bloqueo de caché (cache-blocked)
    ///
    /// La versión paralela utiliza bloques de 8x8 para la eficiencia de la caché y 
    /// se paraleliza a través de las filas de salida, proporcionando un aumento de 
    /// velocidad de 2 a 4 veces en las típicas CPUs multinúcleo.
    ///
    /// # Multiplicación de Matrices por Lotes 4D (Batched Matmul)
    ///
    /// Para cálculos de atención con la forma `[lote, num_cabezas, sec, dim_cabeza]`
    /// Procesa cada par (lote, cabeza) de forma independiente y en paralelo.
    ///
    /// # Pánicos
    ///
    /// Entra en pánico si las dimensiones son incompatibles o no están soportadas
    ///
    /// # Ejemplo
    ///
    /// ```rust
    /// # use molineteai::Tensor;
    /// let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    /// let c = a.matmul(&b);
    /// assert_eq!(c.forma, vec![2, 2]);
    /// ```
    ///
    /// Bucle interno optimizado con SIMD para la multiplicación de matrices
    /// Calcula: resultado[j] += val_a * b[j] para todo j
    /// Estructurado para la auto-vectorización en ARM NEON (Apple Silicon)
    #[inline(always)]
    fn matmul_interno_simd(val_a: f32, b: &[f32], resultado: &mut [f32]) {
        // Bucle simple que LLVM puede auto-vectorizar
        // En Apple Silicon esto utilizará instrucciones SIMD ARM NEON
        for (r, &val_b) in resultado.iter_mut().zip(b.iter()) {
            *r += val_a * val_b;
        }
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        // === MULTIPLICACIÓN DE MATRICES 2D ===
        if self.forma.len() == 2 && other.forma.len() == 2 {
            assert_eq!(
                self.forma[1], other.forma[0],
                "Dimensiones de matriz incompatibles: [{}, {}] @ [{}, {}]",
                self.forma[0], self.forma[1], other.forma[0], other.forma[1]
            );

            let m = self.forma[0];
            let n = other.forma[1];
            let k = self.forma[1];

            // Usa la versión paralela para matrices más grandes (umbral de trabajo: 1000 operaciones)
            // Este umbral equilibra la sobrecarga paralela con las ganancias de rendimiento
            if m * n * k >= 1_000 {
                return self.matmul_paralelo_bloques(other, m, n, k);
            }

            // Versión secuencial para matrices pequeñas (evita la sobrecarga paralela)
            let mut resultado = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut suma = 0.0;
                    for l in 0..k {
                        suma += self.datos[i * k + l] * other.datos[l * n + j];
                    }
                    resultado[i * n + j] = suma;
                }
            }

            return Tensor::new(resultado, vec![m, n]);
        }

        // === MULTIPLICACIÓN DE MATRICES POR LOTES 4D (para atención) ===
        // Forma: [lote, num_cabezas, sec, dim_cabeza] @ [lote, num_cabezas, dim_cabeza, sec]
        if self.forma.len() == 4 && other.forma.len() == 4 {
            let lote = self.forma[0];
            let num_cabezas = self.forma[1];
            let sec1 = self.forma[2];
            let dim_interna = self.forma[3];
            let sec2 = other.forma[3];

            assert_eq!(
                other.forma[2], dim_interna,
                "Las dimensiones internas deben coincidir para la multiplicación por lotes"
            );

            let tam_total = lote * num_cabezas * sec1 * sec2;
            let mut resultado = vec![0.0; tam_total];

            // Paralelizar sobre las combinaciones de (lote, cabeza)
            // Cada par (lote, cabeza) calcula una multiplicación de matrices sec1×sec2 independiente
            resultado
                .par_chunks_mut(sec1 * sec2)
                .enumerate()
                .for_each(|(idx_lc, bloque)| {
                    let b = idx_lc / num_cabezas;
                    let h = idx_lc % num_cabezas;

                    // Calcula el matmul 2D para este lote/cabeza
                    for i in 0..sec1 {
                        for j in 0..sec2 {
                            let mut suma = 0.0;
                            for l in 0..dim_interna {
                                let self_idx = ((b * num_cabezas + h) * sec1 + i) * dim_interna + l;
                                let other_idx = ((b * num_cabezas + h) * dim_interna + l) * sec2 + j;
                                suma += self.datos[self_idx] * other.datos[other_idx];
                            }
                            bloque[i * sec2 + j] = suma;
                        }
                    }
                });

            return Tensor::new(resultado, vec![lote, num_cabezas, sec1, sec2]);
        }

        panic!(
            "Formas de matmul no soportadas: {:?} @ {:?}",
            self.forma, other.forma
        );
    }

    /// Multiplicación de matrices en paralelo con bloqueo de caché
    ///
    /// Esta optimización proporciona un aumento significativo de velocidad para matrices grandes mediante:
    ///
    /// 1. **Bloqueo de caché (Cache blocking)**: Procesa los datos en bloques de 8x8 que caben en la caché L1
    /// 2. **Procesamiento paralelo**: Distribuye los bloques de filas entre los núcleos de la CPU usando Rayon
    /// 3. **Localidad de memoria**: Los bucles internos acceden a la memoria de forma secuencial
    ///
    /// El tamaño de bloque de 8x8 (256 bytes por bloque) equilibra la eficiencia de la caché con
    /// las oportunidades de paralelismo. Bloques más pequeños mejorarían la tasa de aciertos de caché pero
    /// reducirían el paralelismo disponible; bloques más grandes harían lo contrario.
    ///
    /// # Rendimiento
    ///
    /// Típicamente de 2 a 4 veces más rápido que una implementación ingenua en CPUs multinúcleo,
    /// con un aumento de velocidad aún mayor para matrices más grandes.
    ///
    /// # Argumentos
    ///
    /// * `other` - Matriz del lado derecho
    /// * `m` - Filas en self
    /// * `n` - Columnas en other
    /// * `k` - Dimensión interna
    ///
    /// # Retorna
    ///
    /// Tensor resultante con forma `[m, n]`
    fn matmul_paralelo_bloques(&self, other: &Tensor, m: usize, n: usize, k: usize) -> Tensor {
        // Tamaño del bloque para la optimización de la caché
        // Bloques de 8x8 = 256 bytes (cabe muy bien en la caché L1: típicamente 32-64KB)
        const TAM_BLOQUE: usize = 8;

        let mut resultado = vec![0.0; m * n];

        // Paralelizar sobre los bloques de filas de salida
        // Cada hilo procesa TAM_BLOQUE filas de forma independiente
        resultado
            .par_chunks_mut(TAM_BLOQUE * n)
            .enumerate()
            .for_each(|(i_bloque, bloque_resultado)| {
                let i_inicio = i_bloque * TAM_BLOQUE;
                let i_fin = (i_inicio + TAM_BLOQUE).min(m);

                // Iterar sobre los bloques de columnas
                for j_inicio in (0..n).step_by(TAM_BLOQUE) {
                    let j_fin = (j_inicio + TAM_BLOQUE).min(n);

                    // Iterar sobre los bloques de la dimensión interna
                    for k_inicio in (0..k).step_by(TAM_BLOQUE) {
                        let k_fin = (k_inicio + TAM_BLOQUE).min(k);

                        // Calcular este bloque (bucles internos amigables con la caché)
                        for i in i_inicio..i_fin {
                            let desplazamiento_fila = (i - i_inicio) * n;
                            for k_idx in k_inicio..k_fin {
                                let val_a = self.datos[i * k + k_idx];

                                // Bucle más interno optimizado con SIMD
                                Self::matmul_interno_simd(
                                    val_a,
                                    &other.datos[k_idx * n + j_inicio..k_idx * n + j_fin],
                                    &mut bloque_resultado[desplazamiento_fila + j_inicio..desplazamiento_fila + j_fin],
                                );
                            }
                        }
                    }
                }
            });

        Tensor::new(resultado, vec![m, n])
    }

    /// Activación Softmax
    ///
    /// Calcula la función softmax a lo largo del eje especificado:
    ///
    /// ```text
    /// softmax(x)[i] = exp(x[i]) / sum(exp(x[j])) para todo j
    /// ```
    ///
    /// # Estabilidad Numérica
    ///
    /// Utiliza la versión numéricamente estable:
    ///
    /// ```text
    /// softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))
    /// ```
    ///
    /// Restar el máximo evita el desbordamiento (overflow) en exp() mientras produce
    /// el mismo resultado (ya que los factores máximos se cancelan).
    ///
    /// # Argumentos
    ///
    /// * `eje` - Eje a lo largo del cual calcular softmax (usa -1 para el último eje)
    ///
    /// # Rendimiento
    ///
    /// Para tensores 2D con eje=-1, softmax se calcula por fila en paralelo.
    ///
    /// # Ejemplo
    ///
    /// ```rust
    /// # use molineteai::Tensor;
    /// let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
    /// let resultado = tensor.softmax(-1);
    /// // La suma del resultado es 1.0 a lo largo de la última dimensión
    /// ```
    pub fn softmax(&self, eje: isize) -> Tensor {
        // Convertir el eje negativo a positivo
        let eje_pos = if eje < 0 {
            (self.forma.len() as isize + eje) as usize
        } else {
            eje as usize
        };

        // === SOFTMAX 2D POR FILA (caso común para atención) ===
        if self.forma.len() == 2 && eje_pos == 1 {
            let filas = self.forma[0];
            let columnas = self.forma[1];

            // Cálculo de softmax en paralelo por fila
            let resultado: Vec<f32> = (0..filas)
                .into_par_iter()
                .flat_map_iter(|i| {
                    let inicio = i * columnas;
                    let fin = inicio + columnas;
                    let fila = &self.datos[inicio..fin];

                    // Encontrar el máximo para la estabilidad numérica
                    let maximo = fila.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                    // Calcular exp(x - maximo)
                    let valores_exp: Vec<f32> = fila.iter().map(|&x| (x - maximo).exp()).collect();

                    // Normalizar
                    let suma: f32 = valores_exp.iter().sum();
                    valores_exp.into_iter().map(move |val| val / suma)
                })
                .collect();

            return Tensor::new(resultado, self.forma.clone());
        }

        // === RESPALDO: SOFTMAX GLOBAL ===
        // Menos común, pero incluido para estar completos
        let maximo = self.datos.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let valores_exp: Vec<f32> = self.datos.iter().map(|&x| (x - maximo).exp()).collect();
        let suma: f32 = valores_exp.iter().sum();
        let resultado = valores_exp.iter().map(|&x| x / suma).collect();

        Tensor::new(resultado, self.forma.clone())
    }

    /// Suma elemento a elemento con soporte de propagación (broadcasting)
    ///
    /// Soporta varios patrones de broadcasting comunes en transformers:
    ///
    /// 1. **Coincidencia exacta**: Misma forma
    /// 2. **Broadcasting de la última dimensión**: `[*, n] + [n]` (ej., sumar sesgo o bias)
    /// 3. **Broadcasting de lote**: `[lote, sec, dim] + [sec, dim]`
    ///
    /// # Argumentos
    ///
    /// * `other` - Tensor a sumar (puede tener una forma diferente si aplica el broadcasting)
    ///
    /// # Ejemplo
    ///
    /// ```rust
    /// # use molineteai::Tensor;
    /// let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// let b = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
    /// let c = a.add(&b);
    /// assert_eq!(c.datos, vec![2.0, 3.0, 4.0, 5.0]);
    /// ```
    pub fn add(&self, other: &Tensor) -> Tensor {
        // === COINCIDENCIA EXACTA: Misma forma ===
        if self.forma == other.forma {
            let resultado = self
                .datos
                .par_iter()
                .zip(&other.datos)
                .map(|(a, b)| a + b)
                .collect();
            return Tensor::new(resultado, self.forma.clone());
        }

        // === BROADCASTING DE LOTE: [lote, sec, dim] + [sec, dim] ===
        if self.forma.len() == 3 && other.forma.len() == 2 {
            let tam_lote = self.forma[0];
            let longitud_sec = self.forma[1];
            let dim = self.forma[2];

            assert_eq!(
                other.forma[0], longitud_sec,
                "La longitud de la secuencia debe coincidir para el broadcasting"
            );
            assert_eq!(other.forma[1], dim, "La dimensión debe coincidir para el broadcasting");

            let resultado: Vec<f32> = (0..tam_lote * longitud_sec * dim)
                .into_par_iter()
                .map(|i| {
                    let s = (i / dim) % longitud_sec;
                    let d = i % dim;
                    let idx_otro = s * dim + d;
                    self.datos[i] + other.datos[idx_otro]
                })
                .collect();
            return Tensor::new(resultado, self.forma.clone());
        }

        // === BROADCASTING DE LA ÚLTIMA DIMENSIÓN: [*, n] + [n] (ej., suma de sesgo) ===
        if self.forma.len() > other.forma.len() {
            let ultima_dim = *self.forma.last().unwrap();
            if other.datos.len() == ultima_dim {
                let resultado: Vec<f32> = (0..self.datos.len())
                    .into_par_iter()
                    .map(|i| {
                        let idx_otro = i % ultima_dim;
                        self.datos[i] + other.datos[idx_otro]
                    })
                    .collect();
                return Tensor::new(resultado, self.forma.clone());
            }
        }

        panic!(
            "Broadcasting no soportado para suma: {:?} + {:?}",
            self.forma, other.forma
        );
    }

    /// Multiplicación elemento a elemento con propagación (broadcasting)
    ///
    /// Ver `add()` para los patrones de broadcasting.
    pub fn mul(&self, other: &Tensor) -> Tensor {
        // Coincidencia exacta
        if self.forma == other.forma {
            let resultado = self
                .datos
                .par_iter()
                .zip(&other.datos)
                .map(|(a, b)| a * b)
                .collect();
            return Tensor::new(resultado, self.forma.clone());
        }

        // Broadcasting de la última dimensión
        if self.forma.len() > other.forma.len() {
            let ultima_dim = *self.forma.last().unwrap();
            if other.datos.len() == ultima_dim {
                let resultado: Vec<f32> = (0..self.datos.len())
                    .into_par_iter()
                    .map(|i| {
                        let idx_otro = i % ultima_dim;
                        self.datos[i] * other.datos[idx_otro]
                    })
                    .collect();
                return Tensor::new(resultado, self.forma.clone());
            }
        }

        panic!(
            "Broadcasting no soportado para multiplicación: {:?} * {:?}",
            self.forma, other.forma
        );
    }

    /// Resta elemento a elemento con propagación (broadcasting)
    pub fn sub(&self, other: &Tensor) -> Tensor {
        // Coincidencia exacta
        if self.forma == other.forma {
            let resultado = self
                .datos
                .par_iter()
                .zip(&other.datos)
                .map(|(a, b)| a - b)
                .collect();
            return Tensor::new(resultado, self.forma.clone());
        }

        // Broadcasting de la última dimensión (para operaciones keepdim)
        if self.forma.len() == other.forma.len() {
            let ultima_dim = *self.forma.last().unwrap();
            let ultima_dim_otro = *other.forma.last().unwrap();

            if ultima_dim_otro == 1
                && self.forma[..self.forma.len() - 1] == other.forma[..other.forma.len() - 1]
            {
                let resultado: Vec<f32> = (0..self.datos.len())
                    .into_par_iter()
                    .map(|i| {
                        let idx_otro = (i / ultima_dim) * ultima_dim_otro;
                        self.datos[i] - other.datos[idx_otro]
                    })
                    .collect();
                return Tensor::new(resultado, self.forma.clone());
            }
        }

        panic!(
            "Broadcasting no soportado para resta: {:?} - {:?}",
            self.forma, other.forma
        );
    }

    /// División elemento a elemento con propagación (broadcasting)
    pub fn div(&self, other: &Tensor) -> Tensor {
        // Coincidencia exacta
        if self.forma == other.forma {
            let resultado = self
                .datos
                .par_iter()
                .zip(&other.datos)
                .map(|(a, b)| a / b)
                .collect();
            return Tensor::new(resultado, self.forma.clone());
        }

        // Broadcasting de la última dimensión (para operaciones keepdim)
        if self.forma.len() == other.forma.len() {
            let ultima_dim = *self.forma.last().unwrap();
            let ultima_dim_otro = *other.forma.last().unwrap();

            if ultima_dim_otro == 1
                && self.forma[..self.forma.len() - 1] == other.forma[..other.forma.len() - 1]
            {
                let resultado: Vec<f32> = (0..self.datos.len())
                    .into_par_iter()
                    .map(|i| {
                        let idx_otro = (i / ultima_dim) * ultima_dim_otro;
                        self.datos[i] / other.datos[idx_otro]
                    })
                    .collect();
                return Tensor::new(resultado, self.forma.clone());
            }
        }

        panic!(
            "Broadcasting no soportado para división: {:?} / {:?}",
            self.forma, other.forma
        );
    }

    /// Suma un escalar a todos los elementos
    pub fn add_scalar(&self, escalar: f32) -> Tensor {
        let resultado = self.datos.par_iter().map(|&x| x + escalar).collect();
        Tensor::new(resultado, self.forma.clone())
    }

    /// Multiplica todos los elementos por un escalar
    pub fn mul_scalar(&self, escalar: f32) -> Tensor {
        let resultado = self.datos.par_iter().map(|&x| x * escalar).collect();
        Tensor::new(resultado, self.forma.clone())
    }

    /// Divide todos los elementos por un escalar
    pub fn div_scalar(&self, escalar: f32) -> Tensor {
        let resultado = self.datos.par_iter().map(|&x| x / escalar).collect();
        Tensor::new(resultado, self.forma.clone())
    }

    /// Raíz cuadrada elemento a elemento
    pub fn sqrt(&self) -> Tensor {
        let resultado = self.datos.par_iter().map(|&x| x.sqrt()).collect();
        Tensor::new(resultado, self.forma.clone())
    }

   /// Redimensiona (reshape) el tensor a una nueva forma
    ///
    /// El número total de elementos debe seguir siendo el mismo.
    ///
    /// # Ejemplo
    ///
    /// ```rust
    /// # use molineteai::Tensor;
    /// let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    /// let redimensionado = tensor.reshape(&[3, 2]);
    /// assert_eq!(redimensionado.forma, vec![3, 2]);
    /// ```
    pub fn reshape(&self, nueva_forma: &[usize]) -> Tensor {
        let nuevo_tamano: usize = nueva_forma.iter().product();
        assert_eq!(
            self.datos.len(),
            nuevo_tamano,
            "No se puede redimensionar: la cantidad de elementos no coincide"
        );
        Tensor::new(self.datos.clone(), nueva_forma.to_vec())
    }

    /// Transpone dos dimensiones
    ///
    /// # Argumentos
    ///
    /// * `dim1` - Primera dimensión a intercambiar (soporta indexación negativa)
    /// * `dim2` - Segunda dimensión a intercambiar (soporta indexación negativa)
    ///
    /// # Ejemplo
    ///
    /// ```rust
    /// # use molineteai::Tensor;
    /// let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// let transpuesto = tensor.transpose(0, 1);
    /// assert_eq!(transpuesto.forma, vec![2, 2]);
    /// ```
    pub fn transpose(&self, dim1: isize, dim2: isize) -> Tensor {
        let ndim = self.forma.len() as isize;

        // Convertir índices negativos
        let d1 = if dim1 < 0 { ndim + dim1 } else { dim1 } as usize;
        let d2 = if dim2 < 0 { ndim + dim2 } else { dim2 } as usize;

        // Crear nueva forma con las dimensiones intercambiadas
        let mut nueva_forma = self.forma.clone();
        nueva_forma.swap(d1, d2);

        // Para matrices 2D, podemos usar una transposición simple
        if self.forma.len() == 2 {
            let filas = self.forma[0];
            let columnas = self.forma[1];
            let mut resultado = vec![0.0; filas * columnas];

            for i in 0..filas {
                for j in 0..columnas {
                    resultado[j * filas + i] = self.datos[i * columnas + j];
                }
            }

            return Tensor::new(resultado, nueva_forma);
        }

        // Para tensores 4D transponiendo las dos últimas dimensiones (común en atención)
        // [lote, cabezales, sec1, sec2] -> [lote, cabezales, sec2, sec1]
        if self.forma.len() == 4 && d1 == 2 && d2 == 3 {
            let d0 = self.forma[0];
            let dim_d1 = self.forma[1];
            let dim_d2 = self.forma[2];
            let dim_d3 = self.forma[3];

            let mut resultado = vec![0.0; self.datos.len()];

            for i0 in 0..d0 {
                for i1 in 0..dim_d1 {
                    for i2 in 0..dim_d2 {
                        for i3 in 0..dim_d3 {
                            let idx_origen = ((i0 * dim_d1 + i1) * dim_d2 + i2) * dim_d3 + i3;
                            let idx_destino = ((i0 * dim_d1 + i1) * dim_d3 + i3) * dim_d2 + i2;
                            resultado[idx_destino] = self.datos[idx_origen];
                        }
                    }
                }
            }

            return Tensor::new(resultado, nueva_forma);
        }

        // Para dimensiones mayores, hacer una transposición completa con remapeo de pasos (strides)
        let pasos_antiguos = &self.saltos; 
        let mut nuevos_pasos = pasos_antiguos.clone();
        nuevos_pasos.swap(d1, d2);

        let tamano_total = self.datos.len();
        let mut resultado = vec![0.0; tamano_total];

        for (i, item) in resultado.iter_mut().enumerate().take(tamano_total) {
            // Calcular el multi-índice antiguo a partir del índice plano
            let mut idx_antiguo = 0;
            let mut restante = i;

            for (idx_dim, &paso) in nuevos_pasos.iter().enumerate() {
                let coord = restante / paso;
                restante %= paso;
                idx_antiguo += coord * pasos_antiguos[idx_dim];
            }

            *item = self.datos[idx_antiguo];
        }

        Tensor::new(resultado, nueva_forma)
    }

    /// Reemplaza los valores donde la máscara es verdadera con el valor dado
    ///
    /// Utilizado para el enmascaramiento causal en la atención (estableciendo posiciones futuras a -inf)
    ///
    /// Soporta propagación (broadcasting): [lote, num_cabezales, sec, sec] con máscara [sec, sec]
    ///
    /// # Argumentos
    ///
    /// * `mascara` - Máscara booleana (distinto de cero = verdadero)
    /// * `valor` - Valor a rellenar donde la máscara es verdadera
    pub fn masked_fill(&self, mascara: &Tensor, valor: f32) -> Tensor {
        // Manejar coincidencia exacta
        if self.forma == mascara.forma {
            let resultado = self
                .datos
                .par_iter()
                .zip(&mascara.datos)
                .map(|(&x, &m)| if m != 0.0 { valor } else { x })
                .collect();
            return Tensor::new(resultado, self.forma.clone());
        }

        // Manejar broadcasting: [lote, num_cabezales, sec, sec] con máscara [sec, sec]
        if self.forma.len() == 4 && mascara.forma.len() == 2 {
            let lote = self.forma[0];
            let num_cabezales = self.forma[1];
            let sec = self.forma[2];

            assert_eq!(mascara.forma[0], sec, "La dimensión de la secuencia de la máscara debe coincidir");
            assert_eq!(mascara.forma[1], sec, "La dimensión de la secuencia de la máscara debe coincidir");

            let mut resultado = Vec::with_capacity(self.datos.len());
            for _b in 0..lote {
                for _h in 0..num_cabezales {
                    for i in 0..sec {
                        for j in 0..sec {
                            let idx_mascara = i * sec + j;
                            let idx_self = resultado.len();
                            resultado.push(if mascara.datos[idx_mascara] != 0.0 {
                                valor
                            } else {
                                self.datos[idx_self]
                            });
                        }
                    }
                }
            }
            return Tensor::new(resultado, self.forma.clone());
        }

        panic!(
            "Broadcasting no soportado para masked_fill: {:?} con máscara {:?}",
            self.forma, mascara.forma
        );
    }

    /// Calcula la media (promedio) a lo largo de un eje
    ///
    /// # Argumentos
    ///
    /// * `eje` - Eje a lo largo del cual calcular la media (usa -1 para el último eje)
    /// * `mantener_dim` - Indica si se debe mantener la dimensión reducida (tamaño 1)
    pub fn mean(&self, eje: isize, mantener_dim: bool) -> Tensor {
        let eje_pos = if eje < 0 {
            (self.forma.len() as isize + eje) as usize
        } else {
            eje as usize
        };

        // Para tensores 2D, calcular la media a lo largo del eje especificado
        if self.forma.len() == 2 && eje_pos == 1 {
            // Media a lo largo de las columnas (el resultado tiene forma [filas, 1] o [filas])
            let filas = self.forma[0];
            let columnas = self.forma[1];

            let resultado: Vec<f32> = (0..filas)
                .into_par_iter()
                .map(|i| {
                    let inicio = i * columnas;
                    let fin = inicio + columnas;
                    let suma: f32 = self.datos[inicio..fin].iter().sum();
                    suma / columnas as f32
                })
                .collect();

            let nueva_forma = if mantener_dim { vec![filas, 1] } else { vec![filas] };
            return Tensor::new(resultado, nueva_forma);
        }

        // Para tensores 3D [lote, sec, dim], calcular la media a lo largo del último eje
        if self.forma.len() == 3 && eje_pos == 2 {
            let lote = self.forma[0];
            let sec = self.forma[1];
            let dim = self.forma[2];

            let resultado: Vec<f32> = (0..lote * sec)
                .into_par_iter()
                .map(|i| {
                    let inicio = i * dim;
                    let fin = inicio + dim;
                    let suma: f32 = self.datos[inicio..fin].iter().sum();
                    suma / dim as f32
                })
                .collect();

            let nueva_forma = if mantener_dim {
                vec![lote, sec, 1]
            } else {
                vec![lote, sec]
            };
            return Tensor::new(resultado, nueva_forma);
        }

        panic!("Operación mean (media) no soportada para la forma {:?}", self.forma);
    }

    /// Calcula la varianza a lo largo de un eje
    ///
    /// # Argumentos
    ///
    /// * `eje` - Eje a lo largo del cual calcular la varianza (típicamente -1 para la última dimensión)
    /// * `mantener_dim` - Indica si se debe mantener la dimensión reducida
    ///
    /// # Ejemplo
    ///
    /// ```rust
    /// # use molineteai::Tensor;
    /// let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    /// let varianza = x.var(-1, true); // Varianza a lo largo de la última dimensión
    /// assert_eq!(varianza.forma, vec![2, 1]);
    /// ```
    pub fn var(&self, eje: isize, mantener_dim: bool) -> Tensor {
        // Calcula la varianza a lo largo del eje especificado (típicamente la última dimensión para LayerNorm)
        if eje == -1 || eje as usize == self.forma.len() - 1 {
            let ultima_dim = self.forma[self.forma.len() - 1];
            let tamano_exterior = self.datos.len() / ultima_dim;

            let mut resultado = Vec::new();
            for i in 0..tamano_exterior {
                let inicio = i * ultima_dim;
                let fin = inicio + ultima_dim;
                let segmento = &self.datos[inicio..fin];

                // Calcula la media para este segmento
                let media: f32 = segmento.iter().sum::<f32>() / ultima_dim as f32;

                // Calcula la varianza
                let varianza =
                    segmento.iter().map(|&x| (x - media).powi(2)).sum::<f32>() / ultima_dim as f32;

                resultado.push(varianza);
            }

            let forma_resultado = if mantener_dim {
                let mut forma = self.forma.clone();
                forma[self.forma.len() - 1] = 1;
                forma
            } else {
                self.forma[..self.forma.len() - 1].to_vec()
            };

            return Tensor::new(resultado, forma_resultado);
        }

        panic!(
            "Operación var (varianza) no soportada para la forma {:?} con eje {}",
            self.forma, eje
        );
    }

    /// Crea un tensor con enteros secuenciales
    ///
    /// # Argumentos
    ///
    /// * `inicio` - Valor inicial (inclusivo)
    /// * `fin` - Valor final (exclusivo)
    ///
    /// # Ejemplo
    ///
    /// ```rust
    /// # use molineteai::Tensor;
    /// let tensor = Tensor::arange(0, 5);
    /// assert_eq!(tensor.datos, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn arange(inicio: usize, fin: usize) -> Tensor {
        let datos: Vec<f32> = (inicio..fin).map(|i| i as f32).collect();
        let longitud = datos.len();
        Tensor::new(datos, vec![longitud])
    }

    /// Concatena dos tensores a lo largo de la primera dimensión (dimensión de la secuencia)
    ///
    /// Utilizado para la caché KV: concatena K/V almacenados en caché con nuevos K/V.
    ///
    /// # Argumentos
    ///
    /// * `other` - Tensor con el que concatenar (debe tener la misma forma excepto en la primera dimensión)
    ///
    /// # Ejemplo
    ///
    /// ```rust
    /// # use molineteai::Tensor;
    /// let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// let b = Tensor::new(vec![5.0, 6.0], vec![1, 2]);
    /// let c = a.concat(&b);
    /// assert_eq!(c.forma, vec![3, 2]);
    /// assert_eq!(c.datos, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// ```
    pub fn concat(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.forma.len(),
            other.forma.len(),
            "Los tensores deben tener el mismo número de dimensiones para la concatenación"
        );

        // Comprueba que todas las dimensiones excepto la primera coincidan
        for i in 1..self.forma.len() {
            assert_eq!(
                self.forma[i], other.forma[i],
                "Todas las dimensiones excepto la primera deben coincidir para la concatenación"
            );
        }

        // Concatena a lo largo de la primera dimensión
        let mut nueva_forma = self.forma.clone();
        nueva_forma[0] += other.forma[0];

        let mut datos = Vec::with_capacity(self.datos.len() + other.datos.len());
        datos.extend_from_slice(&self.datos);
        datos.extend_from_slice(&other.datos);

        Tensor::new(datos, nueva_forma)
    }
}
