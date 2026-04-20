//! # El Pergamino de los Números — Módulo de Tensores
//!
//! Así como Sancho Panza cargaba las alforjas con todo lo necesario para la aventura,
//! este módulo carga los valores numéricos que alimentan el entendimiento de Molinete.
//!
//! Un **Tensor** no es sino un pergamino de muchas dimensiones: puede ser un número suelto
//! (escalar), una fila de palabras (vector), una tabla de conocimiento (matriz), o incluso
//! un cofre de cofres de cofres (tensor 4D). La magia reside en que toda esa riqueza vive
//! en una única hilera continua de memoria, como cuentas ensartadas en un hilo de seda.
//!
//! ## Los Tres Pilares del Pergamino
//!
//! - **datos**: El hilo de seda — un `Vec<f32>` plano con todos los valores.
//! - **forma**: La descripción del cofre — sus dimensiones, ej. `[lote, secuencia, embedding]`.
//! - **saltos**: El mapa del tesoro — cuántos pasos dar en memoria para avanzar en cada dimensión.
//!
//! ## Ejemplo
//!
//! ```rust
//! use molineteai::Tensor;
//!
//! // Una tabla de sabiduría de 2 filas y 3 columnas
//! let datos = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let pergamino = Tensor::new(datos, vec![2, 3]);
//!
//! // Multiplicación de pergaminos (matmul)
//! let otro = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2]);
//! let resultado = pergamino.matmul(&otro);
//! assert_eq!(resultado.forma, vec![2, 2]);
//! ```
//!
//! ## Las Hazañas del Paralelismo
//!
//! Varias operaciones usan la lanza de Rayon para partir el trabajo entre múltiples núcleos:
//!
//! - **Multiplicación de matrices**: bloques de caché con filas en paralelo (2-4× más veloz).
//! - **Operaciones elemento a elemento**: iteración paralela sobre los datos.
//! - **Softmax**: cálculo por fila en paralelo.

// Rayon: la cuadrilla de escuderos que trabajan al unísono para acelerar el cómputo.
use rayon::prelude::*;

/// Un pergamino multidimensional para los cálculos del Transformador
///
/// Los tensores guardan sus valores en un `Vec<f32>` contiguo con información de forma
/// y saltos, permitiendo indexación multidimensional eficiente. Todo se almacena en
/// orden de fila mayor (estilo C, como los libros que se leen de izquierda a derecha).
///
/// # Campos
///
/// - `datos`: Hilera plana de valores f32 — las palabras del pergamino.
/// - `forma`: Las dimensiones del cofre, ej. `[2, 3]` para una tabla de 2×3.
/// - `saltos`: Los pasos para navegar la memoria — calculados desde la forma.
///
/// # Disposición en Memoria
///
/// Para la forma `[2, 3]`, los datos se guardan así:
/// `[fila0_col0, fila0_col1, fila0_col2, fila1_col0, fila1_col1, fila1_col2]`
///
/// Los saltos serían `[3, 1]`:
/// - Avanzar una fila = saltar 3 posiciones en los datos.
/// - Avanzar una columna = saltar 1 posición en los datos.
#[derive(Clone, Debug)]
pub struct Tensor {
    /// Todos los valores del tensor en una hilera plana de memoria
    pub datos: Vec<f32>,
    /// Las dimensiones del tensor — la forma del cofre
    pub forma: Vec<usize>,
    /// Los saltos para cada dimensión — el mapa del tesoro
    pub saltos: Vec<usize>,
}

impl Tensor {
    /// Forja un nuevo tensor con los datos y la forma dados
    ///
    /// Como un herrero que da forma al metal, este constructor verifica que el número
    /// de valores coincida con el producto de las dimensiones antes de crear el tensor.
    ///
    /// # Argumentos
    ///
    /// * `datos` - La hilera plana de valores
    /// * `forma` - Las dimensiones del tensor
    ///
    /// # Pánicos
    ///
    /// Si el producto de las dimensiones no coincide con la longitud de los datos,
    /// el molino se detiene y lanza un error.
    pub fn new(datos: Vec<f32>, forma: Vec<usize>) -> Self {
        let tamaño_esperado: usize = forma.iter().product();
        assert_eq!(
            datos.len(),
            tamaño_esperado,
            "¡Desatino! Los datos tienen {} elementos pero la forma {:?} exige {}",
            datos.len(),
            forma,
            tamaño_esperado
        );

        let saltos = Self::calcular_saltos(&forma);
        Self { datos, forma, saltos }
    }

    /// Crea un pergamino en blanco — lleno de ceros como una página sin escribir
    pub fn ceros(forma: Vec<usize>) -> Self {
        let tamaño: usize = forma.iter().product();
        Self::new(vec![0.0; tamaño], forma)
    }

    /// Calcula los saltos a partir de la forma (disposición de fila mayor)
    ///
    /// Para la forma `[d0, d1, d2]`, los saltos son `[d1*d2, d2, 1]`.
    /// Es el mapa que permite saltar de una celda a otra dentro del pergamino plano.
    fn calcular_saltos(forma: &[usize]) -> Vec<usize> {
        let mut saltos = vec![1; forma.len()];
        for i in (0..forma.len().saturating_sub(1)).rev() {
            saltos[i] = saltos[i + 1] * forma[i + 1];
        }
        saltos
    }

    /// Multiplicación de pergaminos (matmul)
    ///
    /// La operación más noble del álgebra lineal: combinar dos tablas de conocimiento
    /// para producir una tercera. Como cuando Don Quijote mezcla el bálsamo de Fierabrás,
    /// cada valor del resultado nace de un producto punto entre filas y columnas.
    ///
    /// # Modos de combate
    ///
    /// - **2D × 2D**: Multiplicación estándar de matrices.
    /// - **4D × 4D**: Multiplicación por lotes para los cálculos de atención.
    ///
    /// # Estrategia del caballero
    ///
    /// - **Matrices pequeñas** (< 1K operaciones): Cálculo secuencial, sin overhead de hilos.
    /// - **Matrices grandes** (≥ 1K operaciones): Algoritmo de bloqueo de caché en paralelo.
    #[inline(always)]
    fn matmul_interno_simd(val_a: f32, b: &[f32], resultado: &mut [f32]) {
        // Bucle sencillo que LLVM convierte en instrucciones SIMD (AVX2/NEON)
        // Cuatro gigantes atacados al mismo tiempo con una sola lanza
        for (r, &val_b) in resultado.iter_mut().zip(b.iter()) {
            *r += val_a * val_b;
        }
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        // === MULTIPLICACIÓN DE MATRICES 2D ===
        if self.forma.len() == 2 && other.forma.len() == 2 {
            assert_eq!(
                self.forma[1], other.forma[0],
                "¡Dimensiones incompatibles! [{}, {}] @ [{}, {}] — las filas del segundo deben coincidir con las columnas del primero",
                self.forma[0], self.forma[1], other.forma[0], other.forma[1]
            );

            let m = self.forma[0];
            let n = other.forma[1];
            let k = self.forma[1];

            // Para la batalla grande, convocamos a los escuderos paralelos
            if m * n * k >= 1_000 {
                return self.matmul_paralelo_bloques(other, m, n, k);
            }

            // Para la escaramuza pequeña, bastamos solos
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

        // === MULTIPLICACIÓN POR LOTES 4D (para la atención multicabeza) ===
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

            // Cada par (lote, cabeza) es un caballero independiente que combate su propia batalla
            resultado
                .par_chunks_mut(sec1 * sec2)
                .enumerate()
                .for_each(|(idx_lc, bloque)| {
                    let b = idx_lc / num_cabezas; // índice de lote
                    let h = idx_lc % num_cabezas; // índice de cabeza de atención

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
            "¡Formas de batalla no soportadas! {:?} @ {:?}",
            self.forma, other.forma
        );
    }

    /// Multiplicación de matrices en paralelo con bloqueo de caché
    ///
    /// El arte de dividir la batalla en bloques que caben en la memoria rápida (caché L1).
    /// Como las tropas de Don Quijote divididas en escuadrones de 8, cada bloque procesa
    /// su porción sin interferir con los demás.
    ///
    /// **Estrategia**: Bloques de 8×8 = 256 bytes — caben bien en la caché L1 (32-64 KB).
    fn matmul_paralelo_bloques(&self, other: &Tensor, m: usize, n: usize, k: usize) -> Tensor {
        // Tamaño del escuadrón: 8×8 elementos caben en la caché L1
        const TAM_BLOQUE: usize = 8;

        let mut resultado = vec![0.0; m * n];

        // Cada hilo recibe TAM_BLOQUE filas y las procesa de forma independiente
        resultado
            .par_chunks_mut(TAM_BLOQUE * n)
            .enumerate()
            .for_each(|(i_bloque, bloque_resultado)| {
                let i_inicio = i_bloque * TAM_BLOQUE;
                let i_fin = (i_inicio + TAM_BLOQUE).min(m);

                for j_inicio in (0..n).step_by(TAM_BLOQUE) {
                    let j_fin = (j_inicio + TAM_BLOQUE).min(n);

                    for k_inicio in (0..k).step_by(TAM_BLOQUE) {
                        let k_fin = (k_inicio + TAM_BLOQUE).min(k);

                        // El bucle interno accede a memoria de forma secuencial — la lanza SIMD actúa aquí
                        for i in i_inicio..i_fin {
                            let desplazamiento_fila = (i - i_inicio) * n;
                            for k_idx in k_inicio..k_fin {
                                let val_a = self.datos[i * k + k_idx];
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

    /// Softmax — la balanza que convierte puntuaciones en probabilidades
    ///
    /// Como el juicio de un sabio que pondera los méritos de cada caballero y distribuye
    /// su favor de forma que todo sume exactamente uno. Usa el truco de restar el máximo
    /// para evitar que los números exploten como gigantes enardecidos.
    ///
    /// ```text
    /// softmax(x)[i] = exp(x[i] - max(x)) / Σ exp(x[j] - max(x))
    /// ```
    pub fn softmax(&self, eje: isize) -> Tensor {
        let eje_pos = if eje < 0 {
            (self.forma.len() as isize + eje) as usize
        } else {
            eje as usize
        };

        // === SOFTMAX 2D POR FILA — el caso más común en la atención ===
        if self.forma.len() == 2 && eje_pos == 1 {
            let filas = self.forma[0];
            let columnas = self.forma[1];

            // Cada fila se procesa de forma independiente — un caballero por fila
            let resultado: Vec<f32> = (0..filas)
                .into_par_iter()
                .flat_map_iter(|i| {
                    let inicio = i * columnas;
                    let fin = inicio + columnas;
                    let fila = &self.datos[inicio..fin];

                    // Restamos el máximo para evitar el desbordamiento del exp()
                    let maximo = fila.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let valores_exp: Vec<f32> = fila.iter().map(|&x| (x - maximo).exp()).collect();

                    // Normalizamos para que la fila sume exactamente 1.0
                    let suma: f32 = valores_exp.iter().sum();
                    valores_exp.into_iter().map(move |val| val / suma)
                })
                .collect();

            return Tensor::new(resultado, self.forma.clone());
        }

        // === RESPALDO: SOFTMAX GLOBAL (menos frecuente) ===
        let maximo = self.datos.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let valores_exp: Vec<f32> = self.datos.iter().map(|&x| (x - maximo).exp()).collect();
        let suma: f32 = valores_exp.iter().sum();
        let resultado = valores_exp.iter().map(|&x| x / suma).collect();
        Tensor::new(resultado, self.forma.clone())
    }

    /// Suma elemento a elemento con broadcasting
    ///
    /// Como añadir sabiduría a cada celda del pergamino. Soporta los patrones
    /// más comunes en los transformadores:
    ///
    /// 1. **Misma forma**: Suma directa elemento a elemento.
    /// 2. **Broadcasting de lote**: `[lote, sec, dim] + [sec, dim]`
    /// 3. **Broadcasting de sesgo**: `[*, n] + [n]` — sumar el sesgo a cada fila.
    pub fn add(&self, other: &Tensor) -> Tensor {
        // Misma forma: la suma más sencilla, como sumar dos listas de monedas
        if self.forma == other.forma {
            let resultado = self.datos.par_iter().zip(&other.datos).map(|(a, b)| a + b).collect();
            return Tensor::new(resultado, self.forma.clone());
        }

        // Broadcasting de lote: [lote, sec, dim] + [sec, dim]
        // El vector más pequeño se repite para cada elemento del lote
        if self.forma.len() == 3 && other.forma.len() == 2 {
            let tam_lote = self.forma[0];
            let longitud_sec = self.forma[1];
            let dim = self.forma[2];

            assert_eq!(other.forma[0], longitud_sec, "La longitud de secuencia debe coincidir");
            assert_eq!(other.forma[1], dim, "La dimensión debe coincidir");

            let resultado: Vec<f32> = (0..tam_lote * longitud_sec * dim)
                .into_par_iter()
                .map(|i| {
                    let s = (i / dim) % longitud_sec;
                    let d = i % dim;
                    self.datos[i] + other.datos[s * dim + d]
                })
                .collect();
            return Tensor::new(resultado, self.forma.clone());
        }

        // Broadcasting de la última dimensión: [*, n] + [n] — para sumar sesgos
        if self.forma.len() > other.forma.len() {
            let ultima_dim = *self.forma.last().unwrap();
            if other.datos.len() == ultima_dim {
                let resultado: Vec<f32> = (0..self.datos.len())
                    .into_par_iter()
                    .map(|i| self.datos[i] + other.datos[i % ultima_dim])
                    .collect();
                return Tensor::new(resultado, self.forma.clone());
            }
        }

        panic!("¡Broadcasting no soportado para suma! {:?} + {:?}", self.forma, other.forma);
    }

    /// Multiplicación elemento a elemento con broadcasting — ver `add()` para los patrones.
    pub fn mul(&self, other: &Tensor) -> Tensor {
        if self.forma == other.forma {
            let resultado = self.datos.par_iter().zip(&other.datos).map(|(a, b)| a * b).collect();
            return Tensor::new(resultado, self.forma.clone());
        }

        if self.forma.len() > other.forma.len() {
            let ultima_dim = *self.forma.last().unwrap();
            if other.datos.len() == ultima_dim {
                let resultado: Vec<f32> = (0..self.datos.len())
                    .into_par_iter()
                    .map(|i| self.datos[i] * other.datos[i % ultima_dim])
                    .collect();
                return Tensor::new(resultado, self.forma.clone());
            }
        }

        panic!("¡Broadcasting no soportado para multiplicación! {:?} * {:?}", self.forma, other.forma);
    }

    /// Resta elemento a elemento con broadcasting
    pub fn sub(&self, other: &Tensor) -> Tensor {
        if self.forma == other.forma {
            let resultado = self.datos.par_iter().zip(&other.datos).map(|(a, b)| a - b).collect();
            return Tensor::new(resultado, self.forma.clone());
        }

        // Broadcasting keepdim: última dimensión = 1
        if self.forma.len() == other.forma.len() {
            let ultima_dim = *self.forma.last().unwrap();
            let ultima_dim_otro = *other.forma.last().unwrap();
            if ultima_dim_otro == 1
                && self.forma[..self.forma.len() - 1] == other.forma[..other.forma.len() - 1]
            {
                let resultado: Vec<f32> = (0..self.datos.len())
                    .into_par_iter()
                    .map(|i| self.datos[i] - other.datos[(i / ultima_dim) * ultima_dim_otro])
                    .collect();
                return Tensor::new(resultado, self.forma.clone());
            }
        }

        panic!("¡Broadcasting no soportado para resta! {:?} - {:?}", self.forma, other.forma);
    }

    /// División elemento a elemento con broadcasting
    pub fn div(&self, other: &Tensor) -> Tensor {
        if self.forma == other.forma {
            let resultado = self.datos.par_iter().zip(&other.datos).map(|(a, b)| a / b).collect();
            return Tensor::new(resultado, self.forma.clone());
        }

        if self.forma.len() == other.forma.len() {
            let ultima_dim = *self.forma.last().unwrap();
            let ultima_dim_otro = *other.forma.last().unwrap();
            if ultima_dim_otro == 1
                && self.forma[..self.forma.len() - 1] == other.forma[..other.forma.len() - 1]
            {
                let resultado: Vec<f32> = (0..self.datos.len())
                    .into_par_iter()
                    .map(|i| self.datos[i] / other.datos[(i / ultima_dim) * ultima_dim_otro])
                    .collect();
                return Tensor::new(resultado, self.forma.clone());
            }
        }

        panic!("¡Broadcasting no soportado para división! {:?} / {:?}", self.forma, other.forma);
    }

    /// Suma un escalar a todos los elementos — como añadir una moneda a cada bolsillo
    pub fn add_scalar(&self, escalar: f32) -> Tensor {
        let resultado = self.datos.par_iter().map(|&x| x + escalar).collect();
        Tensor::new(resultado, self.forma.clone())
    }

    /// Multiplica todos los elementos por un escalar — la escala de un caballero
    pub fn mul_scalar(&self, escalar: f32) -> Tensor {
        let resultado = self.datos.par_iter().map(|&x| x * escalar).collect();
        Tensor::new(resultado, self.forma.clone())
    }

    /// Divide todos los elementos por un escalar
    pub fn div_scalar(&self, escalar: f32) -> Tensor {
        let resultado = self.datos.par_iter().map(|&x| x / escalar).collect();
        Tensor::new(resultado, self.forma.clone())
    }

    /// Raíz cuadrada elemento a elemento — √x para cada valor del pergamino
    pub fn sqrt(&self) -> Tensor {
        let resultado = self.datos.par_iter().map(|&x| x.sqrt()).collect();
        Tensor::new(resultado, self.forma.clone())
    }

    /// Redimensiona el pergamino a una nueva forma sin cambiar sus valores
    ///
    /// Como doblar un mapa de otra manera: el territorio no cambia, solo su presentación.
    pub fn reshape(&self, nueva_forma: &[usize]) -> Tensor {
        let nuevo_tamano: usize = nueva_forma.iter().product();
        assert_eq!(
            self.datos.len(), nuevo_tamano,
            "¡No se puede redimensionar! La cantidad de elementos no coincide"
        );
        Tensor::new(self.datos.clone(), nueva_forma.to_vec())
    }

    /// Transpone dos dimensiones — intercambia filas por columnas como un escribano
    pub fn transpose(&self, dim1: isize, dim2: isize) -> Tensor {
        let ndim = self.forma.len() as isize;
        let d1 = if dim1 < 0 { ndim + dim1 } else { dim1 } as usize;
        let d2 = if dim2 < 0 { ndim + dim2 } else { dim2 } as usize;

        let mut nueva_forma = self.forma.clone();
        nueva_forma.swap(d1, d2);

        // Para matrices 2D, transposición simple y eficiente
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

        // Para tensores 4D transponiendo las dos últimas dimensiones (atención multicabeza)
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

        // Transposición general con remapeo de saltos
        let pasos_antiguos = &self.saltos;
        let mut nuevos_pasos = pasos_antiguos.clone();
        nuevos_pasos.swap(d1, d2);

        let tamano_total = self.datos.len();
        let mut resultado = vec![0.0; tamano_total];

        for (i, item) in resultado.iter_mut().enumerate().take(tamano_total) {
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

    /// Aplica la máscara causal — tapa las posiciones futuras con -infinito
    ///
    /// Como el yelmo de Mambrino que protege la cabeza de ver lo que aún no ha sucedido.
    /// Las posiciones tapadas quedan a -1e9, que tras el softmax se convierten en 0%.
    pub fn masked_fill(&self, mascara: &Tensor, valor: f32) -> Tensor {
        if self.forma == mascara.forma {
            let resultado = self.datos.par_iter().zip(&mascara.datos)
                .map(|(&x, &m)| if m != 0.0 { valor } else { x })
                .collect();
            return Tensor::new(resultado, self.forma.clone());
        }

        // Broadcasting: [lote, num_cabezas, sec, sec] con máscara [sec, sec]
        if self.forma.len() == 4 && mascara.forma.len() == 2 {
            let lote = self.forma[0];
            let num_cabezas = self.forma[1];
            let sec = self.forma[2];

            assert_eq!(mascara.forma[0], sec);
            assert_eq!(mascara.forma[1], sec);

            let mut resultado = Vec::with_capacity(self.datos.len());
            for _b in 0..lote {
                for _h in 0..num_cabezas {
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

        panic!("¡Broadcasting no soportado para masked_fill! {:?} con máscara {:?}", self.forma, mascara.forma);
    }

    /// Calcula la media a lo largo de un eje — el promedio del saber acumulado
    pub fn mean(&self, eje: isize, mantener_dim: bool) -> Tensor {
        let eje_pos = if eje < 0 {
            (self.forma.len() as isize + eje) as usize
        } else {
            eje as usize
        };

        if self.forma.len() == 2 && eje_pos == 1 {
            let filas = self.forma[0];
            let columnas = self.forma[1];
            let resultado: Vec<f32> = (0..filas).into_par_iter().map(|i| {
                let inicio = i * columnas;
                let suma: f32 = self.datos[inicio..inicio + columnas].iter().sum();
                suma / columnas as f32
            }).collect();
            let nueva_forma = if mantener_dim { vec![filas, 1] } else { vec![filas] };
            return Tensor::new(resultado, nueva_forma);
        }

        if self.forma.len() == 3 && eje_pos == 2 {
            let lote = self.forma[0];
            let sec = self.forma[1];
            let dim = self.forma[2];
            let resultado: Vec<f32> = (0..lote * sec).into_par_iter().map(|i| {
                let inicio = i * dim;
                let suma: f32 = self.datos[inicio..inicio + dim].iter().sum();
                suma / dim as f32
            }).collect();
            let nueva_forma = if mantener_dim { vec![lote, sec, 1] } else { vec![lote, sec] };
            return Tensor::new(resultado, nueva_forma);
        }

        panic!("¡Operación mean no soportada para la forma {:?}!", self.forma);
    }

    /// Calcula la varianza a lo largo de un eje — la dispersión del conocimiento
    pub fn var(&self, eje: isize, mantener_dim: bool) -> Tensor {
        if eje == -1 || eje as usize == self.forma.len() - 1 {
            let ultima_dim = self.forma[self.forma.len() - 1];
            let tamano_exterior = self.datos.len() / ultima_dim;

            let mut resultado = Vec::new();
            for i in 0..tamano_exterior {
                let inicio = i * ultima_dim;
                let segmento = &self.datos[inicio..inicio + ultima_dim];
                let media: f32 = segmento.iter().sum::<f32>() / ultima_dim as f32;
                let varianza = segmento.iter().map(|&x| (x - media).powi(2)).sum::<f32>() / ultima_dim as f32;
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

        panic!("¡Operación var no soportada para la forma {:?} con eje {}!", self.forma, eje);
    }

    /// Crea un pergamino con enteros secuenciales [inicio, fin)
    pub fn arange(inicio: usize, fin: usize) -> Tensor {
        let datos: Vec<f32> = (inicio..fin).map(|i| i as f32).collect();
        let longitud = datos.len();
        Tensor::new(datos, vec![longitud])
    }

    /// Concatena dos pergaminos a lo largo de la primera dimensión
    ///
    /// Como encuadernar dos manuscritos en un único volumen más grueso.
    pub fn concat(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.forma.len(), other.forma.len(),
            "Los pergaminos deben tener el mismo número de dimensiones"
        );
        for i in 1..self.forma.len() {
            assert_eq!(self.forma[i], other.forma[i],
                "Todas las dimensiones excepto la primera deben coincidir");
        }

        let mut nueva_forma = self.forma.clone();
        nueva_forma[0] += other.forma[0];

        let mut datos = Vec::with_capacity(self.datos.len() + other.datos.len());
        datos.extend_from_slice(&self.datos);
        datos.extend_from_slice(&other.datos);

        Tensor::new(datos, nueva_forma)
    }
}
