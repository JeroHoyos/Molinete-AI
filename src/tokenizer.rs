//! Tokenizador Byte Pair Encoding (BPE)
//!
//! Este módulo implementa la tokenización BPE desde cero. BPE es el método
//! de tokenización estándar utilizado por GPT-2, GPT-3 y muchos otros
//! modelos de lenguaje.
//!
//! ## Cómo funciona BPE
//!
//! 1. **Comienza con codificación a nivel de byte**: 256 tokens base (uno por cada valor de byte: 0–255)
//! 2. **Cuenta pares adyacentes**: Encuentra el par de bytes adyacentes más común en el corpus
//! 3. **Fusiona el par más frecuente**: Crea un nuevo token que representa ese par
//! 4. **Repite**: Continúa hasta que el vocabulario alcance el tamaño objetivo
//!
//! ## Ejemplo
//!
//! Dado el corpus: "hello hello world"
//! - Iteración 1: Par más común = ('l','l') → se fusiona en el token 256: "he[256]o he[256]o world"
//! - Iteración 2: Par más común = ('h','e') → se fusiona en el token 257: "[257][256]o [257][256]o world"
//! - Iteración 3: Par más común = ('[257]','[256]') → se fusiona en el token 258: "[258]o [258]o world"
//! - Continúa hasta alcanzar vocab_size...
//!
//! ## Por qué la tokenización es importante
//!
//! Los modelos de lenguaje nunca ven texto en bruto: ven IDs de tokens.
//! Esta transformación fundamental explica muchos comportamientos “extraños”:
//!
//! - **No pueden contar letras con fiabilidad**: El modelo ve tokens como ["run", "ning"],
//!   no caracteres individuales
//! - **Sensibilidad a los espacios**: " hello" y "hello" se tokenizan de manera diferente
//! - **Dificultad con palabras raras**: Las palabras poco comunes se dividen en fragmentos desconocidos
//! - **Invertir texto es difícil**: Las operaciones a nivel de carácter son complicadas cuando se trabaja con tokens
//!
//! ## Notas de implementación
//!
//! Esta implementación incluye varias optimizaciones para uso práctico:
//!
//! - **Conteo paralelo de pares**: Usa Rayon para contar pares de bytes en múltiples núcleos de CPU,
//!   proporcionando una mejora de velocidad de 2–3x en sistemas multinúcleo
//! - **Codificación paralela**: Los textos grandes se dividen en fragmentos y se codifican en paralelo
//! - **Entrenamiento con muestras**: Para vocabularios muy grandes (>2000 tokens), el entrenamiento
//!   usa un subconjunto del corpus para aprender patrones más rápidamente
//! - **Aplicación eficiente de fusiones**: Construye nuevos vectores de tokens en lugar de eliminar
//!   elementos en el lugar, evitando complejidad O(n²)
//!
//! Estas optimizaciones son necesarias para hacer que el entrenamiento sea práctico en CPU,
//! pero el algoritmo subyacente sigue siendo el BPE estándar.

/// Librería para paralelizar tareas.
use rayon::prelude::*;
/// Librería para serialización y deserialización de datos.
use serde::{Deserialize, Serialize};
/// Librería para la estructura de datos Hashmap.
use std::collections::HashMap;
/// Librería para operaciones de sistema de archivos.
use std::fs;
/// Librería para manejar rutas de archivos.
use std::path::Path;

/// A Byte Pair Encoding tokenizer
///
/// The tokenizer maintains a vocabulary of tokens (strings mapped to IDs) and
/// a sequence of merge rules learned during training. The merges are applied
/// in order during encoding to convert text into token IDs.

/// Genera automáticamente las implementaciones de los traits para esta struct.
#[derive(Clone, Serialize, Deserialize)]
pub struct BPETokenizer {
    /// Mapea cadenas de tokens a sus identificadores enteros.
    /// Siempre comienza con 256 tokens base para codificación a nivel de byte.
    vocab: HashMap<String, usize>,

    /// Secuencia de reglas de fusión aprendidas durante el entrenamiento.
    /// Cada fusión combina dos tokens en un nuevo token.
    /// Se aplican en orden durante la codificación.
    merges: Vec<(String, String)>,

    /// Token desconocido.
    #[allow(dead_code)] // No muestra warning si esta variable no se usa.
    unk_token: String,
}

impl BPETokenizer {
    /// Crea un nuevo tokenizador con vocabulario base.
    ///
    /// Inicializa el tokenizador con 256 tokens base que representan cada posible
    /// valor de byte (0–255). Estos se codifican como cadenas hexadecimales
    /// como "<00>", "<01>", ..., "<ff>".
    ///
    /// # Argumentos
    ///
    /// * `vocab_size` - Tamaño objetivo del vocabulario (actualmente no se utiliza en el constructor;
    ///   el tamaño real del vocabulario se determina durante el entrenamiento).
    ///
    /// # Retorna
    ///
    /// Un nuevo tokenizador con 256 tokens base y sin fusiones aprendidas.
    pub fn new(_vocab_size: usize) -> Self {
        let mut vocab = HashMap::new();

        // Inicializa con tokens a nivel de byte (256 tokens base)
        // Cada valor de byte se representa como una cadena hexadecimal: <00>, <01>, ..., <ff>
        for byte in 0..=255 {
            vocab.insert(format!("<{:02x}>", byte), vocab.len());
        }

        Self {
            vocab,
            merges: Vec::new(),
            unk_token: "<unk>".to_string(),
        }
    }

/// Entrena el modelo BPE sobre un corpus de texto.
///
/// Aprende reglas de fusión encontrando e integrando iterativamente
/// los pares de tokens adyacentes más frecuentes. Esto construye un
/// vocabulario partiendo de los 256 tokens base a nivel de byte
/// hasta alcanzar el tamaño objetivo del vocabulario.
///
/// # Argumentos
///
/// * `text` - Corpus de entrenamiento (típicamente varios MB de texto).
/// * `vocab_size` - Tamaño objetivo del vocabulario (valores comunes: 512, 1024, 2048, 5000).
///
/// # Optimización de rendimiento
///
/// Para vocabularios grandes (>2000 tokens), este método entrena sobre
/// una muestra de 200KB del corpus en lugar del texto completo.
/// Esto es mucho más rápido y aun así aprende eficazmente los patrones
/// más frecuentes. Las fusiones aprendidas luego están disponibles
/// para codificar el corpus completo.
///
/// # Ejemplo
///
/// ```rust
/// use molineteai::BPETokenizer;
///
/// // Entrenar sobre un corpus de texto de ejemplo
/// let text = "hello world hello rust hello world";
/// let mut tokenizer = BPETokenizer::new(300);
/// tokenizer.train(&text, 300);
///
/// // Verificar que el entrenamiento funcionó
/// assert!(tokenizer.vocab_size() > 256);
/// ```
    pub fn train(&mut self, text: &str, vocab_size: usize) {
        // Si el tamaño objetivo del vocabulario es 256 o menos, ya hemos terminado.
        if vocab_size <= 256 {
            return;
        }

        println!("Entrenando el tokenizador BPE...");
        println!("  Tamaño inicial del vocabulario: {}", self.vocab.len());
        println!("  Tamaño objetivo del vocabulario: {}", vocab_size);
        println!("  Tamaño del corpus: {} bytes", text.len());

        // Determina la cantidad de fusiones necesarias.
        let num_merges = vocab_size - 256;
        // Optimización: Para vocabularios grandes (>2000 merges), entrenar sobre una muestra más pequeña.
        // 200KB suelen ser suficientes para capturar los patrones más frecuentes del lenguaje
        const MAX_SAMPLE_SIZE: usize = 200_000;
        let training_text = if num_merges > 2000 && text.len() > MAX_SAMPLE_SIZE {
            // Ajusta el tamaño al límite de carácter UTF-8 válido
            let sample_size = text.floor_char_boundary(MAX_SAMPLE_SIZE);

            println!(
                "  Usando {}KB de muestra de entrenamiento para mayor velocidad (captura patrones frecuentes rápidamente).",
                sample_size / 1000
            );
            // Se usa solo el prefijo del texto como muestra de entrenamiento.
            &text[..sample_size]
        } else {
            // Si el vocabulario es pequeño o el texto ya es corto, se entrena sobre el corpus completo.
            text
        };
        
        // Convierte el texto de entrenamiento a tokens a nivel de byte
        // Cada byte se convierte en un token como "<00>", "<01>", etc.
        let mut tokens: Vec<String> = training_text
            .bytes() // Convierte el texto a bytes
            .map(|b| format!("<{:02x}>", b)) // Formatea cada byte como un token hexadecimal
            .collect(); // Lo convierte en un vector de tokens

        // Buffer auxiliar para evitar realocaciones dinámicas durante los merges (Double Buffer Optimization).
        let mut new_tokens = Vec::with_capacity(tokens.len());

        // Itera hasta aprender el número deseado de fusiones o hasta que no queden pares para fusionar.
        for merge_idx in 0..num_merges {
        // === CONTEO DE PARES EN PARALELO ===
        // Este es el cuello de botella en rendimiento, así que lo paralelizamos.
            
            // Tamaño de chunk: El máximo entre 50,000 tokens o el tamaño total dividido por el número de hilos disponibles.
            let chunk_size = 50_000.max(tokens.len() / rayon::current_num_threads().max(1));
            
            // Cuenta todos los pares adyacentes en paralelo a través de los chunks
            let pair_counts: HashMap<(String, String), usize> = tokens
                .par_chunks(chunk_size)
                .enumerate()
                .fold(HashMap::new, |mut local_counts, (chunk_idx, chunk)| {
                    // Cuenta los pares adyacentes dentro del chunk
                    for window in chunk.windows(2) { // Crea ventanas de tamaño 2 para obtener pares adyacentes
                        let pair = (window[0].clone(), window[1].clone()); // Crea el par de tokens adyacentes
                        *local_counts.entry(pair).or_insert(0) += 1; // Incrementa el conteo para este par si existe, o lo inicializa a 1 si no existe
                    }

                    // Manejar los límites de los chunks: contar el par que se extiende al siguiente chunk
                    if chunk_idx * chunk_size + chunk.len() < tokens.len() { // Verifica si hay un siguiente chunk
                        if let Some(last) = chunk.last() { // Obtiene el último token del chunk actual
                            if let Some(next) = tokens.get(chunk_idx * chunk_size + chunk.len()) { // Obtiene el primer token del siguiente chunk
                                let pair = (last.clone(), next.clone()); // Crea el par que cruza el límite del chunk
                                *local_counts.entry(pair).or_insert(0) += 1; // Incrementa el conteo para este par si existe, o lo inicializa a 1 si no existe
                            }
                        }
                    }

                    local_counts // Retorna el conteo local para este chunk
                })
                .reduce(HashMap::new, |mut a, b| {
                    // Combina los conteos de pares de todos los chunks
                    for (pair, count) in b {
                        *a.entry(pair).or_insert(0) += count;
                    }
                    a
                });
            
            // Si no se encontraron pares, el entrenamiento está completo (no hay más fusiones posibles).
            if pair_counts.is_empty() {
                break;
            }

            // === DESEMPATE DETERMINISTA ===
            // La iteración de HashMap es aleatoria. Para garantizar exactamente las mismas fusiones en cada ejecución,
            // ordenamos los pares.
            // Orden primario: Conteo (descendente)
            // Orden secundario: Cadenas de los tokens (ascendente lexicográfico)
            let mut pairs: Vec<((String, String), usize)> = pair_counts.into_iter().collect(); // Convierte el HashMap en un vector de pares y conteos
            pairs.sort_by(|a, b| {
                b.1.cmp(&a.1) // Conteo descendente
                    .then_with(|| a.0.cmp(&b.0)) // String ascendente
            });

            // El par más frecuente es el primero después de ordenar
            let (best_pair, count) = pairs[0].clone();

            // Crea el nuevo token fusionado a partir del par más frecuente
            let new_token = format!("{}{}", best_pair.0, best_pair.1);

            // Agrega el nuevo token al vocabulario con el siguiente ID disponible
            self.vocab.insert(new_token.clone(), self.vocab.len());
            self.merges.push(best_pair.clone());

            // === APLICAR LA FUSIÓN AL CORPUS ===
            // Reconstruimos la lista de tokens aplicando la nueva fusión.
            // Aquí utilizamos la estrategia de doble búfer para reutilizar memoria.
            new_tokens.clear();

            let mut i = 0;
            while i < tokens.len() {
                // Si encontramos el par, lo reemplazamos con el token fusionado
                if i < tokens.len() - 1 && tokens[i] == best_pair.0 && tokens[i + 1] == best_pair.1
                {
                    new_tokens.push(new_token.clone());
                    i += 2; // Saltamos ambos tokens del par
                } else {
                    new_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }

            // Intercambiamos los búferes para que `tokens` tenga los datos actualizados para la siguiente iteración
            std::mem::swap(&mut tokens, &mut new_tokens);

            // Imprime el progreso cada 50 fusiones para no saturar la salida
            if merge_idx % 50 == 0 {
                println!(
                    "Fusión {}/{}: {:?} (conteo: {}) -> tamaño del vocabulario: {}",
                    merge_idx + 1,
                    num_merges,
                    best_pair,
                    count,
                    self.vocab.len()
                );
            }
        }

        println!("¡Entrenamiento completo! Tamaño final del vocabulario: {}", self.vocab.len());
        println!("Se aprendieron {} fusiones\n", self.merges.len());
    }

    /// Convierte texto en tokens a nivel de byte.
    ///
    /// Función interna que convierte cada byte a su representación hexadecimal.
    ///
    /// # Argumentos
    ///
    /// * `text` - Texto de entrada a convertir
    ///
    /// # Retorna
    ///
    /// Vector de cadenas de tokens como ["<68>", "<65>", "<6c>", "<6c>", "<6f>"] para "hello"
    fn byte_encode(&self, text: &str) -> Vec<String> {
        text.bytes().map(|b| format!("<{:02x}>", b)).collect()
    }

    /// Codifica texto a IDs de tokens
    ///
    /// Convierte el texto en una secuencia de IDs de tokens primero
    /// transformándolo a tokens a nivel de byte y luego aplicando
    /// las reglas de fusión aprendidas en orden.
    ///
    /// # Argumentos
    ///
    /// * `text` - Texto de entrada a codificar
    ///
    /// # Retorna
    ///
    /// Vector de IDs de tokens
    ///
    /// # Optimización de rendimiento
    ///
    /// Para textos grandes (>200KB), este método divide el texto en bloques
    /// y los codifica en paralelo, proporcionando una mejora significativa
    /// de velocidad en sistemas multinúcleo.
    ///
    /// # Ejemplo
    ///
    /// ```rust
    /// # use molineteai::BPETokenizer;
    /// # let mut tokenizer = BPETokenizer::new(256);
    /// let ids = tokenizer.encode("Hello, world!");
    /// println!("IDs de tokens: {:?}", ids);
    /// ```
    pub fn encode(&self, text: &str) -> Vec<usize> {
        // Umbral para paralelismo
        const CHUNK_SIZE: usize = 100_000; // bytes por chunk

        if text.len() > CHUNK_SIZE * 2 { // Decidimos si paralelizar basándonos en el tamaño del texto
        // === CODIFICACIÓN PARALELA PARA TEXTOS GRANDES ===
        // Dividir el texto en fragmentos no superpuestos para codificarlos en paralelo
        //
        // Nota: Esto significa que no aplicamos fusiones (merges) a través de los límites entre fragmentos.
        // Esto es aceptable porque:
        // 1. Los límites son poco frecuentes (cada 100KB)
        // 2. El impacto en la compresión es insignificante
        // 3. La corrección está garantizada (no hay tokens duplicados ni faltantes)

            let mut chunks = Vec::new();
            let mut start = 0;

            while start < text.len() {
                // Calcula el final del chunk.
                let mut end = (start + CHUNK_SIZE).min(text.len());

                // Ajustar el final para no cortar en medio de un carácter UTF-8
                while end < text.len() && !text.is_char_boundary(end) {
                    end += 1;
                }

                chunks.push(&text[start..end]);

                // Moverse al siguiente chunk
                start = end;
            }

            // Codificar cada chunk en paralelo
            let encoded_chunks: Vec<Vec<usize>> = chunks
                .par_iter()
                .map(|chunk| self.encode_sequential(chunk))
                .collect();

            // Concatenar los resultados de los chunks codificados
            let mut result = Vec::new();
            for chunk in encoded_chunks {
                result.extend_from_slice(&chunk);
            }
            result
        } else {
            // Textos pequeños: usar la versión secuencial (evita la sobrecarga del paralelismo)
            self.encode_sequential(text)
        }
    }

    /// Codifica texto de forma secuencial (versión no paralela)
    ///
    /// Método interno que realiza la codificación aplicando las reglas de fusión.
    ///
    /// # Argumentos
    ///
    /// * `text` - Texto de entrada a codificar
    ///
    /// # Retorna
    ///
    /// Vector de identificadores (IDs) de tokens
    fn encode_sequential(&self, text: &str) -> Vec<usize> {
        let mut tokens = self.byte_encode(text);
        // Buffer para evitar realocaciones dinámicas durante la aplicación de fusiones.
        let mut new_tokens = Vec::with_capacity(tokens.len());

        for (pair_a, pair_b) in &self.merges {
            let merged = format!("{}{}", pair_a, pair_b);

            // Ruta rápida: si el par ni siquiera aparece en los tokens, se omite esta regla de fusión.
            // Es útil si N es grande y el par es poco frecuente.)

            new_tokens.clear();
            let mut i = 0;
            while i < tokens.len() {
                if i < tokens.len() - 1 && tokens[i] == *pair_a && tokens[i + 1] == *pair_b {
                    new_tokens.push(merged.clone());
                    i += 2;
                } else {
                    new_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }
            // Intercambiamos los búferes para que `tokens` tenga los datos actualizados para la siguiente iteración
            std::mem::swap(&mut tokens, &mut new_tokens);
        }

        tokens
            .iter()
            .map(|token| *self.vocab.get(token).unwrap_or(&0))
            .collect()
    }

    /// Decodifica IDs de tokens de vuelta a texto
    ///
    /// Convierte una secuencia de IDs de tokens nuevamente en el texto original
    /// buscando cada ID en el vocabulario y procesando los bytes codificados en hexadecimal.
    ///
    /// # Argumentos
    ///
    /// * `ids` - IDs de tokens a decodificar
    ///
    /// # Retorna
    ///
    /// Cadena de texto decodificada
    ///
    /// # Ejemplo
    ///
    /// ```rust
    /// # use molineteai::BPETokenizer;
    /// # let tokenizer = BPETokenizer::new(256);
    /// # let ids = tokenizer.encode("Hello!");
    /// let text = tokenizer.decode(&ids);
    /// assert_eq!(text, "Hello!");
    /// ```
    pub fn decode(&self, ids: &[usize]) -> String {
        // Crea un mapa de vocabulario inverso (ID -> cadena de token)
        let id_to_token: HashMap<usize, String> = self
            .vocab
            .iter()
            .map(|(token, id)| (*id, token.clone()))
            .collect();

        // Convierta cada ID de token a su cadena correspondiente usando el mapa inverso
        let tokens: Vec<String> = ids
            .iter()
            .filter_map(|id| id_to_token.get(id).cloned())
            .collect();

        // Une los tokens en una sola cadena y decodifíquela usando decode_token
        let merged = tokens.join("");
        self.decode_token(&merged)
    }

    /// Obtiene el tamaño del vocabulario actual.
    ///
    /// # Retorna
    ///
    /// Numero total de tokens en el vocabulario (256 tokens base + número de fusiones aprendidas).
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Guarda el tokenizador en un archivo JSON
    ///
    /// Serializa el tokenizador (vocabulario y reglas de fusión)
    /// en un archivo JSON para poder cargarlo posteriormente.
    ///
    /// # Argumentos
    ///
    /// * `path` - Ruta donde se guardará el tokenizador
    ///
    /// # Retorna
    ///
    /// `Result` que indica éxito o error en la operación
    ///
    /// # Ejemplo
    ///
    /// ```rust,no_run
    /// # use molineteai::BPETokenizer;
    /// # let tokenizer = BPETokenizer::new(256);
    /// tokenizer.save("tokenizer.json").expect("Error al guardar");
    /// ```
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Carga un tokenizador desde un archivo JSON
    ///
    /// Deserializa un tokenizador previamente guardado desde un archivo JSON.
    ///
    /// # Argumentos
    ///
    /// * `path` - Ruta del archivo que contiene el tokenizador
    ///
    /// # Retorna
    ///
    /// `Result` que contiene el tokenizador cargado o un error en caso de fallo
    ///
    /// # Ejemplo
    ///
    /// ```rust,no_run
    /// use molineteai::BPETokenizer;
    ///
    /// let tokenizer = BPETokenizer::load("tokenizer.json")
    ///     .expect("Error al cargar el tokenizador");
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let json = fs::read_to_string(path)?;
        let tokenizer: BPETokenizer = serde_json::from_str(&json)?;
        Ok(tokenizer)
    }

    /// Obtiene estadísticas del tokenizador
    ///
    /// # Retorna
    ///
    /// Estructura `TokenizerStats` con información del vocabulario
    pub fn stats(&self) -> TokenizerStats {
        TokenizerStats {
            vocab_size: self.vocab.len(),
            num_merges: self.merges.len(),
            base_tokens: 256,
        }
    }

    /// Analiza el vocabulario y muestra información detallada
    ///
    /// Imprime información detallada sobre el vocabulario del tokenizador, incluyendo:
    /// - Composición de los tokens (base vs. aprendidos)
    /// - Muestra de tokens aprendidos
    /// - Análisis de compresión sobre un texto de ejemplo
    /// - Ejemplos de tokenización
    ///
    /// Esto es útil para comprender qué ha aprendido el tokenizador.
    ///
    /// # Argumentos
    ///
    /// * `sample_text` - Texto utilizado para el análisis de compresión y los ejemplos de tokenización
    pub fn analyze_vocabulary(&self, sample_text: &str) {
        println!("\n=== Vocabulary Analysis ===\n");

        // Encontrar tokens legibles (tokens fusionados, no solo bytes base)
        let mut readable_tokens: Vec<(String, usize)> = self
            .vocab
            .iter()
            .filter(|(token, _)| !token.starts_with('<') || token.len() > 4)
            .map(|(token, id)| (token.clone(), *id))
            .collect();

        // Ordenar por ID de token (refleja aproximadamente el orden de las fusiones durante el entrenamiento)
        readable_tokens.sort_by_key(|(_, id)| *id);

        // Desglose del tipo de token de visualización
        let base_tokens = 256;
        let merged_tokens = self.vocab.len() - base_tokens;
        println!("Token Composition:");
        println!("  Base tokens (bytes): {}", base_tokens);
        println!("  Learned merges: {}", merged_tokens);
        println!("  Total vocabulary: {}\n", self.vocab.len());

        // Muestra ejemplos de los tokens aprendidos
        println!("Sample of Learned Tokens (first 30):");
        let display_count = 30.min(readable_tokens.len());
        for (token, id) in readable_tokens.iter().take(display_count) {
            // Try to decode token for display
            let decoded = self.decode_token(token);
            if decoded.len() <= 20 && !decoded.is_empty() {
                println!("  [{}] \"{}\"", id, decoded);
            }
        }

        // Analiza la compresión en un texto de ejemplo
        if !sample_text.is_empty() {
            println!("\nCompression Analysis (on sample):");
            let sample_chars: String = sample_text.chars().take(10000).collect();
            let tokens = self.encode(&sample_chars);
            let char_count = sample_chars.len();
            let token_count = tokens.len();
            let compression_ratio = char_count as f32 / token_count as f32;

            println!("  Sample size: {} characters", char_count);
            println!("  Token count: {} tokens", token_count);
            println!("  Compression ratio: {:.2}x", compression_ratio);
            println!("  Avg chars per token: {:.1}", compression_ratio);
        }

        // Mostrar ejemplos de tokenización
        println!("\nEjemplos de Tokenización:");
        let examples = vec![
            "En un lugar de la Mancha",
            "De cuyo nombre no quiero acordarme",
            "No ha mucho tiempo que vivía",
            "La libertad, Sancho, es uno de los más preciosos dones",
        ];

        for example in examples {
            let tokens = self.encode(example);
            let token_strs: Vec<String> = tokens
                .iter()
                .map(|&id| {
                    // Encuentra la cadena de token correspondiente al ID y decodifíquela para mostrarla
                    self.vocab
                        .iter()
                        .find(|(_, v)| **v == id)
                        .map(|(k, _)| self.decode_token(k))
                        .unwrap_or_else(|| "?".to_string())
                })
                .collect();
            println!(
                "  \"{}\" -> {} tokens: [{}]",
                example,
                tokens.len(),
                token_strs.join("|")
            );
        }

        println!("\n{}\n", "=".repeat(60));
    }

    /// Función auxiliar para decodificar cadenas de tokens codificadas en hexadecimal
    /// y convertirlas nuevamente en texto legible
    ///
    /// Analiza una cadena que contiene bytes codificados en hexadecimal como
    /// "<68><65><6c><6c><6f>" y los convierte nuevamente en texto UTF-8.
    /// Este método maneja tanto tokens individuales como secuencias
    /// concatenadas de múltiples tokens.
    ///
    /// # Argumentos
    ///
    /// * `token` - Cadena del token (o tokens concatenados) a decodificar
    ///
    /// # Retorna
    ///
    /// Representación en texto decodificada
    fn decode_token(&self, token: &str) -> String {
        // Analiza los bytes codificados en hexadecimal y los convierte nuevamente en texto UTF-8
        let mut bytes = Vec::new();
        let mut chars = token.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '<' {
                // Analiza un byte en hexadecimal: <XX> -> valor del byte
                let mut hex_str = String::new();
                while let Some(&next_ch) = chars.peek() {
                    if next_ch == '>' {
                        chars.next(); // consumir '>'
                        break;
                    }
                    hex_str.push(chars.next().unwrap());
                }

                // Convertir la cadena hexadecimal a byte y almacenarlo
                if let Ok(byte) = u8::from_str_radix(&hex_str, 16) {
                    bytes.push(byte);
                }
            }
            // Ignorar cualquier carácter que no sea parte del formato hexadecimal
            // (no debería haber ninguno en tokens válidos)
        }

        // Convertir los bytes recolectados a cadena UTF-8
        String::from_utf8_lossy(&bytes).to_string()
    }
    }

/// Estadísticas sobre el vocabulario de un tokenizador
#[derive(Debug)]
pub struct TokenizerStats {
    /// Tamaño total del vocabulario (tokens base + fusiones aprendidas)
    pub vocab_size: usize,
    /// Número de reglas de fusión aprendidas
    pub num_merges: usize,
    /// Número de tokens base (siempre 256 para BPE a nivel de byte)
    pub base_tokens: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_token_single_byte() {
        let tokenizer = BPETokenizer::new(256);

        // Token de un solo byte: 'h' = 0x68
        let result = tokenizer.decode_token("<68>");
        assert_eq!(result, "h");
    }

    #[test]
    fn test_decode_token_multiple_bytes() {
        let tokenizer = BPETokenizer::new(256);

        // Múltiples bytes: "hello"
        let result = tokenizer.decode_token("<68><65><6c><6c><6f>");
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_decode_token_with_space() {
        let tokenizer = BPETokenizer::new(256);

        // "hi " (con espacio, 0x20)
        let result = tokenizer.decode_token("<68><69><20>");
        assert_eq!(result, "hi ");
    }

    #[test]
    fn test_decode_token_utf8_multibyte() {
        let tokenizer = BPETokenizer::new(256);

        // "é" en UTF-8 es [0xc3, 0xa9]
        let result = tokenizer.decode_token("<c3><a9>");
        assert_eq!(result, "é");
    }

    #[test]
    fn test_decode_token_empty() {
        let tokenizer = BPETokenizer::new(256);

        let result = tokenizer.decode_token("");
        assert_eq!(result, "");
    }

    #[test]
    fn test_decode_basic() {
        let tokenizer = BPETokenizer::new(256);

        // Crear una prueba simple: codificar "hello" como bytes individuales
        let text = "hello";
        let ids = tokenizer.encode(text);
        let decoded = tokenizer.decode(&ids);

        assert_eq!(decoded, text);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let tokenizer = BPETokenizer::new(256);

        let test_cases = vec![
            "hello",
            "Hello, world!",
            "To be, or not to be",
            "123 456 789",
            "special chars: !@#$%^&*()",
            "newline\nand\ttab",
            "UTF-8: café, naïve, 日本語",
        ];

        for text in test_cases {
            let encoded = tokenizer.encode(text);
            let decoded = tokenizer.decode(&encoded);
            assert_eq!(decoded, text, "Falló el roundtrip para: {}", text);
        }
    }

    #[test]
    fn test_encode_decode_with_merges() {
        // Crear tokenizador y entrenarlo
        let mut tokenizer = BPETokenizer::new(300);
        let training_text = "hello hello world world hello";
        tokenizer.train(training_text, 300);

        // Verificar que encode/decode siga funcionando después del entrenamiento
        let test_text = "hello world";
        let encoded = tokenizer.encode(test_text);
        let decoded = tokenizer.decode(&encoded);

        assert_eq!(decoded, test_text);
    }

    #[test]
    fn test_decode_token_consistency_with_decode() {
        let tokenizer = BPETokenizer::new(256);

        // Verificar que decode_token produce el mismo resultado que decode para un token individual
        let token_str = "<68><65><6c><6c><6f>"; // "hello"

        // Decodificación directa
        let direct_result = tokenizer.decode_token(token_str);

        // Simulación del comportamiento de decode
        let simulated_result = tokenizer.decode_token(token_str);

        assert_eq!(direct_result, simulated_result);
        assert_eq!(direct_result, "hello");
    }

    #[test]
    fn test_decode_token_concatenated() {
        let tokenizer = BPETokenizer::new(256);

        // Múltiples tokens concatenados (lo que hace decode antes de llamar a decode_token)
        let concatenated = "<68><65><6c><6c><6f><20><77><6f><72><6c><64>";
        let result = tokenizer.decode_token(concatenated);

        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_vocab_size() {
        let tokenizer = BPETokenizer::new(256);
        assert_eq!(tokenizer.vocab_size(), 256);

        let mut tokenizer2 = BPETokenizer::new(512);
        tokenizer2.train("hello hello world", 512);

        // Nota: el tamaño real depende de cuántos pares únicos existan en el corpus
        // Un corpus pequeño no alcanzará el tamaño objetivo, así que solo verificamos que aumente
        assert!(tokenizer2.vocab_size() > 256);
        assert!(tokenizer2.vocab_size() <= 512);
    }

    #[test]
    fn test_base_vocab_coverage() {
        let tokenizer = BPETokenizer::new(256);

        // Todos los valores posibles de byte deberían poder codificarse
        for byte in 0u8..=255u8 {
            let text = String::from_utf8(vec![byte]).unwrap_or_else(|_| {
                // Para UTF-8 inválido, crear cadena usando from_utf8_lossy
                String::from_utf8_lossy(&[byte]).to_string()
            });

            let encoded = tokenizer.encode(&text);
            let decoded = tokenizer.decode(&encoded);

            // Debe mantenerse el roundtrip correctamente
            assert_eq!(decoded, text);
        }
    }
}