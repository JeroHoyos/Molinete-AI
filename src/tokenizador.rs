//! Tokenizador de Codificación de Pares de Bytes (BPE)
//!
//! Este módulo implementa la tokenización BPE desde cero. BPE es el método
//! de tokenización estándar utilizado por GPT-2, GPT-3 y muchos otros modelos de lenguaje.
//!
//! ## Cómo funciona BPE
//!
//! 1. **Comienza con codificación a nivel de bytes**: 256 tokens base (uno por cada valor de byte: 0-255)
//! 2. **Cuenta pares adyacentes**: Encuentra el par de bytes adyacentes más común en el corpus
//! 3. **Fusiona el par más frecuente**: Crea un nuevo token que represente ese par
//! 4. **Repite**: Continúa hasta que el vocabulario alcance el tamaño objetivo
//!
//! ## Ejemplo
//!
//! Dado el corpus: "hola hola mundo"
//! - Iteración 1: Par más común = ('h','o') → se fusiona en el token 256: "[256]la [256]la mundo"
//! - Iteración 2: Par más común = ('l','a') → se fusiona en el token 257: "[256][257] [256][257] mundo"
//! - Iteración 3: Par más común = ('[256]','[257]') → se fusiona en el token 258: "[258] [258] mundo"
//! - Continúa hasta que se alcance el tamaño del vocabulario (vocab_size)...
//!
//! ## Por qué es importante la tokenización
//!
//! Los modelos de lenguaje nunca ven el texto sin procesar; ven los ID de los tokens. Esta 
//! transformación fundamental explica muchos comportamientos "peculiares":
//!
//! - **No pueden contar letras de manera confiable**: El modelo ve tokens como ["corri", "endo"],
//!   no caracteres individuales.
//! - **Sensibles al espaciado**: " hola" y "hola" se tokenizan de manera diferente.
//! - **Tienen dificultades con palabras raras**: Las palabras poco comunes se dividen en fragmentos desconocidos.
//! - **Invertir textos es difícil**: Las operaciones a nivel de caracteres son complicadas cuando se trabaja con tokens.
//!
//! ## Notas de implementación
//!
//! Esta implementación incluye varias optimizaciones para su uso práctico:
//!
//! - **Conteo de pares en paralelo**: Utiliza Rayon para contar pares de bytes en todos los núcleos de la CPU,
//!   proporcionando un aumento de velocidad de 2 a 3 veces en sistemas multinúcleo.
//! - **Codificación en paralelo**: Los textos grandes se dividen en fragmentos y se codifican en paralelo.
//! - **Entrenamiento con muestras**: Para vocabularios muy grandes (>2000 tokens), el entrenamiento
//!   utiliza un subconjunto del corpus para aprender patrones más rápido.
//! - **Aplicación de fusión eficiente**: Construye nuevos vectores de tokens en lugar de eliminarlos 
//!   en su lugar original, evitando una complejidad O(n²).
//!
//! Estas optimizaciones son necesarias para que el entrenamiento sea práctico en la CPU, pero el
//! algoritmo subyacente es el BPE estándar.

// Librería externa para procesar datos en paralelo (multihilo) y ganar velocidad.
use rayon::prelude::*;
// Librería externa para guardar (serializar) y cargar (deserializar) el vocabulario.

// Estructura nativa de Rust para contar los pares de bytes usando clave-valor.
use std::collections::HashMap; 
// Módulo nativo para leer el corpus de texto y escribir los resultados en el disco.
use std::fs;
// Utilidad nativa para manejar las rutas de los archivos sin importar el sistema operativo.
use std::path::Path;

/// Un tokenizador de Codificación de Pares de Bytes
///
/// El tokenizador mantiene un vocabulario de tokens (cadenas de texto asociadas a IDs) y
/// una secuencia de reglas de fusión aprendidas durante el entrenamiento. Las fusiones se aplican
/// en orden durante la codificación para convertir el texto en IDs de tokens.

// Genera automáticamente el código para poder copiar la estructura en memoria (Clone), 
// guardarla en un archivo (Serialize) y cargarla desde un archivo (Deserialize).
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct TokenizadorBPE {
    /// Asocia las cadenas de texto de los tokens a sus IDs enteros
    /// Siempre comienza con 256 tokens base para la codificación a nivel de bytes
    vocabulario: HashMap<String, usize>,

    /// Secuencia de reglas de fusión aprendidas durante el entrenamiento
    /// Cada fusión combina dos tokens en un nuevo token
    /// Se aplican en orden durante la codificación
    fusiones: Vec<(String, String)>,
}

impl TokenizadorBPE { 
    /// Crea un nuevo tokenizador con el vocabulario base
    ///
    /// Inicializa el tokenizador con 256 tokens base que representan cada valor
    /// de byte posible (0-255). Estos se codifican como cadenas hexadecimales como "<00>", "<01>", etc.
    ///
    /// # Argumentos
    ///
    /// * `_tam_vocabulario` - Tamaño objetivo del vocabulario (actualmente no se usa en el constructor,
    ///   el tamaño real del vocabulario se determina durante el entrenamiento)
    ///
    /// # Retorna
    ///
    /// Un nuevo tokenizador con 256 tokens base y sin fusiones previas
    pub fn new(_tam_vocabulario: usize) -> Self {
        let mut vocabulario = HashMap::new();

        // Inicializa con tokens a nivel de bytes (256 tokens base)
        // Cada valor de byte se representa como una cadena hexadecimal: <00>, <01>, ..., <ff>
        for byte in 0..=255 {    
            // Guarda cada byte como texto (ej: "<00>") y le asigna un número de ID consecutivo (0, 1, 2...).
            vocabulario.insert(format!("<{:02x}>", byte), vocabulario.len());
        }

        Self {
            vocabulario,
            fusiones: Vec::new(),
        }
    }

    /// Entrena el tokenizador BPE en un corpus de texto
    ///
    /// Aprende las reglas de fusión encontrando y fusionando iterativamente
    /// los pares de tokens adyacentes más frecuentes. Esto construye un vocabulario
    /// desde los 256 tokens de bytes base hasta alcanzar el tamaño objetivo del vocabulario.
    ///
    /// # Argumentos
    ///
    /// * `texto` - Corpus de entrenamiento (típicamente varios MB de texto)
    /// * `tam_vocabulario` - Tamaño objetivo del vocabulario (valores comunes: 512, 1024, 2048, 5000)
    ///
    /// # Optimización de rendimiento
    ///
    /// Para vocabularios grandes (>2000 tokens), este método entrena con una muestra de 200KB
    /// del corpus en lugar del texto completo. Esto es mucho más rápido y aun así aprende
    /// los patrones más comunes de manera efectiva. Las fusiones aprendidas quedan disponibles
    /// para codificar el corpus completo después.
    ///
    /// # Ejemplo
    ///
    /// ```rust
    /// use molineteai::TokenizadorBPE;
    ///
    /// // Entrena en un corpus de texto de muestra
    /// let texto = "hola mundo hola rust hola mundo";
    /// let mut tokenizador = TokenizadorBPE::new(300);
    /// tokenizador.train(&texto, 300);
    ///
    /// // Verifica que el entrenamiento funcionó
    /// assert!(tokenizador.vocabulario.len() > 256);
    /// ```
    pub fn train(&mut self, texto: &str, tam_vocabulario: usize) {
        // Si el tamaño objetivo del vocabulario es 256 o menos, ya hemos terminado
        if tam_vocabulario <= 256 {
            return;
        }

        println!("Entrenando tokenizador BPE...");
        println!("  Tamaño inicial del vocab: {}", self.vocabulario.len());
        println!("  Tamaño objetivo del vocab: {}", tam_vocabulario);
        println!("  Tamaño del corpus: {} bytes", texto.len());

        // Determina el número de fusiones necesarias
        let num_fusiones = tam_vocabulario - 256;

        // Optimización: Para vocabularios grandes (>2000), entrena en un subconjunto más pequeño
        // 200KB es suficiente para aprender patrones comunes y se entrena mucho más rápido
        const TAM_MAX_MUESTRA: usize = 200_000;

        // Decidiendo qué texto usar y guardándolo directamente en 'texto_entrenamiento'.
        let texto_entrenamiento = if num_fusiones > 2000 && texto.len() > TAM_MAX_MUESTRA {
            
            // El 'floor_char_boundary' para no cortar caracteres, busca el límite seguro más cercano hacia abajo para no romper ninguna letra.
            let tam_muestra = texto.floor_char_boundary(TAM_MAX_MUESTRA);
            
            println!(
                "  Usando muestra de {}KB para ganar velocidad (aprende patrones comunes más rápido). {}",
                tam_muestra / 1000,
                tam_muestra
            );
            
            // Tomar desde el principio hasta el límite que calculamos".
            &texto[..tam_muestra]
            
        } else {
            // Si no se cumplen las condiciones, simplemente usamos todo el texto completo.
            texto
        };

        let mut tokens: Vec<String> = texto_entrenamiento
            // .bytes() devuelve un iterador sobre los valores numéricos crudos (bytes) 
            // que componen la cadena de texto UTF-8.
            .bytes()
            // .map() aplica una clausura a cada elemento producido por el iterador.
            // En este caso, la variable 'b' representa el byte actual, el cual es transformado 
            // a su representación hexadecimal de dos dígitos y envuelto en una cadena (ej. "<00>").
            .map(|b| format!("<{:02x}>", b))
            // Debido a que los iteradores en Rust utilizan evaluación perezosa,
            // no se ejecuta ningún procesamiento hasta que se consume el iterador.
            // .collect() consume el iterador y recolecta los elementos transformados 
            // dentro de la colección especificada en la declaración (el Vec<String>).
            .collect();

        // Búfer para evitar asignaciones repetidas en memoria (Optimización de Doble Búfer)
        let mut nuevos_tokens = Vec::with_capacity(tokens.len());

        // Aprende las fusiones de forma iterativa
        for idx_fusion in 0..num_fusiones {
            // === CONTEO DE PARES EN PARALELO ===
            // Este es el cuello de botella del rendimiento computacional, así que se paraleliza.

            // Determina el tamaño óptimo de la partición de datos (chunk) para el paralelismo.
            // Se impone un umbral mínimo de 50,000 tokens para amortizar el sobrecosto (overhead) 
            // derivado de la gestión de hilos del sistema operativo, o bien se divide el total 
            // de forma equitativa entre los hilos de CPU disponibles en el pool de Rayon.
            let tam_fragmento = 50_000.max(tokens.len() / rayon::current_num_threads().max(1));

            // Inicia una canalización (pipeline) de iteración paralela.
            let conteo_pares: HashMap<(String, String), usize> = tokens
                // .par_chunks() divide el vector original en sub-secciones estáticas no superpuestas.
                // Rayon distribuye automáticamente estos fragmentos entre múltiples hilos.
                .par_chunks(tam_fragmento)
                .enumerate()
                // .fold() ejecuta una agregación local en cada hilo de forma aislada.
                // Al proveer un 'HashMap::new' independiente por hilo, evitamos el uso de primitivas 
                // de sincronización (como un Mutex o un RwLock), eliminando por completo la contención 
                // de concurrencia y maximizando el rendimiento (lock-free).
                .fold(HashMap::new, |mut conteos_locales, (idx_fragmento, fragmento)| {
                    
                    // .windows(2) es una utilidad nativa de Rust que genera una "ventana deslizante".
                    // Sobre un arreglo [A, B, C, D], devuelve iterativamente [A, B], [B, C], y [C, D].
                    // Es la forma algorítmicamente más eficiente de procesar pares adyacentes continuos.
                    for ventana in fragmento.windows(2) {
                        let par = (ventana[0].clone(), ventana[1].clone());
                        
                        // La API '.entry()' es el estándar idiomático en Rust para diccionarios.
                        // Busca la tupla 'par'; si no existe, '.or_insert(0)' la inicializa en 0. 
                        // Posteriormente se desreferencia el puntero mutable (*) y se incrementa en 1.
                        *conteos_locales.entry(par).or_insert(0) += 1;
                    }

                    // Manejo de fronteras (Boundary Resolution):
                    // Como '.par_chunks()' realiza cortes drásticos en el vector, el par lógicamente 
                    // conformado por el último elemento del fragmento N y el primer elemento del 
                    // fragmento N+1 queda excluido del análisis anterior. 
                    // Este bloque verifica que no estemos en el límite absoluto de la colección 
                    // y registra manualmente ese par transicional.
                    if idx_fragmento * tam_fragmento + fragmento.len() < tokens.len() {
                        if let Some(ultimo) = fragmento.last() {
                            if let Some(siguiente) = tokens.get(idx_fragmento * tam_fragmento + fragmento.len()) {
                                let par = (ultimo.clone(), siguiente.clone());
                                *conteos_locales.entry(par).or_insert(0) += 1;
                            }
                        }
                    }

                    conteos_locales
                })
                // .reduce() constituye la fase final del patrón MapReduce.
                // Toma los múltiples HashMaps locales independientes generados en el paso '.fold()' 
                // y los consolida jerárquicamente de dos en dos en un único HashMap global.
                .reduce(HashMap::new, |mut a, b| {
                    for (par, conteo) in b {
                        *a.entry(par).or_insert(0) += conteo;
                    }
                    a
                });

            // Condición de salida temprana: si el conjunto de datos no posee pares adyacentes 
            // contabilizables, se interrumpe la iteración puesto que el entrenamiento ha convergido.
            if conteo_pares.is_empty() {
                break;
            }

            // === DESEMPATE DETERMINISTA ===
            // Los diccionarios (HashMap) en Rust utilizan por defecto un algoritmo de hashing 
            // (SipHash) que es resistente a colisiones maliciosas (HashDoS), lo que provoca que 
            // el orden de iteración sea pseudoaleatorio en cada ejecución. 
            // Para garantizar la reproducibilidad estricta del tokenizador, debemos ordenar los resultados.

            // .into_iter() consume el HashMap original, tomando posesión (ownership) de sus datos 
            // para evitar clonaciones costosas en memoria (Zero-cost abstraction).
            // .collect() materializa el iterador resultante en un vector (Vec), una estructura 
            // de memoria contigua que, a diferencia del HashMap, sí permite un ordenamiento posicional.
            let mut pares: Vec<((String, String), usize)> = conteo_pares.into_iter().collect();

            // .sort_by() aplica un algoritmo de ordenamiento en el lugar (in-place) altamente 
            // optimizado (Timsort en la biblioteca estándar de Rust), utilizando una clausura 
            // para definir la lógica de comparación.
            pares.sort_by(|a, b| {
                // Las variables 'a' y 'b' son referencias a las tuplas de la forma: ((String, String), usize).
                
                // b.1 y a.1 acceden al segundo elemento de la tupla (el conteo/frecuencia).
                // Al invocar .cmp() desde 'b' hacia 'a' (en lugar del tradicional 'a' hacia 'b'), 
                // forzamos algorítmicamente un orden DESCENDENTE, priorizando los pares más frecuentes.
                b.1.cmp(&a.1) 
                    
                    // .then_with() actúa como un operador de evaluación de cortocircuito (short-circuit).
                    // Si (y solo si) el resultado de la comparación anterior es 'Ordering::Equal' (un empate 
                    // en la frecuencia), se ejecuta esta clausura secundaria.
                    // Aquí, a.0 y b.0 acceden al primer elemento (la tupla de los dos tokens).
                    // Al comparar de 'a' hacia 'b', obtenemos un orden ASCENDENTE (lexicográfico), 
                    // resolviendo el empate de manera 100% determinista.
                    .then_with(|| a.0.cmp(&b.0)) 
            });

            // El primer par de la lista es el ganador
            let (mejor_par, cantidad) = pares[0].clone();

            // Crea un nuevo token concatenando el par ganador
            let nuevo_token = format!("{}{}", mejor_par.0, mejor_par.1);

            // Añade al vocabulario y registra la regla de fusión
            self.vocabulario.insert(nuevo_token.clone(), self.vocabulario.len());
            self.fusiones.push(mejor_par.clone());

            // === APLICA LA FUSIÓN AL CORPUS ===
            // Reconstruye la lista de tokens con la nueva fusión aplicada.
            // Aquí usamos la estrategia de doble búfer para reutilizar memoria.
            nuevos_tokens.clear();

            let mut i = 0;
            while i < tokens.len() {
                // Si encontramos el par, lo reemplazamos con el token fusionado
                if i < tokens.len() - 1 && tokens[i] == mejor_par.0 && tokens[i + 1] == mejor_par.1
                {
                    nuevos_tokens.push(nuevo_token.clone());
                    i += 2; // Saltamos ambos tokens del par porque ya se fusionaron
                } else {
                    nuevos_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }

            // Intercambia los búferes para que `tokens` tenga los datos actualizados para el próximo ciclo
            std::mem::swap(&mut tokens, &mut nuevos_tokens);

            // Registra el progreso en consola cada 50 fusiones
            if idx_fusion % 50 == 0 {
                println!(
                    "  Fusión {}/{}: {:?} (cantidad: {}) -> tamaño del vocab: {}",
                    idx_fusion + 1,
                    num_fusiones,
                    mejor_par,
                    cantidad,
                    self.vocabulario.len()
                );
            }
        }

        println!("¡Entrenamiento completado! Tamaño final del vocabulario: {}", self.vocabulario.len());
        println!("Se aprendieron {} fusiones\n", self.fusiones.len());
    }

    /// Convierte texto en una lista de cadenas de tokens a nivel de bytes
    ///
    /// Función auxiliar interna que transforma cada byte en su representación hexadecimal.
    ///
    /// # Argumentos
    ///
    /// * `texto` - Texto de entrada para convertir
    ///
    /// # Retorna
    ///
    /// Un vector de cadenas de tokens como ["<68>", "<65>", "<6c>", "<6c>", "<6f>"] para "hello"
    fn codificar_bytes(&self, texto: &str) -> Vec<String> {
        // .bytes() descompone el string en sus valores numéricos de 8 bits (0-255).
        // .map() aplica el formato hexadecimal "<{:02x}>" a cada número.
        // .collect() empaqueta el flujo de datos en el contenedor final Vec<String>.
        texto.bytes().map(|b| format!("<{:02x}>", b)).collect()
    }

    /// Codifica texto en una secuencia de IDs de tokens
    ///
    /// Convierte el texto en una secuencia de IDs transformándolo primero a tokens 
    /// a nivel de bytes y aplicando luego las reglas de fusión aprendidas.
    ///
    /// # Argumentos
    ///
    /// * `texto` - Texto de entrada para codificar
    ///
    /// # Retorna
    ///
    /// Un vector con los IDs de los tokens
    ///
    /// # Optimización de rendimiento
    ///
    /// Para textos grandes (>200KB), este método divide el texto en fragmentos (chunks)
    /// y los codifica en paralelo, acelerando el proceso en sistemas multinúcleo.
    pub fn codificar(&self, texto: &str) -> Vec<usize> {
        // Umbral para el procesamiento en paralelo
        const TAM_FRAGMENTO: usize = 100_000; // bytes por fragmento

        if texto.len() > TAM_FRAGMENTO * 2 {
            // === CODIFICACIÓN EN PARALELO PARA TEXTOS GRANDES ===
            
            let mut fragmentos = Vec::new();
            let mut inicio = 0;

            while inicio < texto.len() {
                // Calcula la posición final tentativa
                let mut fin = (inicio + TAM_FRAGMENTO).min(texto.len());

                // === RESTRICCIÓN DE SEGURIDAD UTF-8 ===
                // Rust no permite "romper" un carácter multibyte por la mitad. 
                // .is_char_boundary(fin) verifica si la posición 'fin' es un límite
                // legal de un carácter. Si no lo es, avanzamos hasta encontrar uno válido.
                while fin < texto.len() && !texto.is_char_boundary(fin) {
                    fin += 1;
                }

                // Creamos una "referencia" (slice) al fragmento del texto original sin copiarlo.
                fragmentos.push(&texto[inicio..fin]);

                inicio = fin;
            }

            // Codifica cada fragmento en paralelo usando Rayon
            // .par_iter() distribuye los fragmentos entre los hilos disponibles.
            let fragmentos_codificados: Vec<Vec<usize>> = fragmentos
                .par_iter()
                .map(|f| self.codificar_secuencial(f))
                .collect();

            // Consolida todos los fragmentos codificados en un solo vector de salida
            let mut resultado = Vec::new();
            for fragmento in fragmentos_codificados {
                // .extend_from_slice() es más eficiente que hacer un push uno por uno.
                resultado.extend_from_slice(&fragmento);
            }
            resultado
        } else {
            // Texto pequeño: usa la versión secuencial para evitar el sobrecosto 
            // de gestionar hilos (parallel overhead).
            self.codificar_secuencial(texto)
        }
    }

    /// Codifica texto de forma secuencial (versión no paralela)
    ///
    /// Método interno que realiza la codificación real aplicando las reglas de fusión.
    ///
    /// # Argumentos
    ///
    /// * `texto` - Texto de entrada para codificar
    ///
    /// # Retorna
    ///
    /// Un vector con los IDs de los tokens
    fn codificar_secuencial(&self, texto: &str) -> Vec<usize> {
        // 1. Transformación inicial a bytes (ej: "h" -> "<68>")
        let mut tokens = self.codificar_bytes(texto);
        
        // Búfer para evitar nuevas reservas de memoria en cada ciclo de fusión
        let mut nuevos_tokens = Vec::with_capacity(tokens.len());

        // 2. Aplicación de reglas de fusión (merges)
        for (par_a, par_b) in &self.fusiones {
            let fusionado = format!("{}{}", par_a, par_b);

            nuevos_tokens.clear();
            let mut i = 0;
            
            while i < tokens.len() {
                // Comprobamos si el par actual coincide con la regla de entrenamiento
                if i < tokens.len() - 1 && tokens[i] == *par_a && tokens[i + 1] == *par_b {
                    nuevos_tokens.push(fusionado.clone());
                    i += 2; // Fusionados: saltamos dos elementos
                } else {
                    nuevos_tokens.push(tokens[i].clone());
                    i += 1; // Sin cambios: avanzamos uno
                }
            }
            
            // Intercambio de punteros de memoria (rápido y eficiente)
            std::mem::swap(&mut tokens, &mut nuevos_tokens);
        }

        // 3. Conversión final: de texto de token a ID numérico
        tokens
            .iter()
            .map(|token| *self.vocabulario.get(token).unwrap_or(&0))
            .collect()
    }

    /// Decodifica IDs de tokens de vuelta a texto original
    ///
    /// Convierte una secuencia de IDs en el texto original buscando cada ID 
    /// en el vocabulario y procesando los bytes codificados en hexadecimal.
    ///
    /// # Argumentos
    ///
    /// * `ids` - Lista de IDs de tokens a decodificar
    ///
    /// # Retorna
    ///
    /// Una cadena de texto (String) decodificada
    pub fn decodificar(&self, ids: &[usize]) -> String {
        // === CREACIÓN DEL MAPA INVERSO ===
        // El vocabulario original es (Texto -> ID). Para decodificar necesitamos lo opuesto (ID -> Texto).
        // .map(|(token, id)| (*id, token.clone())) intercambia la clave por el valor.
        let id_a_token: HashMap<usize, String> = self
            .vocabulario
            .iter()
            .map(|(token, id)| (*id, token.clone()))
            .collect();

        // === RECONSTRUCCIÓN DE LA LISTA DE TOKENS ===
        let tokens: Vec<String> = ids
            .iter()
            // .filter_map() es una herramienta muy potente: intenta obtener el token del mapa.
            // Si el ID existe, lo incluye en la lista; si no existe (None), lo ignora silenciosamente.
            // Esto evita que el programa falle si recibe un ID corrupto.
            .filter_map(|id| id_a_token.get(id).cloned())
            .collect();

        // Unimos todos los tokens en una sola cadena de texto gigante (ej: "<68><65>hola")
        let fusionado = tokens.join("");
        
        // Finalmente, delegamos la limpieza del hexadecimal a la función interna
        self.decodificar_token(&fusionado)
    }

    /// Obtiene el tamaño actual del vocabulario
    ///
    /// # Retorna
    ///
    /// El número total de tokens registrados (256 base + cantidad de fusiones realizadas)
    pub fn tam_vocabulario(&self) -> usize {
        // .len() en un HashMap devuelve la cantidad de pares clave-valor en tiempo constante O(1).
        self.vocabulario.len()
    }

    /// Guarda el tokenizador en un archivo JSON
    ///
    /// Serializa el tokenizador (vocabulario y reglas de fusión) a un archivo JSON
    /// para poder cargarlo posteriormente sin tener que volver a entrenar.
    ///
    /// # Argumentos
    ///
    /// * `ruta` - Ruta del sistema de archivos donde se guardará el tokenizador
    ///
    /// # Retorna
    ///
    /// Un Result que indica éxito o un error detallado
    pub fn guardar<P: AsRef<Path>>(&self, ruta: P) -> Result<(), Box<dyn std::error::Error>> {
        // Convierte la estructura actual a una cadena de texto JSON con formato "bonito" (indentado)
        // El operador '?' captura cualquier error de serialización y lo devuelve inmediatamente.
        let json = serde_json::to_string_pretty(self)?;

        // Escribe la cadena JSON en el archivo especificado
        // El '?' aquí maneja errores del sistema de archivos (ej: falta de permisos).
        fs::write(ruta, json)?;

        // Si todo salió bien, devolvemos Ok
        Ok(())
    }

    /// Carga un tokenizador desde un archivo JSON
    ///
    /// Deserializa un tokenizador previamente guardado desde un archivo JSON.
    ///
    /// # Argumentos
    ///
    /// * `ruta` - Ruta al archivo del tokenizador
    ///
    /// # Retorna
    ///
    /// Un Result que contiene el tokenizador cargado (Self) o un error
    ///
    /// # Ejemplo
    ///
    /// ```rust,no_run
    /// use molineteai::TokenizadorBPE;
    ///
    /// let tokenizador = TokenizadorBPE::cargar("tokenizador.json")
    ///     .expect("No se pudo cargar el tokenizador");
    /// ```
    pub fn cargar<P: AsRef<Path>>(ruta: P) -> Result<Self, Box<dyn std::error::Error>> {
        // Lee todo el contenido del archivo y lo convierte en una cadena de texto (String)
        // El operador '?' maneja errores de lectura (ej: el archivo no existe)
        let json = fs::read_to_string(ruta)?;

        // Convierte la cadena JSON de vuelta a una estructura TokenizadorBPE
        // Para que esto funcione, la estructura debe tener la anotación #[derive(Deserialize)]
        let tokenizador: TokenizadorBPE = serde_json::from_str(&json)?;

        // Si la deserialización fue exitosa, devolvemos la instancia
        Ok(tokenizador)
    }

    /// Obtiene estadísticas detalladas sobre el estado del tokenizador
    ///
    /// # Retorna
    ///
    /// Una estructura 'EstadisticasTokenizador' con la información del vocabulario
    pub fn estadisticas(&self) -> EstadisticasTokenizador {
        // En Rust, la última expresión de una función (sin punto y coma) 
        // se devuelve automáticamente. No es necesario usar la palabra clave 'return'.
        EstadisticasTokenizador {
            // Tamaño total: tokens base + fusiones aprendidas
            tam_vocabulario: self.vocabulario.len(),
            
            // Cantidad de reglas de sustitución generadas durante el entrenamiento
            num_fusiones: self.fusiones.len(),
            
            // Los 256 tokens iniciales representan el espacio completo de un byte (0-255)
            tokens_base: 256,
        }
    }

    /// Analiza el vocabulario y muestra información diagnóstica
    ///
    /// Imprime detalles sobre la composición del vocabulario, incluyendo:
    /// - Composición de tokens (base vs. aprendidos)
    /// - Muestra de tokens fusionados
    /// - Análisis de compresión sobre un texto de ejemplo
    /// - Ejemplos reales de tokenización
    pub fn analizar_vocabulario(&self, texto_ejemplo: &str) {
        println!("\n=== Análisis del Vocabulario ===\n");

        // --- Filtrado de Tokens Legibles ---
        // Buscamos tokens que no sean simplemente bytes individuales (como "<68>")
        // o que, siendo representaciones hexadecimales, representen secuencias fusionadas.
        let mut tokens_legibles: Vec<(String, usize)> = self
            .vocabulario
            .iter()
            // Filtramos tokens que no empiecen con '<' (texto plano) o que sean largos
            .filter(|(token, _)| !token.starts_with('<') || token.len() > 4)
            .map(|(token, id)| (token.clone(), *id))
            .collect();

        // Ordenamos por ID (lo que refleja aproximadamente el orden cronológico de las fusiones)
        tokens_legibles.sort_by_key(|(_, id)| *id);

        // --- Desglose de Composición ---
        let tokens_base = 256;
        let tokens_fusionados = self.vocabulario.len() - tokens_base;
        println!("Composición de Tokens:");
        println!("  Tokens base (bytes): {}", tokens_base);
        println!("  Fusiones aprendidas: {}", tokens_fusionados);
        println!("  Vocabulario total: {}\n", self.vocabulario.len());

        // --- Muestra de Tokens Aprendidos ---
        println!("Muestra de Tokens Aprendidos (primeros 30):");
        let cantidad_a_mostrar = 30.min(tokens_legibles.len());
        for (token, id) in tokens_legibles.iter().take(cantidad_a_mostrar) {
            // Intentamos decodificar el token para mostrarlo de forma humana
            let decodificado = self.decodificar_token(token);
            if !decodificado.is_empty() && decodificado.len() <= 20 {
                println!("  [{}] \"{}\"", id, decodificado);
            }
        }

        // --- Análisis de Compresión ---
        if !texto_ejemplo.is_empty() {
            println!("\nAnálisis de Compresión (en muestra):");
            // Tomamos una muestra de hasta 10,000 caracteres para el análisis
            let muestra_chars: String = texto_ejemplo.chars().take(10000).collect();
            let tokens = self.codificar(&muestra_chars); // Usamos el método 'codificar'
            
            let conteo_chars = muestra_chars.len();
            let conteo_tokens = tokens.len();
            // El ratio de compresión indica cuántos caracteres representa, en promedio, cada token
            let ratio_compresion = conteo_chars as f32 / conteo_tokens as f32;

            println!("  Tamaño de la muestra: {} caracteres", conteo_chars);
            println!("  Cantidad de tokens: {} tokens", conteo_tokens);
            println!("  Ratio de compresión: {:.2}x", ratio_compresion);
            println!("  Promedio de caracteres por token: {:.1}", ratio_compresion);
        }

        // --- Ejemplos Prácticos de Tokenización ---
        println!("\nEjemplos de Tokenización:");
        let ejemplos = vec![
            "En un lugar de la Mancha",
            "el ingenioso hidalgo don Quijote",
            "Con esto que dijo Sancho Panza",
        ];

        for ejemplo in ejemplos {
            let ids_tokens = self.codificar(ejemplo);
            let strings_tokens: Vec<String> = ids_tokens
                .iter()
                .map(|&id| {
                    // Búsqueda inversa: ID -> String de token -> Texto decodificado
                    self.vocabulario
                        .iter()
                        .find(|(_, v)| **v == id)
                        .map(|(k, _)| self.decodificar_token(k))
                        .unwrap_or_else(|| "?".to_string())
                })
                .collect();
            
            println!(
                "  \"{}\" -> {} tokens: [{}]",
                ejemplo,
                ids_tokens.len(),
                strings_tokens.join("|") // Separamos los tokens con un pipe para visualizarlos mejor
            );
        }

        println!("\n{}\n", "=".repeat(60));
    }

    /// Auxiliar para convertir cadenas de tokens hexadecimales en texto legible
    ///
    /// Analiza una cadena que contiene bytes codificados como "<68><65><6c><6c><6f>"
    /// y los convierte de nuevo a texto UTF-8. Maneja tanto tokens individuales
    /// como secuencias concatenadas de tokens.
    ///
    /// # Argumentos
    ///
    /// * `token` - Cadena del token (o tokens concatenados) a decodificar
    ///
    /// # Retorna
    ///
    /// Representación en cadena de texto decodificada
    fn decodificar_token(&self, token: &str) -> String {
        // Almacenamos los bytes crudos antes de convertirlos a texto final
        let mut bytes = Vec::new();
        
        // .peekable() es un adaptador de iterador muy útil en Rust. 
        // Permite "mirar" el siguiente carácter sin consumirlo (sin avanzar el puntero),
        // lo cual es ideal para algoritmos de análisis (parsing) como este.
        let mut caracteres = token.chars().peekable();

        while let Some(ch) = caracteres.next() {
            if ch == '<' {
                // Inicio de un byte hexadecimal: <XX> -> valor numérico
                let mut hex_str = String::new();
                
                // Mientras el siguiente carácter no sea el cierre '>', lo acumulamos
                while let Some(&sig_ch) = caracteres.peek() {
                    if sig_ch == '>' {
                        caracteres.next(); // Consumimos el '>' y salimos del bucle interno
                        break;
                    }
                    // .unwrap() aquí es seguro porque acabamos de confirmar con .peek() que existe
                    hex_str.push(caracteres.next().unwrap());
                }

                // Intentamos convertir la cadena de texto (ej: "68") a un número base 16 (u8)
                if let Ok(byte) = u8::from_str_radix(&hex_str, 16) {
                    bytes.push(byte);
                }
            }
            // Los caracteres fuera de los brackets se ignoran (en un BPE puro no debería haberlos)
        }

        // === CONVERSIÓN FINAL A UTF-8 ===
        // 'from_utf8_lossy' es una elección de diseño robusta. 
        // Si el tokenizador cortó un carácter multi-byte a la mitad y los bytes resultantes 
        // no forman un UTF-8 válido, esta función insertará un carácter de reemplazo () 
        // en lugar de hacer que el programa colapse (panic).
        String::from_utf8_lossy(&bytes).to_string()
    }
}

/// Estadísticas sobre el vocabulario de un tokenizador
#[derive(Debug)]
pub struct EstadisticasTokenizador {
    /// Tamaño total del vocabulario (tokens base + fusiones aprendidas)
    pub tam_vocabulario: usize,
    /// Número de reglas de fusión aprendidas
    pub num_fusiones: usize,
    /// Número de tokens base (siempre 256 para BPE a nivel de bytes)
    pub tokens_base: usize,
}

#[cfg(test)]
mod pruebas {
    use super::*;

    #[test]
    fn test_decodificar_token_un_byte() {
        let tokenizador = TokenizadorBPE::new(256);
        // Token de un solo byte: 'h' = 0x68
        let resultado = tokenizador.decodificar_token("<68>");
        assert_eq!(resultado, "h");
    }

    #[test]
    fn test_decodificar_token_multiples_bytes() {
        let tokenizador = TokenizadorBPE::new(256);
        let resultado = tokenizador.decodificar_token("<68><65><6c><6c><6f>");
        assert_eq!(resultado, "hello");
    }

    #[test]
    fn test_decodificar_token_con_espacio() {
        let tokenizador = TokenizadorBPE::new(256);
        // "hi " (incluye espacio, 0x20)
        let resultado = tokenizador.decodificar_token("<68><69><20>");
        assert_eq!(resultado, "hi ");
    }

    #[test]
    fn test_decodificar_token_utf8_multibyte() {
        let tokenizador = TokenizadorBPE::new(256);
        // "é" en UTF-8 se compone de [0xc3, 0xa9]
        let resultado = tokenizador.decodificar_token("<c3><a9>");
        assert_eq!(resultado, "é");
    }

    #[test]
    fn test_decodificar_token_vacio() {
        let tokenizador = TokenizadorBPE::new(256);
        let resultado = tokenizador.decodificar_token("");
        assert_eq!(resultado, "");
    }

    #[test]
    fn test_decodificar_basico() {
        let tokenizador = TokenizadorBPE::new(256);
        let texto = "hello";
        let ids = tokenizador.codificar(texto);
        let decodificado = tokenizador.decodificar(&ids);

        assert_eq!(decodificado, texto);
    }

    #[test]
    fn test_codificar_decodificar_ciclo_completo() {
        let tokenizador = TokenizadorBPE::new(256);

        let casos_prueba = vec![
            "hello",
            "Hello, world!",
            "To be, or not to be",
            "123 456 789",
            "special chars: !@#$%^&*()",
            "newline\nand\ttab",
            "UTF-8: café, naïve, 日本語",
        ];

        for texto in casos_prueba {
            let codificado = tokenizador.codificar(texto);
            let decodificado = tokenizador.decodificar(&codificado);
            assert_eq!(decodificado, texto, "Fallo en ciclo completo para: {}", texto);
        }
    }

    #[test]
    fn test_codificar_decodificar_con_fusiones() {
        let mut tokenizador = TokenizadorBPE::new(300);
        let texto_entrenamiento = "hello hello world world hello";
        tokenizador.entrenar(texto_entrenamiento, 300);

        // Verifica la integridad algorítmica tras mutar el estado del vocabulario
        let texto_prueba = "hello world";
        let codificado = tokenizador.codificar(texto_prueba);
        let decodificado = tokenizador.decodificar(&codificado);

        assert_eq!(decodificado, texto_prueba);
    }

    #[test]
    fn test_consistencia_decodificar_token() {
        let tokenizador = TokenizadorBPE::new(256);
        let token_str = "<68><65><6c><6c><6f>"; // "hello"

        let resultado_directo = tokenizador.decodificar_token(token_str);
        let resultado_simulado = tokenizador.decodificar_token(token_str);

        assert_eq!(resultado_directo, resultado_simulado);
        assert_eq!(resultado_directo, "hello");
    }

    #[test]
    fn test_decodificar_token_concatenado() {
        let tokenizador = TokenizadorBPE::new(256);
        // Simulación de flujo concatenado previo a la decodificación final
        let concatenado = "<68><65><6c><6c><6f><20><77><6f><72><6c><64>";
        let resultado = tokenizador.decodificar_token(concatenado);

        assert_eq!(resultado, "hello world");
    }

    #[test]
    fn test_tam_vocabulario() {
        let tokenizador = TokenizadorBPE::new(256);
        assert_eq!(tokenizador.tam_vocabulario(), 256);

        let mut tokenizador2 = TokenizadorBPE::new(512);
        tokenizador2.entrenar("hello hello world", 512);
        
        // Verifica incremento sin desbordar el objetivo
        assert!(tokenizador2.tam_vocabulario() > 256);
        assert!(tokenizador2.tam_vocabulario() <= 512);
    }

    #[test]
    fn test_cobertura_vocabulario_base() {
        let tokenizador = TokenizadorBPE::new(256);

        // Validación exhaustiva del espacio de bytes (0x00 - 0xFF)
        for byte in 0u8..=255u8 {
            let texto = String::from_utf8(vec![byte]).unwrap_or_else(|_| {
                String::from_utf8_lossy(&[byte]).to_string()
            });

            let codificado = tokenizador.codificar(&texto);
            let decodificado = tokenizador.decodificar(&codificado);

            assert_eq!(decodificado, texto);
        }
    }
}