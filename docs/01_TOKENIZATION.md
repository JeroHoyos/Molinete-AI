# Tokenización: de texto a números

Este documento acompaña el artículo  
["Building an LLM From Scratch in Rust, Part 1: Tokenization"](https://www.tag1.com/how-to/part1-tokenization-building-an-llm-from-scratch-in-rust/).

Los modelos de lenguaje no leen texto directamente: leen **números**.  
El proceso que convierte texto en números se llama **tokenización**.

En este proyecto utilizamos **Byte Pair Encoding (BPE)**, el mismo algoritmo utilizado por GPT-2 y GPT-3.

## Archivos relevantes

- `src/tokenizer.rs` — Implementación principal de BPE
- `examples/01_train_tokenizers.rs` — Ejemplo de entrenamiento

## Ejecutar el ejemplo

```bash
# Descargar el corpus de Shakespeare
curl -o shakespeare.txt https://www.gutenberg.org/files/100/100-0.txt

# Entrenar tokenizadores con distintos tamaños de vocabulario
cargo run --release --example 01_train_tokenizers
```

El script entrena tokenizadores con los tamaños:

```
[256, 512, 1024, 1536, 20534]
```

y guarda los resultados en:

```
data/example_tokenizer_<timestamp>/
```

## Interpretación de los tamaños de vocabulario

**256 — Nivel byte**

- No se aprenden fusiones
- Cada byte es un token
- Compresión ≈ 1.0x

**512 — Modelo pequeño**

- Se aprenden patrones frecuentes como `th`, `e ` o `and`
- Compresión aproximada: ~2x

**1024 — Modelo intermedio**

- Palabras comunes empiezan a tener su propio token
- Compresión aproximada: ~2.5x

**1536 — Modelo mediano**

- Se aprenden patrones más largos
- Compresión aproximada: ~2.8x

**20534 — Vocabulario grande**

- Se capturan frases frecuentes
- Compresión aproximada: ~3.6x

## Trade-offs del tamaño del vocabulario

**Ventajas de vocabularios grandes**

- Mejor compresión
- Secuencias más cortas
- Representaciones más precisas

**Desventajas**

- Tablas de embedding más grandes
- Más parámetros para entrenar
- Codificación más lenta

Para experimentación y aprendizaje, tamaños entre **512 y 1536** suelen ser el mejor equilibrio.

## API principal

```rust
let mut tokenizer = BPETokenizer::new(vocab_size);

// entrenar
tokenizer.train(&text, vocab_size);

// codificar texto
let ids = tokenizer.encode(text);

// decodificar
let text = tokenizer.decode(&ids);

// guardar / cargar
tokenizer.save("tokenizer.json")?;
let tokenizer = BPETokenizer::load("tokenizer.json")?;
```

## Estructura principal

```rust
pub struct BPETokenizer {
    vocab: HashMap<String, usize>,
    merges: Vec<(String, String)>,
    unk_token: String,
}
```

El vocabulario inicia con **256 tokens base** (bytes).  
Durante el entrenamiento se agregan nuevos tokens aprendidos mediante fusiones BPE.

## Optimizaciones implementadas

- **Conteo paralelo de pares** usando Rayon
- **Manejo de límites de chunks** para evitar perder pares entre bloques
- **Sampling** para vocabularios grandes
- **Aplicación eficiente de merges** creando nuevos vectores en lugar de modificar el existente

Estas optimizaciones reducen significativamente el tiempo de entrenamiento.

## Problemas comunes

**No se encuentra `cervantes.txt`**

Descargar:

```bash
curl -o cervantes.txt https://www.gutenberg.org/files/100/100-0.txt
```

**Entrenamiento lento**

Compilar en modo release:

```bash
cargo run --release --example 01_train_tokenizers
```

**Falla encode/decode**

Si el texto original no coincide tras codificar y decodificar, el tokenizer está corrupto y debe volver a entrenarse.

## Siguiente paso

Una vez entendido el proceso de tokenización, el siguiente paso es implementar **operaciones con tensores**, la base matemática de las redes neuronales y de la arquitectura Transformer.

## Lecturas recomendadas

- *Neural Machine Translation of Rare Words with Subword Units* — Sennrich et al. (2016)
- Implementación de tokenizer de GPT-2
- Biblioteca SentencePiece de Google
- Documentación de Byte-Level BPE de HuggingFace