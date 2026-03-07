# Arquitectura del Modelo: construyendo GPT-2 desde cero

Este documento acompaña el artículo  
["Building an LLM From Scratch in Rust, Part 3: Model Architecture"](https://www.tag1.com/how-to/part3-model-architecture-building-an-llm-from-scratch/).

En esta fase se implementa una **arquitectura Transformer estilo GPT-2 completa**.  
El modelo puede realizar **forward passes (inferencia)**, pero aún no incluye entrenamiento, por lo que los pesos iniciales son aleatorios y las predicciones no tienen significado.

## Archivos relevantes

- `src/model.rs` — Componentes principales del modelo  
- `examples/03_model_architecture.rs` — Creación del modelo y prueba de forward pass

## Ejecutar el ejemplo

```bash
cargo run --release --example 03_model_architecture
```

El ejemplo crea modelos de distintos tamaños y ejecuta una pasada hacia adelante.

El resultado esperado es un tensor de salida con forma:

```
[batch, seq_len, vocab_size]
```

que representa los **logits del modelo sobre el vocabulario**.

---

# Tamaños de modelo

El ejemplo incluye varias configuraciones.

### Tiny (~0.8M parámetros)

```
vocab_size: 512
n_embd: 128
n_heads: 4
n_layers: 3
block_size: 128
```

- Muy rápido
- Ideal para pruebas y depuración

---

### Small (~3.5M parámetros)

```
vocab_size: 512
n_embd: 256
n_heads: 4
n_layers: 4
block_size: 256
```

- Buen equilibrio entre velocidad y capacidad
- Útil para experimentación

---

### Medium (~20–40M parámetros)

```
n_embd: 384
n_heads: 6
n_layers: 6
```

- Mayor capacidad de representación
- Todavía viable en CPU

---

### GPT-2 Small (~163M parámetros)

```
vocab_size: 50257
n_embd: 768
n_heads: 12
n_layers: 12
block_size: 1024
```

Arquitectura estándar de GPT-2.

---

# Estructura del modelo

El modelo GPT se compone de:

```rust
pub struct GPT {
    wte: Embedding,       // token embeddings
    wpe: Embedding,       // position embeddings
    blocks: Vec<Block>,   // transformer blocks
    ln_f: LayerNorm,      // layer norm final
    lm_head: Linear       // proyección al vocabulario
}
```

Cada **Transformer Block** contiene:

```rust
pub struct Block {
    ln_1: LayerNorm,
    attn: MultiHeadAttention,
    ln_2: LayerNorm,
    mlp: MLP
}
```

---

# Atención multi-cabeza

La atención usa proyecciones lineales para obtener:

- **Query**
- **Key**
- **Value**

En lugar de tres multiplicaciones de matrices separadas, se usa **una sola proyección grande**:

```
input → Linear(n_embd, 3*n_embd)
```

Luego se divide en Q, K y V.

Esto mejora el rendimiento.

La dimensión de cada cabeza es:

```
head_dim = n_embd / n_heads
```

Configuraciones comunes usan **head_dim = 64**.

---

# Flujo de un Transformer Block

Cada bloque ejecuta:

```
x → LayerNorm
  → Self-Attention
  → Residual connection
  → LayerNorm
  → Feedforward (MLP)
  → Residual connection
```

El **MLP** expande primero la dimensión:

```
n_embd → 4*n_embd
```

y luego vuelve a proyectarla:

```
4*n_embd → n_embd
```

La activación usada es **GELU**.

---

# Inicialización de pesos

Siguiendo GPT-2:

**Pesos de embeddings y capas lineales**

```
Normal(0, 0.02)
```

**Bias**

```
0
```

**LayerNorm**

```
gamma = 1
beta = 0
```

Esto evita activaciones demasiado grandes al inicio del entrenamiento.

---

# Organización de memoria

Los tensores usan **row-major layout**:

```
[batch, seq, embd]
```

La dimensión del embedding es contigua en memoria, lo que mejora el rendimiento en multiplicaciones de matrices.

---

# Limitaciones de la implementación

Para mantener claridad educativa, esta implementación no incluye:

- weight tying entre embeddings y salida
- dropout
- flash attention
- KV cache para generación eficiente

Estas optimizaciones se usan en modelos de producción.

---

# Problemas comunes

### Dimensiones incorrectas

El número de cabezas debe dividir el embedding:

```
n_embd % n_heads == 0
```

---

### Memoria insuficiente

Modelos grandes consumen mucha memoria.

Soluciones:

- reducir `n_embd`
- reducir `n_layers`
- reducir `block_size`
- reducir `vocab_size`

---

### Forward pass lento

Compilar siempre en modo release:

```bash
cargo run --release --example 03_model_architecture
```

---

# Complejidad computacional

El costo de atención es:

```
O(n²)
```

donde **n es la longitud de la secuencia**.

Duplicar la longitud de contexto cuadruplica el costo de atención.

---

# Próximo paso

Con la arquitectura implementada, el siguiente paso es **entrenar el modelo**.

Esto requiere:

- cálculo de pérdida
- backpropagation
- actualización de pesos
- infraestructura de entrenamiento

---

# Lecturas recomendadas

- *Attention Is All You Need* — Vaswani et al. (2017)
- *Language Models are Unsupervised Multitask Learners* — Radford et al. (2019)
- *The Illustrated Transformer* — Jay Alammar
- *nanoGPT* — Andrej Karpathy