<div align="center">

# Molinete AI  
## Construyendo un Transformer con Rust

</div>

**Autor:**  
Jerónimo Hoyos Botero  

**Repositorio principal:**  
https://github.com/JeroHoyos/Molinete-AI  

**Repositorio original:**  
https://github.com/tag1consulting/feste  

---

## ¿Qué es Molinete AI?

Molinete AI es un fork de **Feste**, una implementación desde cero de un modelo Transformer tipo GPT-2 en Rust desarrollada por Tag1 Consulting como acompañamiento a la serie *Building an LLM From Scratch in Rust*.

Mientras que Feste entrena el modelo con las obras completas de Shakespeare, **Molinete AI propone entrenarlo con la obra de Miguel de Cervantes**, estableciendo una contraposición lingüística y cultural:

- Feste → Shakespeare (inglés isabelino)
- Molinete AI → Cervantes (español del Siglo de Oro)

El objetivo no es solo replicar el experimento original, sino reinterpretarlo en español y convertirlo en una guía técnica rigurosa.

---

## ¿Por qué “Molinete”?

El nombre hace referencia al universo cervantino.  
Si Feste toma su identidad del bufón ingenioso en *Twelfth Night*, Molinete AI toma relación a los fieros oponentes que tuvo Hildago contra los molinos de viento.

---

## Qué es este proyecto

Un modelo Transformer completamente entrenable, implementado desde cero en Rust, sin frameworks de deep learning.

Incluye:

- Tokenización BPE.
- Implementación manual de tensores.
- Multi-Head Self-Attention.
- Máscara causal.
- Feed Forward Networks.
- Normalización y conexiones residuales.
- Infraestructura de entrenamiento.
- Generación autoregresiva de texto.

El propósito es comprender cómo funcionan los modelos de lenguaje implementando cada componente explícitamente.

---

## Diferencias frente al repositorio original

Este fork agrega:

- Scripts experimentales para observar ejemplos y el comportamiento interno de cada componente del modelo (tokenización, atención, capas feed forward, normalización y generación).
- Una presentación desarrollada en Manim que explica visualmente la arquitectura del Transformer y el flujo de información entre capas.
- Documentación completa en español, con explicaciones adicionales orientadas a comprender mejor el código y su estructura interna.

El enfoque es pedagógico y analítico, priorizando la comprensión detallada del funcionamiento del modelo.

---
## Serie del Blog

Cada parte del blog tiene un documento complementario con detalles de configuración y referencia de implementación:

| Parte | Publicación del Blog | Referencia de Código |
|------|-----------|----------------|
| 1 | [Tokenización](https://www.tag1.com/how-to/part1-tokenization-building-an-llm-from-scratch-in-rust/) | [`docs/01_TOKENIZATION.md`](docs/01_TOKENIZATION.md) |
| 2 | [Operaciones con tensores](https://www.tag1.com/how-to/part2-tensor-operations-building-an-llm-from-scratch/) | [`docs/02_TENSOR_OPERATIONS.md`](docs/02_TENSOR_OPERATIONS.md) |
| 3 | [Arquitectura del modelo](https://www.tag1.com/how-to/part3-model-architecture-building-an-llm-from-scratch/) | [`docs/03_MODEL_ARCHITECTURE.md`](docs/03_MODEL_ARCHITECTURE.md) |
| 4 | [Infraestructura de entrenamiento](https://www.tag1.com/how-to/part4-training-infrastructure-building-an-llm-from-scratch/) | [`docs/04_TRAINING.md`](docs/04_TRAINING.md) |
| 5 | [Un bufón sin ingenio](https://www.tag1.com/how-to/part5-witless-fool-building-an-llm-from-scratch/) | [`docs/05_TRAINING_EXAMPLES.md`](docs/05_TRAINING_EXAMPLES.md) |

## Inicio rápido

```bash
# Obtener los datos de entrenamiento
curl -o cervantes.txt https://www.gutenberg.org/files/100/100-0.txt

# Entrenar un modelo pequeño (10–15 minutos)
cargo run --release --example 06_train_cervantes_small
```

## Reproducir los experimentos del blog

El ejemplo de entrenamiento configurable permite reproducir cualquier experimento del artículo de la **Parte 5 del blog** utilizando configuraciones predefinidas (presets).

```bash
# Listar configuraciones disponibles
cargo run --release --example train -- --list-presets

# Ejecutar una configuración predefinida
cargo run --release --example train -- --preset pocket-bard

# Sobrescribir parámetros
cargo run --release --example train -- --preset spider --steps 10000

# Configuración completamente personalizada
cargo run --release --example train -- \
    --embd 256 --layers 6 --heads 12 --context 448 --vocab 8192
```

Consulta [`docs/05_TRAINING_EXAMPLES.md`](docs/05_TRAINING_EXAMPLES.md) para ver la tabla completa de configuraciones predefinidas, instrucciones de *transfer learning* y detalles sobre todos los ejemplos de entrenamiento.

## Ejemplos

### Fundamentos (Partes 1–4)

- `01_train_tokenizers` — Entrenamiento de tokenizadores BPE con diferentes tamaños de vocabulario  
- `02_tensor_operations` — Multiplicación de matrices y operaciones tensoriales  
- `03_model_architecture` — Exploración de la arquitectura Transformer  
- `04_training_infrastructure` — Componentes del bucle de entrenamiento  

### Entrenamiento (Parte 5)

- `05_train_cervantes_tiny` — 50K parámetros, 2–5 minutos  
- `06_train_cervantes_small` — 200K parámetros, 10–20 minutos  
- `07_train_cervantes_medium` — 4M parámetros, 1–2 horas  
- `08_train_cervantes_gpt2` — 163M parámetros (GPT-2 Small), 24–30 horas  
- `train` — Entrenamiento configurable con configuraciones del blog

## License

Apache 2.0

---

