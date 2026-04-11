<div align="center">
<pre>
______________________________________________________________________________________________________________________8______________________
______________________________________________________________________________________________________________________8666___________________
______________________________________________________________________________________________________________________8_868__________________
______________________________________________________________________________________________________________________8_688_8________________
███╗░░░███╗░█████╗░██╗░░░░░██╗███╗░░██╗███████╗████████╗███████╗__░█████╗░██╗____________________88888________________8668888________________
████╗░████║██╔══██╗██║░░░░░██║████╗░██║██╔════╝╚══██╔══╝██╔════╝__██╔══██╗██║__________________888888888______________8___888_________88_8___
██╔████╔██║██║░░██║██║░░░░░██║██╔██╗██║█████╗░░░░░██║░░░█████╗░░__███████║██║__________________888_8888888____________8_____________8668888__
██║╚██╔╝██║██║░░██║██║░░░░░██║██║╚████║██╔══╝░░░░░██║░░░██╔══╝░░__██╔══██║██║___________________668888888888_________86___________8688__8888_
██║░╚═╝░██║╚█████╔╝███████╗██║██║░╚███║███████╗░░░██║░░░███████╗__██║░░██║██║_____________________8668__888888________8_________868888888888_
╚═╝░░░░░╚═╝░╚════╝░╚══════╝╚═╝╚═╝░░╚══╝╚══════╝░░░╚═╝░░░╚══════╝__╚═╝░░╚═╝╚═╝_______________________86688__88888_____86_______868888888888___
______________________________________________________________________________________________________866888888888__88868___8688_8888888_____
________________________________________________________________________________________________________866888888_86888888888_88888888_______
__________________________________________________________________________________________________________866888_88888886888__88888__________
_________________ /\ ____ ,, _______________________________________________________________________________86688688888888668__88____________
________  .---. __|| ____ /|| ________________________________________________________________________________86688888886888888______________
______ --'-----`--||    .'  \ ________________________________________________________________________________88666688666888888______________
________ {{{N `(  ||  .'    @  _____________________________________________________________________________8868868888688888888______________
________ {{{` _/  ||.'    |  \  __________________________________________________________________________88888888866888688_8_88_____________
________ {{{.-.   ||  /  /\   \  ________________________________________________________________________888868_868_888_86668886_____________
________  {( )| .'||    /  `.  \  ______________________________________________________________________888_68866__888888_866__8_____________
__        {|\ \'  / )  / __  \\O| ____________________________________________________________________8888__8868____888888__86888____________
  `-.____.-| \ \ /\/  / ____ `'   __________________________________________________________________88888__8668_______888888__866____________
 -    ////|  \ Y /|  | ____________________________________________________________________________88888__666888_______8888888__868__________
   |   |||||`-|\^/|| |   ________________________________________________________________________888888_86668_88__________888888__66_________
       |||||`-| " [] / _________________________________________________________________________88888__66_68888_____________888888__68_______
      _ \\\\/`-|   []|\  _____________________________________________________________________88888__866__6_888__88888_______888886__668_____
 ) |`---``| _ |__([]| \   ___________________________________________________________________88888__86___688888__888_8________888886__8668___
  / ____  |/ `|  FJ|\ \   ___________________________________________________________________8888_866____686886_86688888__666__888868___868__
 / _____  `|  |  FJ) \ \    ___________________________________________________________________8_868____6886_88_68888886__868__8__866_88_____
/ ______   |  |  FJ|  \ )  _____________________________________________________________________86______6_68_88_6_8888_8__868___8__868_______
| ______   |  F  J  L  || _____________________________________________________________________________88866_88_8_6688_8__888___8___86_______
`.  ____   )-(> '----` || _____________________________________________________________________________6_888888_8_8688_8________8____68______
`.\  ___   | |    |||  || ____________________________________________________________________________86___6668868888888688___886688_86______
| \\  __   |-|    ||| / | ____________________________________________________________________________6888888668888888866888888_88688868_____
 \ )\    *_)/`-.__|| \\ | _________________________________________________________________________8888888888888_88888_8888888_888888886_____
--'--'"""""`------''--'`'""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""888888888888888888888888888888888888888"
</pre>

# Molinete AI

**Un Transformer GPT-2 implementado desde cero en Rust, entrenado con las obras de Miguel de Cervantes.**

[![Licencia](https://img.shields.io/badge/licencia-Apache_2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)

**Autor:** Jerónimo Hoyos Botero
**Repositorio:** [JeroHoyos/Molinete-AI](https://github.com/JeroHoyos/Molinete-AI)
**Basado en:** [tag1consulting/feste](https://github.com/tag1consulting/feste)

</div>

---

## ¿Qué es Molinete AI?

Molinete AI es un fork de **Feste**, una implementación desde cero de un modelo Transformer tipo GPT-2 en Rust, desarrollada por Tag1 Consulting como acompañamiento a la serie *Building an LLM From Scratch in Rust*.

Mientras que Feste entrena el modelo con las obras completas de Shakespeare, **Molinete AI propone entrenarlo con la obra de Miguel de Cervantes**, estableciendo una interesante contraposición lingüística y cultural:

| Proyecto | Corpus | Idioma |
|:---|:---|:---|
| **Feste** | Shakespeare | Inglés isabelino |
| **Molinete AI** | Cervantes | Español del Siglo de Oro |

El objetivo no es solo replicar el experimento original, sino reinterpretarlo en español y convertirlo en una guía técnica rigurosa y accesible.

---

## ¿Por qué "Molinete"?

El nombre rinde homenaje al universo cervantino. Si Feste toma su identidad del bufón ingenioso de *Twelfth Night*, **Molinete AI** hace referencia a los famosos molinos de viento que el ingenioso hidalgo Don Quijote confundió con fieros gigantes — una metáfora perfecta para un modelo de lenguaje que intenta imitar la grandeza de algo mucho más vasto que él mismo.

---

## Sobre el proyecto

Es un modelo Transformer completamente entrenable, implementado desde cero en Rust, **sin depender de frameworks de deep learning** como PyTorch o TensorFlow.

El propósito central es comprender cómo funcionan los modelos de lenguaje implementando cada componente explícitamente:

- **Tokenización BPE** (Byte-Pair Encoding)
- **Implementación manual de tensores**
- **Multi-Head Self-Attention y máscara causal**
- **Feed Forward Networks**
- **Normalización y conexiones residuales**
- **Infraestructura de entrenamiento completa** (warmup, gradient clipping, early stopping)
- **Generación autoregresiva de texto**

### Arquitectura Molinete (~4M params)

```rust
Config {
    vocab_size: 1536,
    n_embd: 256,
    n_layers: 4,
    n_heads: 4,       // head_dim = 64
    block_size: 256,
}
```

| Hiperparámetro | Valor por defecto |
|:---|:---|
| Pasos | `8000` |
| Tasa de aprendizaje | `0.0003` |
| Fracción de calentamiento (warmup) | `0.1` |
| Recorte de gradiente (gradient clipping) | `1.0` |
| Paciencia de early stopping | `3000` |
| Acumulación de gradientes | `8 mini-batches` |
| Datos de entrenamiento | Primeros 2M de caracteres |
| Tiempo estimado (local) | ~4–6 horas |

---

## Estructura del repositorio

```
molinete-ai/
├── src/
│   ├── layers/
│   │   ├── activation.rs
│   │   ├── attention.rs
│   │   ├── block.rs
│   │   ├── dropout.rs
│   │   ├── layer_norm.rs
│   │   ├── linear.rs
│   │   ├── mlp.rs
│   │   └── mod.rs
│   ├── entrenamiento.rs
│   ├── gpt2_entrenable.rs
│   ├── gradientes.rs
│   ├── lib.rs
│   ├── modelo.rs
│   ├── optimizador.rs
│   ├── python_bindings.rs
│   ├── registrador_entrenamiento.rs
│   ├── tensor.rs
│   └── tokenizador.rs
├── examples/
│   ├── presentation/          ← Animaciones Manim
│   └── *.py                   ← Scripts de ejemplo Python
├── cervantes.txt              ← Corpus de entrenamiento (ver DATA.md)
├── Cargo.toml
├── pyproject.toml
├── DATA.md                    ← Guía del corpus de datos
└── README.md
```

---

## Inicio rápido

### Requisitos

- [Rust](https://rustup.rs/) 1.75+
- Python 3.9+ (para los ejemplos Python y los bindings)
- `maturin` para compilar los bindings PyO3

```bash
pip install maturin
```

### 1. Clonar el repositorio

```bash
git clone https://github.com/JeroHoyos/Molinete-AI.git
cd Molinete-AI
```

### 2. Descargar el corpus

```bash
python download_data.py
# Genera: cervantes.txt (~5–7 MB)
```

O manualmente desde Project Gutenberg (ver [DATA.md](DATA.md) para más detalles).

### 3. Compilar los bindings Python

```bash
maturin develop --release
```

### 4. Ejecutar el lanzador interactivo

```bash
python molineteai.py
```

Esto abre el menú principal con todas las opciones disponibles, incluyendo los ejemplos de aprendizaje, los distintos tamaños de modelo y el chat con modelos entrenados.

---

## Ejemplos en Rust

También puedes ejecutar los ejemplos directamente con Cargo:

```bash
# 1. Tokenización BPE con diferentes tamaños de vocabulario
cargo run --release --example 01_train_tokenizers

# 2. Multiplicación de matrices y operaciones tensoriales
cargo run --release --example 02_tensor_operations

# 3. Exploración visual de la arquitectura Transformer
cargo run --release --example 03_model_architecture

# 4. Análisis de los componentes del bucle de entrenamiento
cargo run --release --example 04_training_infrastructure

# 5. Entrenamiento completo con la obra de Cervantes
cargo run --release --example 05_train_cervantes

# 6. Inferencia, generación de texto y prompts
cargo run --release --example 06_promting
```

---

## Bindings Python

El proyecto incluye bindings completos vía PyO3/maturin. Una vez compilado con `maturin develop --release`:

```python
import molineteai

# Tokenizador
tok = molineteai.TokenizadorBPE(1536)
tok.entrenar(texto, 1536)
ids = tok.codificar("En un lugar de la Mancha")
print(tok.decodificar(ids))

# Configuración y modelo
config = molineteai.Config.mediana(tok.tam_vocabulario())
modelo = molineteai.GPT2Entrenable(config)
print(f"Parámetros: {modelo.num_parametros():,}")

# Entrenamiento
modelo.entrenar(tok, texto, pasos=8000, tasa_aprendizaje=3e-4,
                dir_salida="data/run/", paciencia=3000)

# Generación
ids_out = modelo.generar(tok.codificar("En un lugar"), max_tokens=100, temperatura=0.8)
print(tok.decodificar(ids_out))
```

Ver [la guía de integración de bindings](docs/bindings.md) para instrucciones detalladas de instalación y la API completa.

---

## Menú interactivo (`molineteai.py`)

El lanzador principal ofrece un menú con todas las funcionalidades:

| Opción | Descripción |
|:---:|:---|
| `1` | Tokenizadores BPE — varios tamaños de vocabulario |
| `2` | Operaciones tensoriales — matmul, softmax, máscaras causales |
| `3` | Arquitectura GPT-2 — conteo de parámetros y benchmarks |
| `4` | Infraestructura de entrenamiento — data loaders y logging |
| `5` | Entrenar GPT-2 Diminuto (~170K params, ~5 min) |
| `6` | Entrenar GPT-2 Pequeño (~200K params, ~15 min) |
| `7` | Entrenar GPT-2 Mediano (~4M params, ~2 h) |
| `8` | Entrenar GPT-2 Small completo (~163M params) |
| `9` | Entrenador con presets (pocket-bard, spider, cyclops...) |
| `10` | Chat con modelo entrenado |
| `11` | Descargar corpus (Cervantes / Shakespeare) |

---

## Presentación

Este proyecto incluye una charla con **animaciones desarrolladas en Manim** que explora visualmente cómo se construye un Transformer desde cero. La presentación recorre paso a paso la tokenización, embeddings, mecanismos de self-attention, redes feed forward, conexiones residuales y generación autoregresiva — desmontando la "magia" detrás de los grandes modelos de lenguaje.

Los archivos de la presentación están en `examples/presentation/`.

---

## Diferencias frente al repositorio original

Este fork enriquece el proyecto base con los siguientes aportes:

1. **Corpus en español:** Sustitución de Shakespeare por Cervantes como datos de entrenamiento, con un script de descarga automatizado.
2. **Scripts experimentales:** Permiten aislar y observar el comportamiento interno de cada componente del modelo.
3. **Bindings Python completos:** API Python con PyO3/maturin para interactuar con el modelo desde Python.
4. **Lanzador interactivo:** Menú en consola (`molineteai.py`) con animaciones y todas las opciones del proyecto.
5. **Recursos visuales:** Presentación en Manim con animaciones de la arquitectura Transformer.
6. **Documentación en español:** Explicaciones adicionales orientadas a la comprensión del código.

---

## Licencia

Distribuido bajo la licencia **Apache 2.0**. Ver [LICENSE](LICENSE) para más detalles.

---

<div align="center">

*"No con quien naces, sino con quien paces."* — Miguel de Cervantes

</div>
