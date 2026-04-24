<div align="center">
<pre>
________________________________________________________________________________________________________8______________________
________________________________________________________________________________________________________8666___________________
________________________________________________________________________________________________________8_868__________________
________________________________________________________________________________________________________8_688_8________________
███╗░░░███╗░█████╗░██╗░░░░░██╗███╗░░██╗███████╗████████╗███████╗__░█████╗░██╗______88888________________8668888________________
████╗░████║██╔══██╗██║░░░░░██║████╗░██║██╔════╝╚══██╔══╝██╔════╝__██╔══██╗██║____888888888______________8___888_________88_8___
██╔████╔██║██║░░██║██║░░░░░██║██╔██╗██║█████╗░░░░░██║░░░█████╗░░__███████║██║____888_8888888____________8_____________8668888__
██║╚██╔╝██║██║░░██║██║░░░░░██║██║╚████║██╔══╝░░░░░██║░░░██╔══╝░░__██╔══██║██║_____668888888888_________86___________8688__8888_
██║░╚═╝░██║╚█████╔╝███████╗██║██║░╚███║███████╗░░░██║░░░███████╗__██║░░██║██║_______8668__888888________8_________868888888888_
╚═╝░░░░░╚═╝░╚════╝░╚══════╝╚═╝╚═╝░░╚══╝╚══════╝░░░╚═╝░░░╚══════╝__╚═╝░░╚═╝╚═╝_________86688__88888_____86_______868888888888___
________________________________________________________________________________________866888888888__88868___8688_8888888_____
__________________________________________________________________________________________866888888_86888888888_88888888_______
____________________________________________________________________________________________866888_88888886888__88888__________
_________________ /\ ____ ,, _________________________________________________________________86688688888888668__88____________
________  .---. __|| ____ /|| __________________________________________________________________86688888886888888______________
______ --'-----`--||    .'  \ __________________________________________________________________88666688666888888______________
________ {{{N `(  ||  .'    @  _______________________________________________________________8868868888688888888______________
________ {{{` _/  ||.'    |  \  ____________________________________________________________88888888866888688_8_88_____________
________ {{{.-.   ||  /  /\   \  __________________________________________________________888868_868_888_86668886_____________
________  {( )| .'||    /  `.  \  ________________________________________________________888_68866__888888_866__8_____________
__        {|\ \'  / )  / __  \\O| ______________________________________________________8888__8868____888888__86888____________
  `-.____.-| \ \ /\/  / ____ `'   ____________________________________________________88888__8668_______888888__866____________
 -    ////|  \ Y /|  | ______________________________________________________________88888__666888_______8888888__868__________
   |   |||||`-|\^/|| |   __________________________________________________________888888_86668_88__________888888__66_________
       |||||`-| " [] / ___________________________________________________________88888__66_68888_____________888888__68_______
      _ \\\\/`-|   []|\  ________________________________________________________88888__866__6_888__88888_______888886__668_____
 ) |`---``| _ |__([]| \   ______________________________________________________88888__86___688888__888_8________888886__8668___
  / ____  |/ `|  FJ|\ \   ______________________________________________________8888_866____686886_86688888__666__888868___868__
 / _____  `|  |  FJ) \ \    ______________________________________________________8_868____6886_88_68888886__868__8__866_88_____
/ ______   |  |  FJ|  \ )  ________________________________________________________86______6_68_88_6_8888_8__868___8__868_______
| ______   |  F  J  L  || ________________________________________________________________88866_88_8_6688_8__888___8___86_______
`.  ____   )-(> '----` || ________________________________________________________________6_888888_8_8688_8________8____68______
`.\  ___   | |    |||  || _______________________________________________________________86___6668868888888688___886688_86______
| \\  __   |-|    ||| / | _______________________________________________________________6888888668888888866888888_88688868_____
 \ )\    *_)/`-.__|| \\ | ____________________________________________________________8888888888888_88888_8888888_888888886_____
--'--'"""""`------''--'`'"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""888888888888888888888888888888888888888"""
</pre>

# Molinete AI

**Un Transformer GPT-2 implementado desde cero en Rust, entrenado con las obras de Miguel de Cervantes.**

[![Licencia](https://img.shields.io/badge/licencia-Apache_2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyO3](https://img.shields.io/badge/PyO3-maturin-red.svg)](https://github.com/PyO3/maturin)
[![Corpus](https://img.shields.io/badge/corpus-Cervantes-gold.svg)](DATA.md)

**Autor:** Jerónimo Hoyos Botero  
**Basado en:** [tag1consulting/feste](https://github.com/tag1consulting/feste)

</div>

---

## Tabla de contenidos

- [¿Qué es Molinete AI?](#qué-es-molinete-ai)
- [¿Por qué "Molinete"?](#por-qué-molinete)
- [Sobre el proyecto](#sobre-el-proyecto)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Inicio rápido](#inicio-rápido)
- [Módulos Python](#módulos-python)
- [Presentación](#presentación)
- [Diferencias frente al repositorio original](#diferencias-frente-al-repositorio-original)
- [Licencia](#licencia)

---

## ¿Qué es Molinete AI?

Molinete AI es un fork de **Feste**, una implementación desde cero de un modelo Transformer tipo GPT-2 en Rust, desarrollada por Tag1 Consulting como acompañamiento a la serie *Building an LLM From Scratch in Rust*.

Mientras que Feste entrena el modelo con las obras completas de Shakespeare, **Molinete AI propone entrenarlo con la obra de Miguel de Cervantes**, estableciendo una contraposición lingüística y cultural:

| Proyecto | Corpus | Idioma |
|:---|:---|:---|
| **Feste** | Shakespeare | Inglés isabelino |
| **Molinete AI** | Cervantes | Español del Siglo de Oro |

El objetivo no es solo replicar el experimento original, sino reinterpretarlo en español y convertirlo en una guía técnica rigurosa y accesible.

<div align="right"><a href="#molinete-ai">↑ Volver arriba</a></div>

---

## ¿Por qué "Molinete"?

Si Feste toma su identidad del bufón ingenioso de *Twelfth Night*, **Molinete AI** rinde homenaje a los famosos molinos de viento que el ingenioso hidalgo Don Quijote confundió con fieros gigantes — una metáfora perfecta para un modelo de lenguaje que intenta imitar la grandeza de algo mucho más vasto que él mismo.

<div align="right"><a href="#molinete-ai">↑ Volver arriba</a></div>

---

## Sobre el proyecto

Un modelo Transformer completamente entrenable, implementado desde cero en Rust, **sin depender de frameworks de deep learning** como PyTorch o TensorFlow.

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

<div align="right"><a href="#molinete-ai">↑ Volver arriba</a></div>

---

## Estructura del repositorio

```
molineteai/
├── src/                        ← Implementación del modelo en Rust
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
├── ejemplos/                   ← Scripts de exploración en Python
│   ├── molineteai.py           ← Punto de entrada principal Python
│   ├── REFERENCIA.md           ← Documentación completa de la API
│   └── modulos/
│       ├── arquitectura.py     ← Exploración de la arquitectura del modelo
│       ├── chat.py             ← Interfaz de chat con el modelo entrenado
│       ├── datos.py            ← Descarga y preprocesamiento del corpus
│       ├── entrenamiento.py    ← Bucle de entrenamiento paso a paso
│       ├── infraestructura.py  ← Warmup, clipping y utilidades de entrenamiento
│       ├── tensores.py         ← Operaciones tensoriales básicas
│       ├── tokenizadores.py    ← Entrenamiento y uso del tokenizador BPE
│       └── ui.py               ← Interfaz de usuario y utilidades de consola
├── presentation/               ← Animaciones Manim de la arquitectura
│   ├── main.py                 ← Escenas de la presentación
│   ├── README.md               ← Instrucciones de la presentación
│   └── requirements.txt
├── cervantes.txt               ← Corpus de entrenamiento (Cervantes)
├── Cargo.toml
├── Cargo.lock
├── pyproject.toml
├── DATA.md                     ← Guía del corpus de datos
└── README.md
```

<div align="right"><a href="#molinete-ai">↑ Volver arriba</a></div>

---

## Inicio rápido

### Requisitos

- [Rust](https://rustup.rs/) 1.75+
- Python 3.9+
- `maturin` para compilar los bindings PyO3:

```bash
pip install maturin
```

### 1. Clonar el repositorio

```bash
git clone https://github.com/JeroHoyos/Molinete-AI.git
cd Molinete-AI
```

### 2. Preparar el corpus

El corpus de Cervantes debe descargarse antes de entrenar. Ver [DATA.md](DATA.md) para instrucciones detalladas y fuentes recomendadas. También puede usarse el menú interactivo del proyecto:

```bash
python ejemplos/molineteai.py
# → Opción 11: Descargar corpus
```

### 3. Compilar los bindings Python

```bash
maturin develop --release
```

### 4. Ejecutar el modelo

```bash
python ejemplos/molineteai.py
```

<div align="right"><a href="#molinete-ai">↑ Volver arriba</a></div>

---

## Módulos Python

Una vez compilados los bindings con `maturin develop --release`, los módulos en `ejemplos/modulos/` permiten explorar cada componente del sistema de forma aislada:

| Módulo | Descripción |
|:---|:---|
| `tensores.py` | Operaciones tensoriales: multiplicación de matrices, strides, broadcasting |
| `tokenizadores.py` | Entrenamiento del tokenizador BPE y exploración del vocabulario |
| `arquitectura.py` | Inspección de la arquitectura: capas, parámetros, flujo de datos |
| `infraestructura.py` | Warmup, gradient clipping y utilidades del bucle de entrenamiento |
| `entrenamiento.py` | Entrenamiento completo con registro de métricas |
| `datos.py` | Carga, preprocesamiento y exploración del corpus |
| `chat.py` | Chat interactivo con un modelo ya entrenado |
| `ui.py` | Interfaz para generación de texto con control de temperatura |

Para la documentación completa de la API Python, ver [ejemplos/REFERENCIA.md](ejemplos/REFERENCIA.md).

### Ejemplo de uso desde Python

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

<div align="right"><a href="#molinete-ai">↑ Volver arriba</a></div>

---

## Presentación

La carpeta `presentation/` contiene una charla con **animaciones desarrolladas en Manim** que explora visualmente cómo se construye un Transformer desde cero: tokenización, embeddings, mecanismos de self-attention, redes feed forward, conexiones residuales y generación autoregresiva.

```bash
cd presentation
pip install -r requirements.txt
py -m manim_slides render main.py Presentacion
py -m manim_slides present Presentacion
```

Ver [presentation/README.md](presentation/README.md) para instrucciones detalladas y controles de la presentación.

<div align="right"><a href="#molinete-ai">↑ Volver arriba</a></div>

---

## Diferencias frente al repositorio original

| Aporte | Descripción |
|:---|:---|
| **Corpus en español** | Cervantes en lugar de Shakespeare, con guía de descarga en [DATA.md](DATA.md) |
| **Módulos de exploración** | Scripts Python que aíslan y demuestran el comportamiento de cada componente |
| **Bindings Python** | API completa con PyO3/maturin para usar el modelo desde Python |
| **Presentación Manim** | Animaciones de la arquitectura Transformer para uso pedagógico |
| **Documentación en español** | Explicaciones adicionales orientadas a la comprensión del código |

<div align="right"><a href="#molinete-ai">↑ Volver arriba</a></div>

---

## Licencia

Distribuido bajo la licencia **Apache 2.0**. Ver [LICENSE](LICENSE) para más detalles.

---

<div align="center">

*"En un lugar de la Mancha, de cuyo nombre no quiero acordarme..."*

</div>
