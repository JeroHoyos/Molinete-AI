# Molinete AI — Referencia Completa de la Librería

[![PyO3](https://img.shields.io/badge/PyO3-maturin-red.svg)](https://github.com/PyO3/maturin)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)

Molinete AI es una implementación educativa de un transformer GPT-2 escrita en Rust y compilada como módulo Python via PyO3. Todos los pesos, capas, optimizador y backpropagation están implementados desde cero, sin PyTorch ni TensorFlow.

```python
import molineteai
```

---

## Índice

- [Tensor](#tensor)
- [TokenizadorBPE](#tokenizadorbpe)
- [Config](#config)
- [GPT2](#gpt2)
- [GPT2Entrenable](#gpt2entrenable)
- [Funciones de módulo](#funciones-de-módulo)
- [Ejemplo completo end-to-end](#ejemplo-completo-end-to-end)
- [Instalación](#instalación)

---

## Tensor

Arreglo N-dimensional de `float32`. Todas las operaciones devuelven un nuevo `Tensor` (son inmutables).

### Creación

```python
# Constructor: datos aplanados + lista de dimensiones
t = molineteai.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
print(t.forma)   # [2, 3]
print(t.datos)   # [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

# Tensor de ceros
ceros = molineteai.Tensor.ceros([4, 4])

# Rango de enteros [start, end)
rango = molineteai.Tensor.arange(0, 10)
print(rango.datos)  # [0.0, 1.0, ..., 9.0]
```

| Método | Firma | Descripción |
|---|---|---|
| `Tensor(datos, forma)` | `(list[float], list[int]) → Tensor` | Crea tensor desde lista aplanada |
| `Tensor.ceros(forma)` | `(list[int]) → Tensor` | Tensor lleno de ceros |
| `Tensor.arange(start, end)` | `(int, int) → Tensor` | Rango `[start, end)` como tensor 1-D |

### Propiedades

```python
t = molineteai.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
print(t.forma)   # [2, 2]
print(t.datos)   # [1.0, 2.0, 3.0, 4.0]
print(t.numel()) # 4
```

### Álgebra Lineal

```python
a = molineteai.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
b = molineteai.Tensor([1.0, 0.0, 0.0, 1.0], [2, 2])  # identidad

c = a.matmul(b)
print(c.datos)  # [1.0, 2.0, 3.0, 4.0]  — igual que A

# Matrices grandes: el backend usa bloqueo de caché + SIMD automáticamente
grande = molineteai.Tensor([1.0] * (256 * 256), [256, 256])
resultado = grande.matmul(grande)
```

### Operaciones Elemento a Elemento

```python
x = molineteai.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
y = molineteai.Tensor([2.0, 2.0, 2.0, 2.0], [2, 2])

print(x.add(y).datos)   # [3.0, 4.0, 5.0, 6.0]
print(x.sub(y).datos)   # [-1.0, 0.0, 1.0, 2.0]
print(x.mul(y).datos)   # [2.0, 4.0, 6.0, 8.0]
print(x.div(y).datos)   # [0.5, 1.0, 1.5, 2.0]
print(x.sqrt().datos)   # [1.0, 1.414, 1.732, 2.0]
```

### Operaciones Escalares

```python
x = molineteai.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])

print(x.add_scalar(10.0).datos)  # [11.0, 12.0, 13.0, 14.0]
print(x.mul_scalar(3.0).datos)   # [3.0, 6.0, 9.0, 12.0]
print(x.div_scalar(2.0).datos)   # [0.5, 1.0, 1.5, 2.0]
```

### Broadcasting

El broadcasting se aplica automáticamente en la última dimensión:

```python
matriz = molineteai.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
sesgo  = molineteai.Tensor([0.1, 0.2, 0.3], [3])

resultado = matriz.add(sesgo)
print(resultado.datos)  # [1.1, 2.2, 3.3, 4.1, 5.2, 6.3]
```

### Forma

```python
t = molineteai.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])

# Reshape: el número de elementos debe conservarse
r = t.reshape([3, 2])
print(r.forma)  # [3, 2]

aplanado = t.reshape([6])
print(aplanado.forma)  # [6]

# Transpose: intercambia dos ejes
trans = t.transpose(0, 1)
print(trans.forma)  # [3, 2]
```

### Estadísticas

```python
t = molineteai.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])

# mean(eje, mantener_dimensión)
medias = t.mean(-1, True)   # media por fila, mantiene shape [2, 1]
print(medias.datos)         # [2.0, 5.0]

# var(eje, mantener_dimensión)  — varianza sin sesgo (ddof=0)
varianzas = t.var(-1, True)
print([round(v, 4) for v in varianzas.datos])  # [0.6667, 0.6667]
```

### Softmax y Enmascaramiento

```python
# Softmax numéricamente estable (resta el máximo antes de exp)
logits = molineteai.Tensor([1.0, 2.0, 3.0, 4.0], [1, 4])
probs  = logits.softmax(-1)
print(sum(probs.datos))  # 1.0

# También funciona con valores muy grandes (sin overflow)
grandes = molineteai.Tensor([1000.0, 2000.0, 3000.0], [1, 3])
print(grandes.softmax(-1).datos)  # [0.0, 0.0, 1.0]

# masked_fill: reemplaza posiciones donde máscara != 0 con un valor
scores  = molineteai.Tensor([0.5, 0.8, 0.3, 0.9], [2, 2])
mascara = molineteai.Tensor([0.0, 1.0,   # pos 0 no puede ver la pos 1
                              0.0, 0.0],  # pos 1 puede ver todo
                             [2, 2])
enmascarado = scores.masked_fill(mascara, float("-inf"))
# luego softmax convierte -inf en probabilidad 0
```

### Concatenación

```python
a = molineteai.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
b = molineteai.Tensor([5.0, 6.0, 7.0, 8.0], [2, 2])

# concat(otro, eje)
c = a.concat(b, 0)  # apilar filas → [4, 2]
d = a.concat(b, 1)  # apilar columnas → [2, 4]
print(c.forma)  # [4, 2]
print(d.forma)  # [2, 4]
```

### Tabla resumen — Tensor

| Método | Descripción |
|---|---|
| `.numel()` | Total de elementos |
| `.add(otro)` / `.sub(otro)` / `.mul(otro)` / `.div(otro)` | Ops elemento a elemento con broadcasting |
| `.sqrt()` | Raíz cuadrada elemento a elemento |
| `.add_scalar(f)` / `.mul_scalar(f)` / `.div_scalar(f)` | Ops con escalar |
| `.matmul(otro)` | Multiplicación de matrices (SIMD + cache-blocking) |
| `.softmax(eje)` | Softmax numéricamente estable |
| `.mean(eje, keepdim)` | Media por eje |
| `.var(eje, keepdim)` | Varianza por eje |
| `.reshape(forma)` | Cambia la forma sin mover datos |
| `.transpose(eje_a, eje_b)` | Intercambia dos ejes |
| `.masked_fill(mascara, valor)` | Reemplaza con `valor` donde `mascara != 0` |
| `.concat(otro, eje)` | Concatena a lo largo de un eje |

<div align="right"><a href="#molinete-ai--referencia-completa-de-la-librería">↑ Volver arriba</a></div>

---

## TokenizadorBPE

Tokenizador Byte-Pair Encoding (BPE) entrenado desde cero, compatible con cualquier corpus de texto.

### Entrenamiento

```python
with open("cervantes.txt", encoding="utf-8") as f:
    texto = f.read()

# Crear e inicializar el tokenizador
tok = molineteai.TokenizadorBPE(vocab_size=1024)

# Entrenar — aprende fusiones de pares de bytes hasta vocab_size
tok.entrenar(texto, 1024)

print(tok.tam_vocabulario())  # vocabulario aprendido (puede diferir levemente del target)
```

### Codificación y Decodificación

```python
# Texto → lista de IDs enteros
ids = tok.codificar("En un lugar de la Mancha")
print(ids)  # [42, 317, 8, 205, ...]

# IDs → texto  (ciclo de ida y vuelta perfecto)
texto_rec = tok.decodificar(ids)
assert texto_rec == "En un lugar de la Mancha"

# Verificar que la reconstrucción es fiel
def verificar_ciclo(tok, texto):
    return tok.decodificar(tok.codificar(texto)) == texto

print(verificar_ciclo(tok, "Don Quijote de la Mancha"))  # True
```

### Estadísticas y Análisis

```python
# Diccionario con info del vocabulario
stats = tok.estadisticas()
print(stats)
# {'tam_vocabulario': 1024, 'num_fusiones': 768, 'tokens_base': 256}

# Imprime un análisis detallado del vocabulario aprendido
tok.analizar_vocabulario(texto[:5000])

# Propiedad directa
print(tok.tam_vocabulario())  # entero
```

### Persistencia

```python
# Guardar a JSON (incluye vocab completo + tabla de fusiones)
tok.guardar("data/tokenizador_1024.json")

# Cargar tokenizador guardado
tok2 = molineteai.TokenizadorBPE.cargar("data/tokenizador_1024.json")
print(tok2.tam_vocabulario())  # 1024
```

### Tabla resumen — TokenizadorBPE

| Método | Firma | Descripción |
|---|---|---|
| `TokenizadorBPE(vocab_size)` | `(int) → TokenizadorBPE` | Crea tokenizador vacío |
| `.entrenar(texto, vocab_size)` | `(str, int)` | Aprende fusiones BPE sobre el corpus |
| `.codificar(texto)` | `(str) → list[int]` | Texto → secuencia de IDs |
| `.decodificar(ids)` | `(list[int]) → str` | IDs → texto |
| `.tam_vocabulario()` | `() → int` | Tamaño del vocabulario aprendido |
| `.estadisticas()` | `() → dict` | `{tam_vocabulario, num_fusiones, tokens_base}` |
| `.analizar_vocabulario(texto)` | `(str)` | Imprime análisis del vocab |
| `.guardar(ruta)` | `(str)` | Serializa a JSON |
| `TokenizadorBPE.cargar(ruta)` | `(str) → TokenizadorBPE` | Carga desde JSON |

<div align="right"><a href="#molinete-ai--referencia-completa-de-la-librería">↑ Volver arriba</a></div>

---

## Config

Define la arquitectura del modelo. Todas las configuraciones predefinidas están ajustadas para entrenamiento desde cero.

### Constructores predefinidos

```python
vocab = 1024  # tamaño del vocabulario (se obtiene del tokenizador)

# ~50K parámetros — entrena en 2-5 minutos
cfg_dim  = molineteai.Config.diminuta(vocab)

# ~200K parámetros — entrena en 15-20 minutos
cfg_peq  = molineteai.Config.pequena(vocab)

# ~4M parámetros — entrena en 1-3 horas
cfg_med  = molineteai.Config.mediana(vocab)

# ~163M parámetros — GPT-2 Small original de OpenAI (entrenamiento largo)
cfg_gpt2 = molineteai.Config.gpt2_small(vocab)
```

### Propiedades

```python
cfg = molineteai.Config.mediana(1024)

print(cfg.tam_vocabulario)  # 1024
print(cfg.n_embd)           # dimensión de embeddings
print(cfg.n_capas)          # número de bloques transformer
print(cfg.n_cabezas)        # número de cabezas de atención
print(cfg.tam_bloque)       # longitud máxima de contexto (tokens)
print(cfg.tasa_dropout)     # tasa de dropout (0.0 durante inferencia)
```

### Comparativa de configuraciones

| Nombre | Paráms | n_embd | Cabezas | Capas | Contexto | Tiempo aprox. |
|---|---|---|---|---|---|---|
| `diminuta` | ~50K | 64 | 4 | 4 | 256 | 2-5 min |
| `pequena` | ~200K | 128 | 4 | 4 | 256 | 15-20 min |
| `mediana` | ~4M | 256 | 8 | 6 | 256 | 1-3 h |
| `gpt2_small` | ~163M | 768 | 12 | 12 | 1024 | horas/días |

### Tabla resumen — Config

| Método | Descripción |
|---|---|
| `Config.diminuta(vocab_size)` | Configuración mínima, entrenamiento rápido |
| `Config.pequena(vocab_size)` | Configuración pequeña, buen punto de partida |
| `Config.mediana(vocab_size)` | Configuración media, resultados de calidad |
| `Config.gpt2_small(vocab_size)` | Réplica exacta de GPT-2 Small de OpenAI |

<div align="right"><a href="#molinete-ai--referencia-completa-de-la-librería">↑ Volver arriba</a></div>

---

## GPT2

Modelo de **inferencia únicamente** (sin backpropagation). Útil para benchmarks y exploración de la arquitectura.

### Uso básico

```python
cfg    = molineteai.Config.diminuta(512)
modelo = molineteai.GPT2(cfg)

# forward: lista de listas de IDs → logits aplanados
tokens = [[1, 2, 3, 4, 5, 6, 7, 8],   # muestra 1
          [10, 20, 30, 40, 50, 60, 70, 80]]  # muestra 2
logits = modelo.forward(tokens)

# Obtener la forma del tensor de salida [batch, seq_len, vocab]
batch, seq_len = 2, 8
B, T, V = modelo.forma_salida(batch, seq_len)
print(f"Salida: [{B}, {T}, {V}]")  # [2, 8, 512]

# Contar parámetros del modelo cargado
print(modelo.num_parametros())  # ~50K
```

### Tabla resumen — GPT2

| Método | Firma | Descripción |
|---|---|---|
| `GPT2(config)` | `(Config) → GPT2` | Crea modelo con pesos aleatorios |
| `.forward(tokens)` | `(list[list[int]]) → list[float]` | Forward pass, devuelve logits aplanados |
| `.forma_salida(batch, seq_len)` | `(int, int) → (int, int, int)` | Shape del tensor de salida `[B, T, V]` |
| `.num_parametros()` | `() → int` | Total de parámetros del modelo |

<div align="right"><a href="#molinete-ai--referencia-completa-de-la-librería">↑ Volver arriba</a></div>

---

## GPT2Entrenable

Modelo **entrenable** con backpropagation completo. Incluye generación de texto y persistencia de checkpoints.

### Crear y entrenar

```python
import molineteai

# 1. Preparar corpus y tokenizador
with open("cervantes.txt", encoding="utf-8") as f:
    texto = f.read()

tok = molineteai.TokenizadorBPE(1024)
tok.entrenar(texto, 1024)

# 2. Crear modelo
cfg    = molineteai.Config.diminuta(tok.tam_vocabulario())
modelo = molineteai.GPT2Entrenable(cfg)
print(f"Parámetros: {modelo.num_parametros():,}")

# 3. Entrenar (todos los argumentos con sus valores por defecto)
modelo.entrenar(
    tokenizador        = tok,
    texto              = texto,
    pasos              = 10_000,      # pasos de optimización
    tasa_aprendizaje   = 3e-4,        # LR máximo (cosine decay)
    long_secuencia     = 256,         # tokens de contexto por muestra
    dir_salida         = "checkpoints/",  # None → solo stdout
    paciencia          = 5_000,       # early stopping (pasos sin mejora)
    fraccion_calentamiento = 0.1,     # warmup lineal: 10% de los pasos
    norma_recorte      = 1.0,         # clipping de gradientes (L2)
    fraccion_validacion = 0.1,        # 10% del corpus reservado para val
    decaimiento_peso   = 0.01,        # weight decay en AdamW
)
```

### Generar texto

```python
# Codificar el prompt
prompt = "Don Quijote salió"
ids_prompt = tok.codificar(prompt)

# Generar tokens nuevos
ids_gen = modelo.generar(
    prompt_ids  = ids_prompt,
    max_tokens  = 200,       # tokens a generar (sin contar el prompt)
    temperature = 0.8,       # <1 más conservador, >1 más creativo
)

# Decodificar resultado completo
print(tok.decodificar(ids_gen))
```

**Temperature y creatividad:**

```python
# Determinista (temperatura baja)
ids = modelo.generar(ids_prompt, max_tokens=100, temperature=0.3)

# Balanceado
ids = modelo.generar(ids_prompt, max_tokens=100, temperature=0.8)

# Creativo / caótico (temperatura alta)
ids = modelo.generar(ids_prompt, max_tokens=100, temperature=1.5)
```

### Guardar y cargar

```python
# Guardar solo el modelo (sin tokenizador)
modelo.guardar("checkpoints/modelo_final.bin")

# Cargar modelo guardado (necesitas reconstruir Config manualmente)
modelo2 = molineteai.GPT2Entrenable.cargar("checkpoints/modelo_final.bin")

# Cargar checkpoint completo (modelo + tokenizador opcional)
modelo3, tok3 = molineteai.GPT2Entrenable.cargar_checkpoint("checkpoints/")
# tok3 puede ser None si el checkpoint no incluye tokenizador
if tok3 is not None:
    print("Tokenizador restaurado automáticamente")
```

### Flujo completo de entrenamiento a generación

```python
import molineteai

# --- Preparación ---
with open("cervantes.txt", encoding="utf-8") as f:
    texto = f.read()

tok = molineteai.TokenizadorBPE(1024)
tok.entrenar(texto, 1024)
tok.guardar("checkpoints/tok.json")

cfg    = molineteai.Config.diminuta(tok.tam_vocabulario())
modelo = molineteai.GPT2Entrenable(cfg)

# --- Entrenamiento ---
modelo.entrenar(
    tokenizador        = tok,
    texto              = texto,
    pasos              = 5_000,
    tasa_aprendizaje   = 3e-4,
    dir_salida         = "checkpoints/",
)

# --- Generación ---
prompt_ids = tok.codificar("En un lugar de la Mancha")
output_ids = modelo.generar(prompt_ids, max_tokens=150, temperature=0.9)
print(tok.decodificar(output_ids))
```

### Tabla resumen — GPT2Entrenable

| Método | Firma | Descripción |
|---|---|---|
| `GPT2Entrenable(config)` | `(Config) → GPT2Entrenable` | Crea modelo entrenable |
| `.num_parametros()` | `() → int` | Total de parámetros |
| `.entrenar(tok, texto, ...)` | Ver tabla abajo | Bucle de entrenamiento completo |
| `.generar(prompt_ids, max_tokens, temperature)` | `(list[int], int, float) → list[int]` | Genera texto desde un prompt |
| `.guardar(ruta)` | `(str)` | Serializa pesos a binario |
| `GPT2Entrenable.cargar(ruta)` | `(str) → GPT2Entrenable` | Carga pesos desde binario |
| `GPT2Entrenable.cargar_checkpoint(dir)` | `(str) → (GPT2Entrenable, TokenizadorBPE?)` | Carga modelo y tokenizador |

**Parámetros de `.entrenar()`:**

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `tokenizador` | `TokenizadorBPE` | — | Tokenizador ya entrenado |
| `texto` | `str` | — | Corpus de texto completo |
| `pasos` | `int` | `10_000` | Pasos de optimización |
| `tasa_aprendizaje` | `float` | `3e-4` | LR máximo (se aplica cosine decay) |
| `long_secuencia` | `int` | `256` | Longitud de contexto por lote |
| `dir_salida` | `str?` | `None` | Directorio para logs y checkpoints |
| `paciencia` | `int` | `5_000` | Pasos sin mejora antes de parar |
| `fraccion_calentamiento` | `float` | `0.1` | Fracción del total usada en warmup |
| `norma_recorte` | `float` | `1.0` | Norma L2 máxima de gradientes |
| `fraccion_validacion` | `float` | `0.1` | Fracción del corpus reservada para val |
| `decaimiento_peso` | `float` | `0.01` | Weight decay en AdamW |

---

## Funciones de Módulo

Funciones globales disponibles directamente en el módulo `molineteai`.

### `dividir_entrenamiento_validacion`

Divide una secuencia de tokens en conjunto de entrenamiento y validación.

```python
tok = molineteai.TokenizadorBPE(512)
tok.entrenar(texto, 512)
todos_los_tokens = tok.codificar(texto)

tokens_train, tokens_val = molineteai.dividir_entrenamiento_validacion(
    todos_los_tokens,
    fraccion_val = 0.1   # 10% para validación
)

print(f"Train: {len(tokens_train):,} tokens")
print(f"Val:   {len(tokens_val):,} tokens")
```

La división es determinista (sin mezclar aleatoriamente): el 10% final va a validación para respetar el orden temporal del texto.

### `contar_parametros_config`

Estima el número de parámetros del modelo **sin instanciarlo**. Útil para comparar configuraciones antes de alocar memoria.

```python
cfg = molineteai.Config.mediana(1024)
n   = molineteai.contar_parametros_config(cfg)
print(f"Parámetros: {n:,}")        # ~4M
print(f"Memoria:    {n*4/1e6:.1f} MB")  # float32 = 4 bytes

# Comparar todos los tamaños
for nombre, fn in [
    ("diminuta",   molineteai.Config.diminuta),
    ("pequena",    molineteai.Config.pequena),
    ("mediana",    molineteai.Config.mediana),
    ("gpt2_small", molineteai.Config.gpt2_small),
]:
    cfg = fn(1024)
    n   = molineteai.contar_parametros_config(cfg)
    print(f"{nombre:<12} {n:>12,} params  {n*4/1e6:>7.1f} MB")
```

### Tabla resumen — Funciones de módulo

| Función | Firma | Descripción |
|---|---|---|
| `dividir_entrenamiento_validacion(tokens, fraccion_val)` | `(list[int], float) → (list[int], list[int])` | Split train/val determinista |
| `contar_parametros_config(config)` | `(Config) → int` | Cuenta parámetros sin crear el modelo |

<div align="right"><a href="#molinete-ai--referencia-completa-de-la-librería">↑ Volver arriba</a></div>

---

## Ejemplo Completo End-to-End

```python
import molineteai

# 1. Cargar corpus
with open("cervantes.txt", encoding="utf-8") as f:
    texto = f.read()
print(f"Corpus: {len(texto)/1e6:.2f} MB")

# 2. Tokenizador BPE
tok = molineteai.TokenizadorBPE(1024)
tok.entrenar(texto, 1024)
print(f"Vocabulario: {tok.tam_vocabulario()} tokens")
print(f"Stats: {tok.estadisticas()}")

# 3. Tokenizar y dividir datos
tokens = tok.codificar(texto)
tokens_train, tokens_val = molineteai.dividir_entrenamiento_validacion(tokens, 0.1)
print(f"Train: {len(tokens_train):,} | Val: {len(tokens_val):,}")

# 4. Crear modelo y ver parámetros
cfg    = molineteai.Config.pequena(tok.tam_vocabulario())
n_par  = molineteai.contar_parametros_config(cfg)
print(f"Modelo con {n_par:,} parámetros")

modelo = molineteai.GPT2Entrenable(cfg)

# 5. Entrenar
modelo.entrenar(
    tokenizador        = tok,
    texto              = texto,
    pasos              = 3_000,
    tasa_aprendizaje   = 3e-4,
    dir_salida         = "checkpoints/",
)

# 6. Guardar
tok.guardar("checkpoints/tokenizador.json")
modelo.guardar("checkpoints/modelo.bin")

# 7. Cargar y generar
modelo_cargado = molineteai.GPT2Entrenable.cargar("checkpoints/modelo.bin")
tok_cargado    = molineteai.TokenizadorBPE.cargar("checkpoints/tokenizador.json")

prompt  = tok_cargado.codificar("Sancho Panza respondió")
ids_gen = modelo_cargado.generar(prompt, max_tokens=200, temperature=0.85)
print(tok_cargado.decodificar(ids_gen))
```

---

## Instalación

```bash
# Requiere Rust + maturin
pip install maturin
maturin develop --release   # compila e instala el módulo
```

Una vez instalado, `import molineteai` funciona desde cualquier script Python en el entorno activo.

<div align="right"><a href="#molinete-ai--referencia-completa-de-la-librería">↑ Volver arriba</a></div>
