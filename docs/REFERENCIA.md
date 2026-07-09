# Referencia de la API Python

`molineteai` es el modelo Rust del proyecto compilado como módulo de Python (PyO3). Tensores, capas, optimizador y backpropagation están implementados desde cero, sin PyTorch ni TensorFlow. Se instala junto con el resto del proyecto con `uv sync` (ver el [README](../README.md)).

```python
import molineteai
```

## Índice

- [Tensor](#tensor)
- [TokenizadorBPE](#tokenizadorbpe)
- [Config](#config)
- [GPT2](#gpt2)
- [GPT2Entrenable](#gpt2entrenable)
- [Funciones de módulo](#funciones-de-módulo)
- [Ejemplo completo](#ejemplo-completo)

---

## Tensor

Arreglo N-dimensional de `float32`. Todas las operaciones devuelven un `Tensor` nuevo (son inmutables) y el broadcasting se aplica automáticamente en la última dimensión.

```python
t = molineteai.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
print(t.forma)   # [2, 3]
print(t.datos)   # [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

# Broadcasting en la última dimensión
sesgo = molineteai.Tensor([0.1, 0.2, 0.3], [3])
print(t.add(sesgo).datos)   # [1.1, 2.2, 3.3, 4.1, 5.2, 6.3]

# Forma y álgebra lineal
c = t.transpose(0, 1).matmul(t)   # [3, 3]

# Softmax numéricamente estable (base de la atención)
logits = molineteai.Tensor([1.0, 2.0, 3.0, 4.0], [1, 4])
print(sum(logits.softmax(-1).datos))   # 1.0
```

| Método | Descripción |
|---|---|
| `Tensor(datos, forma)` | Crea un tensor desde una lista aplanada y sus dimensiones |
| `Tensor.ceros(forma)` | Tensor lleno de ceros |
| `Tensor.arange(inicio, fin)` | Rango `[inicio, fin)` como tensor 1-D |
| `.forma` / `.datos` / `.numel()` | Dimensiones, datos aplanados y total de elementos |
| `.add(t)` / `.sub(t)` / `.mul(t)` / `.div(t)` | Operaciones elemento a elemento con broadcasting |
| `.add_scalar(f)` / `.mul_scalar(f)` / `.div_scalar(f)` | Operaciones con un escalar |
| `.sqrt()` | Raíz cuadrada elemento a elemento |
| `.matmul(t)` | Multiplicación de matrices (SIMD y bloqueo de caché) |
| `.softmax(eje)` | Softmax numéricamente estable |
| `.mean(eje, keepdim)` / `.var(eje, keepdim)` | Media y varianza por eje |
| `.reshape(forma)` / `.transpose(eje_a, eje_b)` | Cambia la forma / intercambia dos ejes |
| `.masked_fill(mascara, valor)` | Escribe `valor` donde `mascara != 0` (máscara causal de atención) |
| `.concat(t)` | Concatena a lo largo de la primera dimensión |

---

## TokenizadorBPE

Tokenizador Byte-Pair Encoding entrenado desde cero sobre cualquier corpus. La reconstrucción es exacta: `decodificar(codificar(texto)) == texto`.

```python
texto = open("cervantes.txt", encoding="utf-8").read()

tok = molineteai.TokenizadorBPE(1024)
tok.entrenar(texto, 1024)

ids = tok.codificar("En un lugar de la Mancha")
print(tok.decodificar(ids))   # "En un lugar de la Mancha"
print(tok.estadisticas())     # {'tam_vocabulario': 1024, 'num_fusiones': 768, 'tokens_base': 256}

tok.guardar("checkpoints/tok.json")
tok2 = molineteai.TokenizadorBPE.cargar("checkpoints/tok.json")
```

| Método | Descripción |
|---|---|
| `TokenizadorBPE(tam_vocabulario)` | Crea un tokenizador vacío |
| `.entrenar(texto, tam_vocabulario)` | Aprende las fusiones BPE sobre el corpus |
| `.codificar(texto)` / `.decodificar(ids)` | Texto a IDs y viceversa |
| `.tam_vocabulario()` | Tamaño del vocabulario aprendido |
| `.estadisticas()` | Diccionario con vocabulario, fusiones y tokens base |
| `.analizar_vocabulario(texto)` | Imprime un análisis del vocabulario |
| `.guardar(ruta)` / `TokenizadorBPE.cargar(ruta)` | Persistencia en JSON |

---

## Config

Define la arquitectura del modelo mediante cuatro configuraciones predefinidas:

| Constructor | Paráms | n_embd | Capas | Cabezas | Contexto | Entrenamiento aprox. |
|---|---|---|---|---|---|---|
| `Config.diminuta(vocab)` | ~50K | 64 | 4 | 4 | 256 | 2-5 min |
| `Config.pequena(vocab)` | ~200K | 128 | 4 | 4 | 256 | 15-20 min |
| `Config.mediana(vocab)` | ~4M | 256 | 6 | 8 | 256 | 1-3 h |
| `Config.gpt2_small(vocab)` | ~163M | 768 | 12 | 12 | 1024 | horas/días |

```python
cfg = molineteai.Config.mediana(tok.tam_vocabulario())

# Propiedades de solo lectura
print(cfg.tam_vocabulario, cfg.n_embd, cfg.n_capas,
      cfg.n_cabezas, cfg.tam_bloque, cfg.tasa_dropout)

# También se puede definir una arquitectura a medida
cfg = molineteai.Config(tam_vocabulario=1024, n_embd=128, n_capas=4,
                        n_cabezas=4, tam_bloque=256, tasa_dropout=0.1)
```

---

## GPT2

Modelo de inferencia únicamente, sin backpropagation. Útil para explorar la arquitectura y medir el forward pass.

```python
modelo = molineteai.GPT2(molineteai.Config.diminuta(512))

logits = modelo.forward([[1, 2, 3, 4], [5, 6, 7, 8]])   # logits aplanados
print(modelo.forma_salida(2, 4))   # (2, 4, 512) = [lote, secuencia, vocabulario]
print(modelo.num_parametros())
```

| Método | Descripción |
|---|---|
| `GPT2(config)` | Crea el modelo con pesos aleatorios |
| `.forward(tokens)` | Forward pass sobre `list[list[int]]`, devuelve logits aplanados |
| `.forma_salida(lote, secuencia)` | Forma `[B, T, V]` del tensor de salida |
| `.num_parametros()` | Total de parámetros |

---

## GPT2Entrenable

Modelo completo: backpropagation, generación de texto y checkpoints. El uso de principio a fin está en el [ejemplo completo](#ejemplo-completo).

| Método | Descripción |
|---|---|
| `GPT2Entrenable(config)` | Crea el modelo entrenable |
| `.num_parametros()` | Total de parámetros |
| `.entrenar(tokenizador, texto, ...)` | Bucle de entrenamiento completo (parámetros abajo) |
| `.generar(prompt_ids, max_tokens, temperatura)` | Genera tokens desde un prompt; temperatura <1 conservador, >1 creativo |
| `.guardar(ruta)` / `GPT2Entrenable.cargar(ruta)` | Guarda o carga solo los pesos (binario) |
| `GPT2Entrenable.cargar_checkpoint(ruta)` | Devuelve `(modelo, tokenizador o None)` desde un checkpoint |

El entrenamiento incluye warmup lineal, decaimiento coseno del LR, recorte de gradientes, early stopping y logging CSV. Parámetros de `.entrenar()`:

| Parámetro | Default | Descripción |
|---|---|---|
| `tokenizador` | requerido | Tokenizador ya entrenado |
| `texto` | requerido | Corpus de texto completo |
| `pasos` | `10_000` | Pasos de optimización |
| `tasa_aprendizaje` | `3e-4` | LR máximo |
| `long_secuencia` | `tam_bloque` de la config | Tokens de contexto por muestra |
| `dir_salida` | `None` | Directorio para checkpoints y logs; `None` = solo stdout |
| `paciencia` | `5_000` | Pasos sin mejora antes de parar |
| `fraccion_calentamiento` | `0.1` | Fracción de los pasos usada en warmup |
| `norma_recorte` | `1.0` | Norma L2 máxima de los gradientes |
| `fraccion_validacion` | `0.1` | Fracción del corpus reservada para validación |
| `decaimiento_peso` | `0.01` | Weight decay de AdamW |

---

## Funciones de módulo

| Función | Descripción |
|---|---|
| `dividir_entrenamiento_validacion(tokens, fraccion_val)` | Split train/val determinista: la fracción final va a validación, respetando el orden del texto |
| `contar_parametros_config(config)` | Estima los parámetros de una config sin instanciar el modelo |

```python
tokens = tok.codificar(texto)
tokens_train, tokens_val = molineteai.dividir_entrenamiento_validacion(tokens, 0.1)

n = molineteai.contar_parametros_config(molineteai.Config.mediana(1024))
print(f"{n:,} parámetros ({n * 4 / 1e6:.1f} MB en float32)")
```

---

## Ejemplo completo

```python
import molineteai

# Corpus y tokenizador
texto = open("cervantes.txt", encoding="utf-8").read()
tok = molineteai.TokenizadorBPE(1024)
tok.entrenar(texto, 1024)

# Modelo
cfg = molineteai.Config.pequena(tok.tam_vocabulario())
modelo = molineteai.GPT2Entrenable(cfg)
print(f"Parámetros: {modelo.num_parametros():,}")

# Entrenamiento (guarda checkpoints y métricas en dir_salida)
modelo.entrenar(tok, texto, pasos=3_000, tasa_aprendizaje=3e-4, dir_salida="checkpoints/")

# Guardar y recargar
tok.guardar("checkpoints/tokenizador.json")
modelo.guardar("checkpoints/modelo.bin")
modelo2 = molineteai.GPT2Entrenable.cargar("checkpoints/modelo.bin")

# Generación
ids = modelo2.generar(tok.codificar("Sancho Panza respondió"), max_tokens=200, temperatura=0.85)
print(tok.decodificar(ids))
```
