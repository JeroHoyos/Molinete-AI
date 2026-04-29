# Presentación — Molinete AI

Instrucciones para preparar el entorno, instalar dependencias, compilar las diapositivas y ejecutar la presentación animada de la arquitectura Transformer.

---

## Tabla de contenidos

- [1. Instalar dependencias](#1-instalar-dependencias)
- [2. Activar el entorno virtual](#2-activar-el-entorno-virtual)
- [3. Compilar las diapositivas](#3-compilar-las-diapositivas)
- [4. Presentar las diapositivas](#4-presentar-las-diapositivas)
- [5. Ver en el navegador](#5-ver-en-el-navegador)
- [6. Controles durante la presentación](#6-controles-durante-la-presentación)
- [7. Flujo típico de trabajo](#7-flujo-típico-de-trabajo)
- [Notas](#notas)

---

## 1. Instalar dependencias

Se recomienda usar [`uv`](https://docs.astral.sh/uv/) para gestionar el entorno virtual e instalar dependencias.

### Instalar uv (si no lo tienes)

```bash
pip install uv
```

### Crear entorno e instalar dependencias

```bash
uv venv .env
uv pip install -r requirements.txt
```

Esto crea el entorno en `.env/` e instala todo en un solo paso.

<div align="right"><a href="#presentación--molinete-ai">↑ Volver arriba</a></div>

---

## 2. Activar el entorno virtual

**Windows**
```bash
.env\Scripts\activate
```

**Linux / Mac**
```bash
source .env/bin/activate
```

<div align="right"><a href="#presentación--molinete-ai">↑ Volver arriba</a></div>

---

## 3. Compilar las diapositivas

Para generar el video de las diapositivas:

```bash
py -m manim_slides render main.py Presentacion
```

Los videos renderizados se guardan en la carpeta `media/`.

<div align="right"><a href="#presentación--molinete-ai">↑ Volver arriba</a></div>

---

## 4. Presentar las diapositivas

Una vez renderizadas, inicia la presentación en escritorio:

```bash
py -m manim_slides present Presentacion
```

<div align="right"><a href="#presentación--molinete-ai">↑ Volver arriba</a></div>

---

## 5. Ver en el navegador

Puedes exportar las diapositivas a un archivo HTML autocontenido y abrirlo en cualquier navegador, sin necesidad de PyQt ni de tener Manim instalado en el equipo receptor.

### Exportar a HTML

```bash
py -m manim_slides convert Presentacion presentacion.html
```

Esto genera `presentacion.html` en el directorio actual. El proceso puede tardar unos segundos mientras empaqueta los vídeos dentro del archivo.

### Servir localmente y abrir en el navegador

```bash
py -m http.server 8080
```

Luego abre **http://localhost:8080/presentacion.html** en tu navegador.

### Controles en el navegador

| Tecla / acción | Función |
|:---|:---|
| `→` / clic | Siguiente diapositiva |
| `←` | Diapositiva anterior |
| `f` | Pantalla completa |
| `Espacio` | Siguiente diapositiva |

> **Nota:** Algunos navegadores bloquean la reproducción automática de vídeo.
> Si las animaciones no avanzan automáticamente, habilita la reproducción automática
> para `localhost` en la configuración del navegador.

<div align="right"><a href="#presentación--molinete-ai">↑ Volver arriba</a></div>

---

## 6. Controles durante la presentación

| Tecla | Acción |
|:---|:---|
| `→` | Siguiente diapositiva |
| `←` | Diapositiva anterior |
| `f` | Pantalla completa |
| `q` | Salir |

<div align="right"><a href="#presentación--molinete-ai">↑ Volver arriba</a></div>

---

## 7. Flujo típico de trabajo

```bash
# 1. Crear entorno e instalar
uv venv .env
uv pip install -r requirements.txt

# 2. Activar entorno
.env\Scripts\activate                        # Windows
# source .env/bin/activate                   # Linux / Mac

# 3. Compilar
py -m manim_slides render main.py Presentacion

# 4a. Presentar en escritorio
py -m manim_slides present Presentacion

# 4b. O exportar y ver en el navegador
py -m manim_slides convert Presentacion presentacion.html
py -m http.server 8080
# Abrir http://localhost:8080/presentacion.html
```

<div align="right"><a href="#presentación--molinete-ai">↑ Volver arriba</a></div>

---

## Estructura del proyecto

```
main.py                     — Clase Presentacion; hereda de todos los mixins de slides
colores.py                  — Constantes de color y tamaño de fuente (tema Cervantes)
objetos.py                  — Fábricas de Mobjects reutilizables (molino, sol, pergamino…)
snippets.py                 — Dict RUST_SNIPPETS con los fragmentos de código de las diapositivas
slides/
  slides_00_inicio.py       — Pantalla "pronto iniciamos"
  slides_01_intro.py        — Introducción, créditos, ¿qué es un Transformer?, ¿por qué Rust?, roadmap
  slides_02_tensores.py     — Tensores, strides, matmul, SIMD, cache blocking, softmax
  slides_03_tokenizacion.py — Tokenización, BPE, el problema de la fresa
  slides_04_embeddings.py   — Embeddings y embeddings posicionales
  slides_05_atencion.py     — Atención multi-cabeza (intuición → QKV → fórmula → multi-head)
  slides_06_mlp.py          — MLP, activación GELU, LayerNorm, residual, bloque Transformer
  slides_07_entrenamiento.py— Entrenamiento, descenso de gradiente, backprop, AdamW, dropout, temperatura
  slides_08_final.py        — Bridge Rust-Python, modelo en acción, diapositiva final
assets/                     — Imágenes PNG usadas por ImageMobject (logos, arte)
media/                      — Videos generados (creados al renderizar, no en el repo)
```

## Notas

- Los métodos de diapositiva llaman a `self._siguiente()` para marcar puntos de pausa interactivos.
- `self.limpiar_pantalla()` hace fade out de todos los mobjects antes de la siguiente diapositiva.
- Para habilitar una diapositiva, descomenta su llamada en `Presentacion.construct()` en `main.py`.
- Los videos renderizados se guardan en la carpeta `media/`.

### Compatibilidad de rutas entre Windows y Linux

Si al presentar aparecen errores de rutas, ejecuta:

```bash
sed -i 's/\\\\/\//g' slides/Presentacion.json
```
