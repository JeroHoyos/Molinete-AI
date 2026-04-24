# Presentación — Molinete AI

Instrucciones para preparar el entorno, instalar dependencias, compilar las diapositivas y ejecutar la presentación animada de la arquitectura Transformer.

---

## Tabla de contenidos

- [1. Instalar dependencias](#1-instalar-dependencias)
- [2. Activar el entorno virtual](#2-activar-el-entorno-virtual)
- [3. Compilar las diapositivas](#3-compilar-las-diapositivas)
- [4. Presentar las diapositivas](#4-presentar-las-diapositivas)
- [5. Controles durante la presentación](#5-controles-durante-la-presentación)
- [6. Flujo típico de trabajo](#6-flujo-típico-de-trabajo)
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

Una vez renderizadas, inicia la presentación:

```bash
py -m manim_slides present Presentacion
```

<div align="right"><a href="#presentación--molinete-ai">↑ Volver arriba</a></div>

---

## 5. Controles durante la presentación

| Tecla | Acción |
|:---|:---|
| `→` | Siguiente diapositiva |
| `←` | Diapositiva anterior |
| `f` | Pantalla completa |
| `q` | Salir |

<div align="right"><a href="#presentación--molinete-ai">↑ Volver arriba</a></div>

---

## 6. Flujo típico de trabajo

```bash
uv venv .env
uv pip install -r requirements.txt
.env\Scripts\activate                        # Windows
# source .env/bin/activate                   # Linux / Mac
py -m manim_slides render main.py Presentacion
py -m manim_slides present Presentacion
```

<div align="right"><a href="#presentación--molinete-ai">↑ Volver arriba</a></div>

---

## Notas

- Cada diapositiva se define usando `self.next_slide()`.
- Cada clase de presentación debe heredar de `Slide`.
- Los videos renderizados se guardan en la carpeta `media/`.

### Compatibilidad de rutas entre Windows y Linux

Si al presentar aparecen errores de rutas, ejecuta:

```bash
sed -i 's/\\\\/\//g' slides/Presentacion.json
```
