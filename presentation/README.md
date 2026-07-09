# Presentación de Molinete AI

Diapositivas animadas de la arquitectura Transformer, hechas con [Manim Slides](https://manim-slides.eertmans.be/).

## Uso

Requiere haber instalado el proyecto con `uv sync` (ver [Inicio rápido](../README.md#inicio-rápido)). Los comandos con `uv run` usan el entorno del proyecto automáticamente, sin activar nada.

```bash
# 1. Entrar a esta carpeta
cd presentation

# 2. Compilar (los videos se guardan en media/)
uv run manim-slides render main.py Presentacion

# 3. Presentar en escritorio
uv run manim-slides present Presentacion
```

## Notas

- Los métodos de diapositiva llaman a `self._siguiente()` para marcar puntos de pausa interactivos.
- `self.limpiar_pantalla()` hace fade out de todos los mobjects antes de la siguiente diapositiva.
