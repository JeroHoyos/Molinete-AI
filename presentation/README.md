# Presentación de Molinete AI

Diapositivas animadas hechas con [Manim Slides](https://manim-slides.eertmans.be/).

## Uso

Requiere haber instalado el proyecto con `uv sync` (ver [Inicio rápido](../README.md#inicio-rápido)). Los comandos con `uv run` usan el entorno del proyecto automáticamente, sin activar nada.

```bash
# 1. Entrar a esta carpeta
cd presentation

# 2. Compilar (los videos se guardan en media/)
uv run python -m manim_slides render main.py Presentacion

# 3. Presentar en escritorio
uv run python -m manim_slides present Presentacion
```

> Se invoca como `python -m manim_slides` (y no `manim-slides` a secas) porque
> Smart App Control de Windows puede bloquear el lanzador `.exe` del entorno
> con el error "An Application Control policy has blocked this file".
> La forma con `python -m` hace exactamente lo mismo.

## Notas

- Los métodos de diapositiva llaman a `self._siguiente()` para marcar puntos de pausa interactivos.
- `self.limpiar_pantalla()` hace fade out de todos los mobjects antes de la siguiente diapositiva.

