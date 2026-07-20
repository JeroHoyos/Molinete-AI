# Presentación de Molinete AI

Diapositivas animadas hechas con [Manim Slides](https://manim-slides.eertmans.be/).

## Uso

Requiere haber instalado el proyecto con `uv sync` (ver [Inicio rápido](../README.md#inicio-rápido)). Los comandos con `uv run` usan el entorno del proyecto automáticamente, sin activar nada.

```bash
# 1. Entrar a esta carpeta
cd presentation

# 2. Compilar (renderiza y deja todo listo en portable/)
uv run python compilar.py

# 3. Presentar en escritorio (desde portable/)
cd portable
uv run python -m manim_slides present Presentacion
```

La carpeta `portable/` es autocontenida: contiene la presentacion renderizada
(`slides/` + `Presentacion.mp4` como respaldo) y se puede copiar tal cual a un
USB para presentar en otro PC (ver `portable/LEEME.md`). La cache de manim se
queda en `media/`, asi que no hace falta llevarla.

> Se invoca como `python -m manim_slides` (y no `manim-slides` a secas) porque
> Smart App Control de Windows puede bloquear el lanzador `.exe` del entorno
> con el error "An Application Control policy has blocked this file".
> La forma con `python -m` hace exactamente lo mismo.

## Notas

- Los métodos de diapositiva llaman a `self._siguiente()` para marcar puntos de pausa interactivos.
- `self.limpiar_pantalla()` hace fade out de todos los mobjects antes de la siguiente diapositiva.

