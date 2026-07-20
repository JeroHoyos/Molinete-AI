# Presentación de Molinete AI

Diapositivas animadas hechas con [Manim Slides](https://manim-slides.eertmans.be/).

Requiere haber instalado el proyecto con `uv sync` (ver [Inicio rápido](../README.md#inicio-rápido)). Los comandos con `uv run` usan el entorno del proyecto automáticamente, sin activar nada.

## Compilar y presentar

```bash
cd presentation

# Renderiza y deja la presentación lista en portable/
uv run python compilar.py

# Presenta (siempre desde portable/)
cd portable
uv run python -m manim_slides present Presentacion
```

## Llevar la presentación a otro PC

La carpeta `portable/` es autocontenida: la presentación renderizada más un
`Presentacion.mp4` de respaldo. Se puede copiar tal cual a un USB, o empaquetar:

```bash
uv run python comprimir.py      # crea presentacion_portable.zip
```

En el otro PC basta el zip y `descomprimir.py`:

```bash
uv run python descomprimir.py
```

Las instrucciones completas para presentar fuera del repositorio, los controles
de teclado y el caso Linux/Mac están en [portable/README.md](portable/README.md).
El script `portable/normalizar_rutas.py` ajusta las rutas del JSON para
Linux/Mac; `compilar.py` ya lo ejecuta automáticamente al final.
