# Ejemplos de la librería Python

Ejemplos mínimos de uso directo de `molineteai` (los bindings PyO3 del
modelo en Rust), sin pasar por la interfaz web.

## Requisitos

Instalar y compilar el proyecto desde la raíz:

```bash
uv sync
```

## Ejemplos

| Archivo | Qué muestra |
|:---|:---|
| [`00_descargar_corpus.py`](00_descargar_corpus.py) | Descarga 5 obras de Cervantes desde Project Gutenberg y las une en `cervantes.txt` |
| [`01_presets_pequenos.py`](01_presets_pequenos.py) | Los dos presets más pequeños (`Config.diminuta` y `Config.pequena`): tokenizador BPE, entrenamiento corto y el mismo prompt generado por ambos |
| [`02_arquitectura_personalizada.py`](02_arquitectura_personalizada.py) | Un `Config` a medida (embd, capas, cabezas y contexto propios) y generación con dos temperaturas |

## Ejecutar

Desde la raíz del repositorio, en orden:

```bash
uv run python examples/00_descargar_corpus.py
uv run python examples/01_presets_pequenos.py
uv run python examples/02_arquitectura_personalizada.py
```

El primero solo hace falta una vez (deja `cervantes.txt` en la raíz).
Los entrenamientos son cortos a propósito (600 a 800 pasos, un par de
minutos): sirven para ver la API completa en acción, no para obtener un
buen modelo. Los checkpoints se guardan en `data/ejemplo_*`.

Para la referencia completa de la API, ver [docs/REFERENCIA.md](../docs/REFERENCIA.md).
