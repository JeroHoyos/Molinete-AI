# Molinete AI — Estructura del Proyecto

```
molinete_ai/
│
├── molineteai.py              ← Lanzador principal  (python molineteai.py)
│
└── modulos/
    ├── __init__.py
    │
    ├── ui.py                  ← Arte ASCII, animaciones, helpers de consola
    │                             Exporta: mostrar_arte, barra_progreso,
    │                                      imprimir_lento, pedir_input, titulo
    │
    ├── datos.py               ← Descarga y verificación de corpus
    │                             Exporta: run_descargar_datos, verificar_corpus
    │                             Ejecución directa: python -m modulos.datos
    │
    ├── tokenizadores.py       ← Opción 01: Tokenizadores BPE
    ├── tensores.py            ← Opción 02: Operaciones tensoriales
    ├── arquitectura.py        ← Opción 03: Arquitectura GPT-2
    ├── infraestructura.py     ← Opción 04: Infraestructura de entrenamiento
    ├── entrenamiento.py       ← Opciones 05-09: Modelos y presets
    └── chat.py                ← Opción 10: Chat con modelo entrenado
```

## Cómo usar

```bash
# Lanzar el menú principal
python molineteai.py

# Ejecutar un módulo directamente (sin pasar por el menú)
python -m modulos.datos
```

## Menú de opciones

| Opción | Módulo              | Descripción                          |
|--------|---------------------|--------------------------------------|
| 1      | tokenizadores.py    | Tokenizadores BPE                    |
| 2      | tensores.py         | Operaciones tensoriales              |
| 3      | arquitectura.py     | Arquitectura GPT-2                   |
| 4      | infraestructura.py  | Infraestructura de entrenamiento     |
| 5      | entrenamiento.py    | GPT-2 Diminuto (~170K params)        |
| 6      | entrenamiento.py    | GPT-2 Pequeño (~200K params)         |
| 7      | entrenamiento.py    | GPT-2 Mediano (~4M params)           |
| 8      | entrenamiento.py    | GPT-2 Small completo (~163M params)  |
| 9      | entrenamiento.py    | Entrenador con presets               |
| 10     | chat.py             | Chat con modelo entrenado            |
| 11     | datos.py            | Descargar corpus (Cervantes/Shakespeare) |

## Requisitos

- Python 3.10+
- Módulo `molineteai` compilado con `maturin develop --release`
- `numpy` para la opción 02

## Flujo recomendado para Cervantes

1. Opción **11** → Descargar corpus de Cervantes (`cervantes.txt`)
2. Opción **9** → Entrenador con presets → elegir `pocket-bard`
   → cambiar archivo de datos a `cervantes.txt`
3. Opción **10** → Chatear con el modelo entrenado
