"""
molineteai.py
━━━━━━━━━━━━━
Lanzador unificado de Molinete AI.

Muestra el arte ASCII animado, luego presenta un menú interactivo
para ejecutar cualquier ejemplo o herramienta del proyecto.

Estructura del proyecto:
    molineteai.py          ← Este archivo (lanzador principal)
    modulos/
        ui.py              ← Arte ASCII, animaciones y helpers de consola
        datos.py           ← Descarga y verificación de corpus
        tokenizadores.py   ← Ejemplo 01: Tokenizadores BPE
        tensores.py        ← Ejemplo 02: Operaciones tensoriales
        arquitectura.py    ← Ejemplo 03: Arquitectura GPT-2
        infraestructura.py ← Ejemplo 04: Infraestructura de entrenamiento
        entrenamiento.py   ← Ejemplos 05-09: Modelos de entrenamiento
        chat.py            ← Ejemplo 10: Chat con modelo entrenado

Uso:
    python molineteai.py
"""

import sys
import os

# ─────────────────────────────────────────────────────────────────────────────
# Compatibilidad Windows: activar colores ANSI en la terminal
# ─────────────────────────────────────────────────────────────────────────────
if sys.platform == "win32":
    os.system("")   # habilita secuencias ANSI en Windows 10+

# ─────────────────────────────────────────────────────────────────────────────
# Importar módulos del proyecto
# ─────────────────────────────────────────────────────────────────────────────
from modulos.ui import (
    mostrar_arte, barra_progreso, imprimir_lento, pedir_input
)
from modulos.tokenizadores  import run_01_tokenizadores
from modulos.tensores        import run_02_tensores
from modulos.arquitectura    import run_03_arquitectura
from modulos.infraestructura import run_04_infraestructura
from modulos.entrenamiento   import (
    run_05_diminuto, run_06_pequeno, run_07_mediano,
    run_08_gpt2, run_entrenar_presets
)
from modulos.chat  import run_chat
from modulos.datos import run_descargar_datos

# ─────────────────────────────────────────────────────────────────────────────
# Definición del menú
# ─────────────────────────────────────────────────────────────────────────────

CATEGORIAS = [
    {
        "titulo": "📚  APRENDIZAJE — Conceptos fundamentales",
        "items": [
            {
                "clave": "1",
                "nombre": "Tokenizadores BPE",
                "desc":   "Entrena tokenizadores con varios tamaños de vocabulario y analiza la compresión",
                "funcion": run_01_tokenizadores,
            },
            {
                "clave": "2",
                "nombre": "Operaciones tensoriales",
                "desc":   "Demuestra matmul, softmax, máscaras causales y forward pass real",
                "funcion": run_02_tensores,
            },
            {
                "clave": "3",
                "nombre": "Arquitectura GPT-2",
                "desc":   "Crea modelos de distintos tamaños, cuenta parámetros y hace benchmarks",
                "funcion": run_03_arquitectura,
            },
            {
                "clave": "4",
                "nombre": "Infraestructura de entrenamiento",
                "desc":   "Data loaders, división train/val y logging CSV (sin entrenar)",
                "funcion": run_04_infraestructura,
            },
        ],
    },
    {
        "titulo": "🏋️  ENTRENAMIENTO — Modelos en Shakespeare",
        "items": [
            {
                "clave": "5",
                "nombre": "GPT-2 Diminuto  (~170K params)",
                "desc":   "Entrena en minutos. Ideal para ver el ciclo completo rápido",
                "funcion": run_05_diminuto,
            },
            {
                "clave": "6",
                "nombre": "GPT-2 Pequeño   (~200K params)",
                "desc":   "~15 minutos. Mejor coherencia, buena opción para experimentar",
                "funcion": run_06_pequeno,
            },
            {
                "clave": "7",
                "nombre": "GPT-2 Mediano   (~4M params)",
                "desc":   "~1-3 horas. Texto más natural y fluido",
                "funcion": run_07_mediano,
            },
            {
                "clave": "8",
                "nombre": "GPT-2 Small completo (~163M params)",
                "desc":   "Toda la noche. Arquitectura original de OpenAI",
                "funcion": run_08_gpt2,
            },
        ],
    },
    {
        "titulo": "⚙️   HERRAMIENTAS — Configuración avanzada",
        "items": [
            {
                "clave": "9",
                "nombre": "Entrenador con presets",
                "desc":   "pocket-bard, spider, cyclops, wide, narrow... con argumentos CLI",
                "funcion": run_entrenar_presets,
            },
            {
                "clave": "10",
                "nombre": "Chat con modelo entrenado",
                "desc":   "Carga un checkpoint .bin y chatea con él en tiempo real",
                "funcion": run_chat,
            },
        ],
    },
    {
        "titulo": "🌐  DATOS — Descarga y gestión de corpus",
        "items": [
            {
                "clave": "11",
                "nombre": "Descargar corpus de Cervantes",
                "desc":   "Descarga obras completas de Cervantes desde Project Gutenberg",
                "funcion": run_descargar_datos,
            },
        ],
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Mapa de funciones por clave (construido automáticamente desde CATEGORIAS)
# ─────────────────────────────────────────────────────────────────────────────

FUNCIONES = {}
for cat in CATEGORIAS:
    for item in cat["items"]:
        FUNCIONES[item["clave"]] = item["funcion"]


# ─────────────────────────────────────────────────────────────────────────────
# Menú principal
# ─────────────────────────────────────────────────────────────────────────────

def mostrar_menu():
    print(f"\n{'═'*70}")
    print("  MOLINETE AI — Menú Principal")
    print(f"{'═'*70}")
    for cat in CATEGORIAS:
        print(f"\n  {cat['titulo']}")
        print(f"  {'─'*66}")
        for item in cat["items"]:
            print(f"  [{item['clave']:>2}]  {item['nombre']}")
            print(f"        {item['desc']}")
    print(f"\n  [ 0]  Salir")
    print(f"{'═'*70}")


def main():
    mostrar_arte()
    barra_progreso("Iniciando Molinete AI", segundos=1.2)
    imprimir_lento("¡Bienvenido a Molinete AI! Un transformer GPT-2 desde cero en Rust.", ms_por_letra=12)

    while True:
        mostrar_menu()
        opcion = pedir_input("\n  Elige una opción: ").strip()

        if opcion == "0":
            imprimir_lento("\n¡Hasta pronto!", ms_por_letra=25)
            break
        elif opcion in FUNCIONES:
            try:
                FUNCIONES[opcion]()
            except KeyboardInterrupt:
                print("\n\n  [Interrumpido] Volviendo al menú...\n")
            except Exception as e:
                print(f"\n⚠️  Error inesperado: {e}")
                import traceback
                traceback.print_exc()
            input("\n  Presiona Enter para volver al menú...")
        else:
            print(f"\n  ⚠️  Opción '{opcion}' no válida. Elige un número del menú.")


if __name__ == "__main__":
    main()
