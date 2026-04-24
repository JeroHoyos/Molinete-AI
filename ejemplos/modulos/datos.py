"""
modulos/datos.py
━━━━━━━━━━━━━━━━
Descarga y verificación de corpus de texto para Molinete AI.

El corpus central del proyecto es Cervantes (cervantes.txt).
El usuario también puede indicar la ruta de cualquier archivo .txt propio.

Exporta:
    elegir_corpus()         — Devuelve cervantes.txt si existe, o pide ruta personalizada
    verificar_corpus(ruta)  — Comprueba si el corpus existe e informa su tamaño
    es_corpus_cervantes(ruta) — True si el corpus es el archivo central de Cervantes
    run_descargar_datos()   — Opción 11: descarga cervantes.txt desde Project Gutenberg
"""

import os
import time
import urllib.request

from modulos.ui import titulo, pedir_input, barra_progreso, imprimir_lento

# ─────────────────────────────────────────────────────────────────────────────
# Corpus de Cervantes — obras descargables desde Project Gutenberg
# ─────────────────────────────────────────────────────────────────────────────

ARCHIVO_CERVANTES = "cervantes.txt"

OBRAS_CERVANTES = [
    {"titulo": "Don Quijote de la Mancha", "url": "https://www.gutenberg.org/cache/epub/2000/pg2000.txt"},
    {"titulo": "Novelas Ejemplares",       "url": "https://www.gutenberg.org/cache/epub/61202/pg61202.txt"},
    {"titulo": "La Galatea",               "url": "https://www.gutenberg.org/cache/epub/1445/pg1445.txt"},
    {"titulo": "Novelas y Teatro",         "url": "https://www.gutenberg.org/cache/epub/15115/pg15115.txt"},
    {"titulo": "Entremeses",               "url": "https://www.gutenberg.org/cache/epub/57955/pg57955.txt"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers reutilizables por otros módulos
# ─────────────────────────────────────────────────────────────────────────────

def elegir_corpus() -> str | None:
    """
    Devuelve la ruta del corpus a usar:
      - Si cervantes.txt existe → lo usa directamente, sin preguntar.
      - Si no existe → informa cómo descargarlo y ofrece indicar un .txt propio.

    Returns:
        Ruta al archivo .txt, o None si el usuario cancela.
    """
    # Caso feliz: Cervantes ya está en disco
    if os.path.exists(ARCHIVO_CERVANTES):
        tam_mb = os.path.getsize(ARCHIVO_CERVANTES) / 1e6
        print(f"\n  Corpus: {ARCHIVO_CERVANTES} ({tam_mb:.2f} MB) ✓")
        return ARCHIVO_CERVANTES

    # Cervantes no está: informar y ofrecer alternativa
    print(f"\n⚠️  '{ARCHIVO_CERVANTES}' no encontrado.")
    print("   Descárgalo desde el menú principal → opción 11")
    print()
    print("   Alternativa: indica la ruta de cualquier archivo .txt propio.")
    ruta = pedir_input("   Ruta del corpus (o Enter para cancelar): ")

    if not ruta:
        return None

    if not ruta.endswith(".txt"):
        print("⚠️  Solo se aceptan archivos .txt")
        return None

    if not os.path.exists(ruta):
        print(f"⚠️  Archivo no encontrado: '{ruta}'")
        return None

    return ruta


def verificar_corpus(ruta: str) -> bool:
    """
    Comprueba si el corpus existe en disco e informa su tamaño.

    Returns:
        True si existe, False en caso contrario.
    """
    if not os.path.exists(ruta):
        print(f"\n⚠️  Corpus '{ruta}' no encontrado.")
        print("   Descárgalo desde el menú principal → opción 11\n")
        return False
    tam_mb = os.path.getsize(ruta) / 1e6
    print(f"  Corpus: {ruta} ({tam_mb:.2f} MB)")
    return True


def es_corpus_cervantes(ruta: str) -> bool:
    """Devuelve True si el corpus es el archivo central de Cervantes."""
    return os.path.abspath(ruta) == os.path.abspath(ARCHIVO_CERVANTES)


# ─────────────────────────────────────────────────────────────────────────────
# Lógica de descarga
# ─────────────────────────────────────────────────────────────────────────────

def _descargar_obra(obra: dict) -> str | None:
    try:
        with urllib.request.urlopen(obra["url"]) as respuesta:
            return respuesta.read().decode("utf-8")
    except Exception as e:
        print(f"  ✗ Error al descargar '{obra['titulo']}': {e}")
        return None


def _descargar_cervantes(ruta_salida: str):
    print(f"\nDescargando obras de Cervantes desde Project Gutenberg...")
    print(f"Destino: {ruta_salida}\n")

    with open(ruta_salida, "w", encoding="utf-8") as archivo_final:
        for obra in OBRAS_CERVANTES:
            print(f"  Descargando: {obra['titulo']}...")
            contenido = _descargar_obra(obra)
            if contenido is None:
                continue
            archivo_final.write(f"\n\n{'='*50}\n")
            archivo_final.write(f"--- {obra['titulo'].upper()} ---\n")
            archivo_final.write(f"{'='*50}\n\n")
            archivo_final.write(contenido)
            print(f"  ✓ '{obra['titulo']}' añadido.")
            if obra is not OBRAS_CERVANTES[-1]:
                time.sleep(2)   # Pausa cortés con los servidores de Gutenberg

    tam = os.path.getsize(ruta_salida)
    print(f"\n✅ Descarga completada. {ruta_salida}  ({tam/1e6:.2f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# Punto de entrada del menú principal
# ─────────────────────────────────────────────────────────────────────────────

def run_descargar_datos():
    """Opción 11 — Descarga cervantes.txt desde Project Gutenberg."""
    titulo("11 — Descargar Corpus de Cervantes")

    obras_str = ", ".join(o["titulo"] for o in OBRAS_CERVANTES)
    print(f"Se descargarán {len(OBRAS_CERVANTES)} obras y se concatenarán en '{ARCHIVO_CERVANTES}':")
    print(f"  {obras_str}\n")

    if os.path.exists(ARCHIVO_CERVANTES):
        tam_mb = os.path.getsize(ARCHIVO_CERVANTES) / 1e6
        confirmar = pedir_input(
            f"⚠️  '{ARCHIVO_CERVANTES}' ya existe ({tam_mb:.2f} MB). ¿Sobreescribir? (s/n): ", "n"
        )
        if confirmar.lower() != "s":
            print("Operación cancelada.")
            return

    barra_progreso("Preparando descarga", segundos=0.5)
    _descargar_cervantes(ARCHIVO_CERVANTES)
    imprimir_lento("\n¡Cervantes listo! Ahora puedes entrenar modelos.", ms_por_letra=12)


# ─────────────────────────────────────────────────────────────────────────────
# Ejecución directa
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_descargar_datos()
