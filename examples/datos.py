"""
modulos/datos.py
━━━━━━━━━━━━━━━━
Descarga y verificación de corpus de texto para Molinete AI.

Integra la funcionalidad de download_data.py como una opción
interactiva del menú principal, permitiendo elegir qué corpus
descargar y dónde guardarlo.

Exporta:
    run_descargar_datos()   — Opción 11 del menú
    verificar_corpus(ruta)  — Helper reutilizable por otros módulos
"""

import os
import time
import urllib.request

from modulos.ui import titulo, pedir_input, barra_progreso, imprimir_lento

# ─────────────────────────────────────────────────────────────────────────────
# Catálogo de corpus disponibles
# ─────────────────────────────────────────────────────────────────────────────

CORPUS_DISPONIBLES = {
    "cervantes": {
        "nombre": "Obras de Cervantes (español)",
        "archivo": "cervantes.txt",
        "obras": [
            {"titulo": "Don Quijote de la Mancha",  "url": "https://www.gutenberg.org/cache/epub/2000/pg2000.txt"},
            {"titulo": "Novelas Ejemplares",         "url": "https://www.gutenberg.org/cache/epub/61202/pg61202.txt"},
            {"titulo": "La Galatea",                 "url": "https://www.gutenberg.org/cache/epub/1445/pg1445.txt"},
            {"titulo": "Novelas y Teatro",           "url": "https://www.gutenberg.org/cache/epub/15115/pg15115.txt"},
            {"titulo": "Entremeses",                 "url": "https://www.gutenberg.org/cache/epub/57955/pg57955.txt"},
        ],
    },
    "shakespeare": {
        "nombre": "Shakespeare completo (inglés)",
        "archivo": "shakespeare.txt",
        "obras": [
            {"titulo": "The Complete Works of Shakespeare", "url": "https://www.gutenberg.org/files/100/100-0.txt"},
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper reutilizable por otros módulos
# ─────────────────────────────────────────────────────────────────────────────

def verificar_corpus(ruta: str = "shakespeare.txt") -> bool:
    """
    Comprueba si el corpus existe en disco.
    Si no existe, informa al usuario cómo descargarlo.
    """
    if not os.path.exists(ruta):
        print(f"\n⚠️  Corpus '{ruta}' no encontrado.")
        print("   Descárgalo desde el menú principal → opción 11")
        print("   o ejecuta:  python -m modulos.datos\n")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Lógica de descarga
# ─────────────────────────────────────────────────────────────────────────────

def _descargar_obra(obra: dict) -> str | None:
    """
    Descarga una sola obra y devuelve su contenido como string.
    Retorna None si hay un error.
    """
    try:
        with urllib.request.urlopen(obra["url"]) as respuesta:
            return respuesta.read().decode("utf-8")
    except Exception as e:
        print(f"  ✗ Error al descargar '{obra['titulo']}': {e}")
        return None


def _descargar_corpus(clave: str, ruta_salida: str):
    """
    Descarga todas las obras del corpus indicado y las
    concatena en un único archivo de salida.
    """
    corpus = CORPUS_DISPONIBLES[clave]
    obras  = corpus["obras"]

    print(f"\nIniciando descarga: {corpus['nombre']}")
    print(f"Destino: {ruta_salida}\n")

    with open(ruta_salida, "w", encoding="utf-8") as archivo_final:
        for obra in obras:
            print(f"  Descargando: {obra['titulo']}...")
            contenido = _descargar_obra(obra)

            if contenido is None:
                continue

            # Cabecera separadora entre obras
            archivo_final.write(f"\n\n{'='*50}\n")
            archivo_final.write(f"--- {obra['titulo'].upper()} ---\n")
            archivo_final.write(f"{'='*50}\n\n")
            archivo_final.write(contenido)

            print(f"  ✓ '{obra['titulo']}' añadido.")

            # Pausa cortés para no saturar los servidores de Gutenberg
            if obra is not obras[-1]:
                time.sleep(2)

    # Estadísticas finales
    tam = os.path.getsize(ruta_salida)
    print(f"\n✅ Descarga completada.")
    print(f"   Archivo: {ruta_salida}  ({tam/1e6:.2f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# Punto de entrada del menú
# ─────────────────────────────────────────────────────────────────────────────

def run_descargar_datos():
    """Opción 11 — Descarga interactiva de corpus desde Project Gutenberg."""
    titulo("11 — Descargar Corpus de Texto")

    # Mostrar opciones disponibles
    print("Corpus disponibles:\n")
    claves = list(CORPUS_DISPONIBLES.keys())
    for i, clave in enumerate(claves, start=1):
        corpus = CORPUS_DISPONIBLES[clave]
        n_obras = len(corpus["obras"])
        print(f"  [{i}] {corpus['nombre']}")
        print(f"       {n_obras} obra(s)  →  {corpus['archivo']}")
    print()

    seleccion = pedir_input(f"Elige un corpus (1-{len(claves)}, o Enter para cancelar): ")
    if not seleccion:
        print("Cancelado.")
        return

    try:
        idx = int(seleccion) - 1
        if not (0 <= idx < len(claves)):
            raise ValueError
        clave = claves[idx]
    except ValueError:
        print(f"\n⚠️  Opción inválida: '{seleccion}'")
        return

    corpus_info = CORPUS_DISPONIBLES[clave]
    ruta_defecto = corpus_info["archivo"]

    # Preguntar ruta de salida
    ruta = pedir_input(
        f"Ruta de salida (Enter para '{ruta_defecto}'): ",
        default=ruta_defecto,
    )

    # Si ya existe, pedir confirmación
    if os.path.exists(ruta):
        confirmar = pedir_input(f"\n⚠️  '{ruta}' ya existe. ¿Sobreescribir? (s/n): ", "n")
        if confirmar.lower() != "s":
            print("Operación cancelada.")
            return

    barra_progreso("Preparando descarga", segundos=0.5)
    _descargar_corpus(clave, ruta)

    imprimir_lento(f"\n¡Corpus listo! Ahora puedes usarlo para entrenar modelos.", ms_por_letra=12)


# ─────────────────────────────────────────────────────────────────────────────
# Ejecución directa (python -m modulos.datos)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_descargar_datos()
