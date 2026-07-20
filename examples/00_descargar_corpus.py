"""
Ejemplo 0 — Descargar el corpus de Cervantes.

Baja 5 obras de Cervantes desde Project Gutenberg y las une en un
solo archivo de texto: cervantes.txt (~7 MB). Los demás ejemplos
(y el entrenamiento desde la web) usan ese archivo.

Cómo ejecutarlo (desde la raíz del repositorio):
    uv run python examples/00_descargar_corpus.py
"""

import os
import time
import urllib.request

OBRAS = [
    ("Don Quijote de la Mancha", "https://www.gutenberg.org/cache/epub/2000/pg2000.txt"),
    ("Novelas Ejemplares",       "https://www.gutenberg.org/cache/epub/61202/pg61202.txt"),
    ("La Galatea",               "https://www.gutenberg.org/cache/epub/1445/pg1445.txt"),
    ("Novelas y Teatro",         "https://www.gutenberg.org/cache/epub/15115/pg15115.txt"),
    ("Entremeses",               "https://www.gutenberg.org/cache/epub/57955/pg57955.txt"),
]

with open("cervantes.txt", "w", encoding="utf-8") as archivo:
    for titulo, url in OBRAS:
        print(f"Descargando {titulo}...")
        with urllib.request.urlopen(url) as respuesta:
            texto = respuesta.read().decode("utf-8")
        archivo.write(f"\n\n--- {titulo.upper()} ---\n\n")
        archivo.write(texto)
        time.sleep(2)   # pausa cortés con los servidores de Gutenberg

mb = os.path.getsize("cervantes.txt") / 1e6
print(f"\nListo: cervantes.txt ({mb:.1f} MB). Ya puedes entrenar modelos.")
