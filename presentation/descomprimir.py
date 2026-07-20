#!/usr/bin/env python3
"""Descomprime presentacion_portable.zip y deja la carpeta portable/ lista.

Uso, desde la carpeta donde esten este script y el zip (en el otro PC
basta tener Python, no hay que instalar nada):

    python3 descomprimir.py

En Windows: py descomprimir.py

Si ya existe una carpeta portable/slides de una compilacion anterior, la
reemplaza para no mezclar videos de versiones distintas. El zip no se
borra; se puede eliminar a mano despues de comprobar que todo funciona.
"""

import shutil
import zipfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ZIP = AQUI / "presentacion_portable.zip"


def main() -> None:
    if not ZIP.exists():
        print(f"No se encuentra {ZIP.name} junto a este script.")
        return

    slides_previo = AQUI / "portable" / "slides"
    if slides_previo.exists():
        shutil.rmtree(slides_previo)

    with zipfile.ZipFile(ZIP) as zf:
        zf.extractall(AQUI)

    print(f"Listo: presentacion descomprimida en {AQUI / 'portable'}")
    print("Para presentar, entra a portable/ y sigue su README.md.")


if __name__ == "__main__":
    main()
