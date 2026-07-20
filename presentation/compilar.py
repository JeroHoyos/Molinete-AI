"""Compila la presentacion y deja todo listo en portable/.

Uso (desde la carpeta presentation):

    uv run python compilar.py

Renderiza con manim-slides y luego mueve la salida (slides/Presentacion.json,
slides/files/ y el video completo) a portable/, que es la carpeta autocontenida
para llevar a otro PC. La cache de manim se queda en media/ para que las
siguientes compilaciones sean rapidas.
"""

import shutil
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
PORTABLE = AQUI / "portable"
SLIDES = AQUI / "slides"
VIDEO = AQUI / "media" / "videos" / "main" / "1080p60" / "Presentacion.mp4"


def main() -> None:
    subprocess.run(
        [sys.executable, "-m", "manim_slides", "render", "main.py", "Presentacion"],
        cwd=AQUI,
        check=True,
    )

    destino = PORTABLE / "slides"
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)

    shutil.move(str(SLIDES / "Presentacion.json"), str(destino / "Presentacion.json"))
    shutil.move(str(SLIDES / "files"), str(destino / "files"))

    if VIDEO.exists():
        destino_video = PORTABLE / "Presentacion.mp4"
        destino_video.unlink(missing_ok=True)
        shutil.move(str(VIDEO), str(destino_video))

    print(f"Listo: presentacion portable en {PORTABLE}")


if __name__ == "__main__":
    main()
