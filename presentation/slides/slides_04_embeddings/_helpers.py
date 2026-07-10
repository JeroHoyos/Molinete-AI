import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from manim import *
from manim_slides import Slide
from manim_code_blocks import *
import numpy as np
import random
import math
import os
from colores import *
from snippets import RUST_SNIPPETS
from objetos import *


class HelpersEmbeddings:
    _MAGNITUD_BASE = 2.3
    _VALORES_EMBEDDING = ["0.9", "-0.1", "0.4", "0.6", "...", "0.7"]
    _FILA_ACTIVA_W     = 3

    class _TokenVector:
        def __init__(self, nombre, angulo_deg, magnitud, color, dir_etiqueta):
            self.nombre       = nombre
            self.angulo_deg   = angulo_deg
            self.magnitud     = magnitud
            self.color        = color
            self.dir_etiqueta = dir_etiqueta

        @property
        def angulo_rad(self):
            return math.radians(self.angulo_deg)

        @property
        def coordenada(self):
            return np.array([
                self.magnitud * math.cos(self.angulo_rad),
                self.magnitud * math.sin(self.angulo_rad),
                0.0,
            ])

    _TOKENS = {
        "Rey":    _TokenVector("Rey",    38.0,   _MAGNITUD_BASE, NARANJA_TERRACOTA, UR),
        "Reina":  _TokenVector("Reina",  142.0,  _MAGNITUD_BASE, VERDE_OLIVA,       UL),
        "Hombre": _TokenVector("Hombre", -38.0,  _MAGNITUD_BASE, NARANJA_TERRACOTA, DR),
        "Mujer":  _TokenVector("Mujer",  -142.0, _MAGNITUD_BASE, VERDE_OLIVA,       DL),
    }


