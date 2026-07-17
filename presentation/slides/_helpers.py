import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import numpy as np
import os
from colores import *
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

    _TOKENS = {
        "Rey":    _TokenVector("Rey",    38.0,   _MAGNITUD_BASE, NARANJA_TERRACOTA, UR),
        "Reina":  _TokenVector("Reina",  142.0,  _MAGNITUD_BASE, VERDE_OLIVA,       UL),
        "Hombre": _TokenVector("Hombre", -38.0,  _MAGNITUD_BASE, NARANJA_TERRACOTA, DR),
        "Mujer":  _TokenVector("Mujer",  -142.0, _MAGNITUD_BASE, VERDE_OLIVA,       DL),
    }

    # ── estilo "espacio vectorial" (inspirado en slide_mha_acto2_qkv) ───
    @staticmethod
    def _emb_grid(centro, unidad=0.78, n=4):
        centro = np.asarray(centro, dtype=float)
        lineas = VGroup()
        for i in range(-n, n + 1):
            lineas.add(Line(centro + np.array([i * unidad, -n * unidad, 0]),
                            centro + np.array([i * unidad, n * unidad, 0]),
                            stroke_color=PAPEL_TAN, stroke_width=1.0, stroke_opacity=0.4))
            lineas.add(Line(centro + np.array([-n * unidad, i * unidad, 0]),
                            centro + np.array([n * unidad, i * unidad, 0]),
                            stroke_color=PAPEL_TAN, stroke_width=1.0, stroke_opacity=0.4))
        ax = DoubleArrow(centro + np.array([-(n * unidad + 0.3), 0, 0]),
                         centro + np.array([n * unidad + 0.3, 0, 0]), buff=0,
                         color=MARRON_OSCURO, stroke_width=3, tip_length=0.2)
        ay = DoubleArrow(centro + np.array([0, -(n * unidad + 0.3), 0]),
                         centro + np.array([0, n * unidad + 0.3, 0]), buff=0,
                         color=MARRON_OSCURO, stroke_width=3, tip_length=0.2)
        return VGroup(lineas, ax, ay)

    @staticmethod
    def _emb_punto(pos, color, nombre, dir_lbl=UR, fs=20):
        pos = np.asarray(pos, dtype=float)
        halo = Dot(pos, radius=0.19, color=color).set_opacity(0.22)
        dot = Dot(pos, radius=0.11, color=color).set_stroke(MARRON_OSCURO, width=1.5)
        lbl = Text(nombre, font=FUENTE, font_size=fs, color=color, weight=BOLD)
        lbl.set_background_stroke(color=PAPEL_CREMA, width=5)
        lbl.next_to(pos, dir_lbl, buff=0.14)
        return VGroup(halo, dot, lbl)

    @staticmethod
    def _emb_vec(a, b, color, width=4.5, opacity=1.0):
        return Arrow(a, b, buff=0.12, color=color, stroke_width=width,
                     max_tip_length_to_length_ratio=0.14, stroke_opacity=opacity)


