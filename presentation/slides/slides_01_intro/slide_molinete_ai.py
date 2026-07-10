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


class SlideMolineteAi:
    def slide_molinete_ai(self):
        titulo, linea = self.crear_titulo(
            "¿Por qué Molinete?",
            palabra_clave="Molinete",
            color_clave=NARANJA_TERRACOTA,
        )
        llanuras_fondo = crear_llanuras_manchegas()
        adornos = self._crear_adornos_esquinas(escala=0.6)
        self._animar_entrada_slide(titulo, linea, adornos=adornos, fondo=llanuras_fondo)

        num_1 = Text("1.", font=FUENTE, font_size=38,
                     color=NARANJA_TERRACOTA, weight=BOLD)
        tit_1 = Text("El Quijote", font=FUENTE, font_size=28,
                     color=MARRON_OSCURO, weight=BOLD)
        enc_1 = VGroup(num_1, tit_1).arrange(RIGHT, buff=0.22, aligned_edge=DOWN)

        imagen_quijote = ImageMobject(
            os.path.join("assets", "quijote_vs_molinos.png")
        ).set_height(3.1)
        marco_1 = SurroundingRectangle(
            imagen_quijote, buff=0.06,
            color=NARANJA_TERRACOTA, stroke_width=2.0, corner_radius=0.05,
        )
        txt_1 = Text(
            "Arremetía contra molinos\ncreyéndolos gigantes.",
            font=FUENTE, font_size=18, color=TINTA_NEGRA, line_spacing=1.3,
        )
        grupo_img_1 = Group(imagen_quijote, marco_1).move_to(LEFT * 3.3 + UP * 0.35)
        enc_1.next_to(grupo_img_1, UP, buff=0.25)
        txt_1.next_to(grupo_img_1, DOWN, buff=0.25)

        num_2 = Text("2.", font=FUENTE, font_size=38,
                     color=HIERRO, weight=BOLD)
        tit_2 = Text("Dark Souls", font=FUENTE, font_size=28,
                     color=HIERRO, weight=BOLD)
        enc_2 = VGroup(num_2, tit_2).arrange(RIGHT, buff=0.22, aligned_edge=DOWN)

        imagen_souls = ImageMobject(
            os.path.join("assets", "molinete_dark_souls.png")
        ).set_height(3.1)
        marco_2 = SurroundingRectangle(
            imagen_souls, buff=0.06,
            color=HIERRO, stroke_width=2.0, corner_radius=0.05,
        )
        txt_2 = Text(
            "El boss de Las Catacumbas, «Molinete»:\nel más fácil del juego.",
            font=FUENTE, font_size=18, color=TINTA_NEGRA, line_spacing=1.3,
        )
        grupo_img_2 = Group(imagen_souls, marco_2).move_to(RIGHT * 3.3 + UP * 0.35)
        enc_2.next_to(grupo_img_2, UP, buff=0.25)
        txt_2.next_to(grupo_img_2, DOWN, buff=0.25)
        enc_2.shift(UP * (num_1.get_bottom()[1] - num_2.get_bottom()[1]))

        conclusion = Text(
            "Por eso: Molinete.",
            font=FUENTE, font_size=24,
            color=NARANJA_TERRACOTA, weight=BOLD,
        )
        caja_conclusion = SurroundingRectangle(
            conclusion,
            color=NARANJA_TERRACOTA, fill_color=FONDO_CAJA, fill_opacity=0.97,
            corner_radius=0.15, buff=0.2, stroke_width=2.5,
        )
        grupo_conclusion = VGroup(caja_conclusion, conclusion).move_to(DOWN * 2.95)

        self.play(
            FadeIn(enc_1, shift=RIGHT * 0.2),
            FadeIn(imagen_quijote, shift=UP * 0.15), FadeIn(marco_1),
            FadeIn(txt_1, shift=UP * 0.1),
            run_time=0.9,
        )
        self._siguiente()

        self.play(
            FadeIn(enc_2, shift=LEFT * 0.2),
            FadeIn(imagen_souls, shift=UP * 0.15), FadeIn(marco_2),
            FadeIn(txt_2, shift=UP * 0.1),
            run_time=0.9,
        )
        self._siguiente()

        self.play(FadeIn(grupo_conclusion, shift=UP * 0.2), run_time=0.7)
        self._siguiente()
        self.limpiar_pantalla()
