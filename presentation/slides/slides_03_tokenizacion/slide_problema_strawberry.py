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


class SlideProblemaStrawberry:
    def slide_problema_strawberry(self):

        titulo, linea = self.crear_titulo(
            "¿Por qué los LLM no saben 'leer'?",
            palabra_clave="'leer'?",
            color_clave=NARANJA_TERRACOTA
        )
        llanuras_fondo = crear_llanuras_manchegas()
        sol_fondo = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN * 0.2 + LEFT * 0.2)

        burbuja_pregunta = self._crear_burbuja_chat(
            "¿Cuántas letras 'r' hay en 'strawberry'?",
            color_fondo=MARRON_OSCURO,
            color_texto=PAPEL_CREMA, es_usuario=True
        )

        burbuja_respuesta = self._crear_burbuja_chat(
            "Hay 2 letras 'r' en 'strawberry'.",
            color_fondo=FONDO_CAJA,
            color_texto=TINTA_NEGRA, es_usuario=False,
            t2c_dict={"2": NARANJA_TERRACOTA}
        )

        grupo_chat = VGroup(burbuja_pregunta, burbuja_respuesta).arrange(DOWN, buff=0.5)
        burbuja_pregunta.shift(RIGHT * 1.5)
        burbuja_respuesta.shift(LEFT * 1.5)
        grupo_chat.next_to(linea, DOWN, buff=0.6)

        fresa_der = self._crear_fresa().to_corner(DR).shift(UP * 0.3 + LEFT * 0.3)
        fresa_izq = self._crear_fresa().to_corner(DL).shift(UP * 0.3 + RIGHT * 0.3)

        token1 = self.crear_bloque("str", ancho=1.2)
        token2 = self.crear_bloque("aw", ancho=1.2)
        token3 = self.crear_bloque("berry", ancho=1.6)
        tokens_straw = VGroup(token1, token2, token3).arrange(RIGHT, buff=0.15)

        texto_explicacion = Text(
            "strawberry  →  tokens:",
            font=FUENTE, font_size=26, color=TINTA_NEGRA,
            t2c={"tokens:": NARANJA_TERRACOTA}
        )
        grupo_visual = VGroup(texto_explicacion, tokens_straw).arrange(DOWN, buff=0.5)
        grupo_visual.next_to(grupo_chat, DOWN, buff=1.0)

        self._animar_entrada_slide(titulo, linea, fondo=VGroup(llanuras_fondo, sol_fondo))

        self.add(fresa_der, fresa_izq)
        self.play(FadeIn(burbuja_pregunta, shift=UP * 0.2, scale=0.9))
        self.wait(0.5)
        self.play(FadeIn(burbuja_respuesta, shift=UP * 0.2, scale=0.9))

        self.play(FadeIn(texto_explicacion, shift=UP * 0.2))
        self.play(LaggedStart(
            GrowFromCenter(token1),
            GrowFromCenter(token2),
            GrowFromCenter(token3),
            lag_ratio=0.2
        ))

        cruz = Cross(tokens_straw, stroke_color=NARANJA_TERRACOTA, stroke_width=6)
        self.play(Create(cruz))
        self._siguiente()

        self.limpiar_pantalla()


