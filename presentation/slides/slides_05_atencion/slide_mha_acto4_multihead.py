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


class SlideMhaActo4Multihead:
    def slide_mha_acto4_multihead(self):

        titulo, linea = self.crear_titulo(
            "Multi-Head Self-Attention",
            palabra_clave="Multi-Head",
            color_clave=NARANJA_TERRACOTA
        )
        llanuras_fondo = crear_llanuras_manchegas()
        adornos = self._crear_adornos_esquinas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo, adornos=adornos)


        subtitulo = Text(
            "¿Por qué multi-head?",
            font=FUENTE, font_size=28, color=MARRON_OSCURO
        ).next_to(linea, DOWN, buff=0.4)

        self.play(FadeIn(subtitulo, shift=DOWN))


        etiqueta_vec = Text(
            "Vector de Embedding (768 dimensiones)",
            font=FUENTE, font_size=22, color=MARRON_OSCURO
        ).next_to(subtitulo, DOWN, buff=0.4)

        vector = Rectangle(
            width=10, height=0.75,
            fill_color=PAPEL_CREMA, fill_opacity=1,
            stroke_color=MARRON_OSCURO, stroke_width=2
        ).next_to(etiqueta_vec, DOWN, buff=0.2)

        self.play(FadeIn(etiqueta_vec), FadeIn(vector))


        colores_h = [NARANJA_TERRACOTA, MARRON_OSCURO, PAPEL_TAN, NARANJA_CLARO] * 3

        cabezas = VGroup(*[
            Rectangle(
                width=10/12, height=1.1,
                fill_color=colores_h[i], fill_opacity=0.9,
                stroke_color=PAPEL_CREMA, stroke_width=1.5
            )
            for i in range(12)
        ]).arrange(RIGHT, buff=0).move_to(vector.get_center())

        etiqueta_h = Text(
            "12 cabezas independientes (64 dims cada una)",
            font=FUENTE, font_size=22, color=MARRON_OSCURO
        ).move_to(etiqueta_vec.get_center())

        self.play(ReplacementTransform(vector, cabezas), Transform(etiqueta_vec, etiqueta_h))
        self.play(cabezas.animate.arrange(RIGHT, buff=0.1).move_to(cabezas.get_center()))
        self.play(
            LaggedStart(*[Indicate(c, scale_factor=1.1, color=PAPEL_CREMA) for c in cabezas], lag_ratio=0.08),
            run_time=1.5
        )


        textos_ejemplos = [
            ("Cabeza 1\n(Q1, K1, V1)", NARANJA_TERRACOTA),
            ("Cabeza 2\n(Q2, K2, V2)", MARRON_OSCURO),
            ("Cabeza h\n(Qh, Kh, Vh)", PAPEL_TAN)
        ]

        tarjetas_ej = VGroup()
        for texto, color in textos_ejemplos:
            c = RoundedRectangle(
                corner_radius=0.15, width=2.4, height=1.2,
                fill_color=PAPEL_CREMA, fill_opacity=1,
                stroke_color=color, stroke_width=2.5
            )
            t = Text(texto, font=FUENTE, font_size=18, weight=BOLD, color=color, line_spacing=1.2
            ).move_to(c.get_center())
            tarjetas_ej.add(VGroup(c, t))

        puntos = Text("...", font=FUENTE, font_size=32, color=MARRON_OSCURO, weight=BOLD)

        fila_ejemplos = VGroup(tarjetas_ej[0], tarjetas_ej[1], puntos, tarjetas_ej[2])
        fila_ejemplos.arrange(RIGHT, buff=0.6).next_to(cabezas, DOWN, buff=0.6)
        fila_ejemplos.set_x(0)

        self.play(
            LaggedStart(*[FadeIn(t, shift=UP, scale=0.9) for t in fila_ejemplos], lag_ratio=0.2),
            run_time=1.5
        )

        self._siguiente()


        self.play(FadeOut(fila_ejemplos), FadeOut(etiqueta_vec))

        etiqueta_mezcla = Text(
            "Primero se combinan (concatenan) las cabezas, luego una transformación lineal las mezcla.",
            font=FUENTE, font_size=20, color=MARRON_OSCURO
        ).next_to(cabezas, DOWN, buff=0.6)

        formula_mezcla = MathTex(
            r"\text{MultiHead}(Q, K, V) = \underbrace{\text{Concat}(head_1, \dots, head_h)}_{\text{operación estructural}} \xrightarrow{W^O} \underbrace{\text{proyección lineal}}_{\text{mezcla}}",
            color=TINTA_NEGRA, font_size=34
        ).next_to(etiqueta_mezcla, DOWN, buff=0.4)

        self.play(FadeIn(etiqueta_mezcla, shift=UP), FadeIn(formula_mezcla, shift=UP))

        vector_final = Rectangle(
            width=10, height=0.75,
            fill_color=PAPEL_CREMA, fill_opacity=1,
            stroke_color=NARANJA_TERRACOTA, stroke_width=3
        ).move_to(cabezas.get_center())

        self.play(
            *[c.animate.move_to(vector_final.get_center()).set_opacity(0.15) for c in cabezas],
            run_time=1.2
        )
        self.play(ReplacementTransform(cabezas, vector_final))
        self.wait(0.5)

        self._siguiente()
        self.limpiar_pantalla()