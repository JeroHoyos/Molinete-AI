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


class SlideTemperature:
    def slide_temperature(self):

        sol_fondo = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN*0.2 + LEFT*0.2)
        herradura_fondo = crear_herradura().scale(0.6).to_corner(DL).shift(UP*0.3 + RIGHT*0.3)
        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "La Temperatura: Controlando la Locura",
            palabra_clave="Temperatura",
            color_clave=NARANJA_TERRACOTA
        )

        self._animar_entrada_slide(titulo, linea, fondo=VGroup(llanuras_fondo, sol_fondo, herradura_fondo))

        label_eq = Text("Softmax con Temperatura (T)", font=FUENTE, font_size=18, color=MARRON_OSCURO)

        eq_temp = MathTex(
            r"P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}",
            color=TINTA_NEGRA
        ).scale(1.3)

        eq_temp.set_color_by_tex("T", NARANJA_TERRACOTA)

        math_group = VGroup(label_eq, eq_temp).arrange(DOWN, buff=0.3).move_to(ORIGIN).shift(UP * 0.5)

        self.play(FadeIn(label_eq, shift=DOWN*0.2), Write(eq_temp), run_time=1.5)
        self.play(eq_temp.animate.scale(1.1).set_glow(0.3), rate_func=there_and_back, run_time=1)
        self._siguiente()

        explicacion = Tex(
            r"$T \to 0$: Determinista, conservador (Como Sancho)\\$T > 1$: Aleatorio, creativo, ``alucinaciones'' (Como el Quijote)",
            font_size=28, color=MARRON_OSCURO, tex_environment="flushleft"
        ).next_to(math_group, DOWN, buff=0.8)

        self.play(FadeIn(explicacion, shift=UP))

        self.play(FadeOut(math_group, explicacion))

        rect_prompt = RoundedRectangle(corner_radius=0.15, height=1.2, width=8)
        rect_prompt.set_fill(color=MARRON_OSCURO, opacity=0.1).set_stroke(color=MARRON_OSCURO, width=1.5)

        user_label = Text("Usuario", font=FUENTE, font_size=14, color=MARRON_OSCURO).next_to(rect_prompt, UP, aligned_edge=LEFT).shift(DOWN*0.1 + RIGHT*0.2)

        texto_prompt = Text(
            "Prompt: \"En un lugar de la Mancha...\"",
            font=FUENTE, font_size=22, color=TINTA_NEGRA, weight=BOLD
        ).move_to(rect_prompt)

        grupo_prompt = VGroup(rect_prompt, user_label, texto_prompt).to_edge(UP, buff=1.5)

        self.play(FadeIn(grupo_prompt, shift=DOWN))

        def crear_respuesta(temp_val, intento, texto, color_perfil, titulo_perfil):
            sombra = RoundedRectangle(corner_radius=0.15, height=2.2, width=8)
            sombra.set_fill(MARRON_OSCURO, opacity=0.1).set_stroke(width=0)
            sombra.shift(RIGHT * 0.08 + DOWN * 0.08)

            rect = RoundedRectangle(corner_radius=0.15, height=2.2, width=8)
            rect.set_fill(color=PAPEL_CREMA, opacity=1).set_stroke(color=MARRON_OSCURO, width=1.5)

            icon = Circle(radius=0.25, color=color_perfil, fill_opacity=1)
            label_icon = Text(titulo_perfil[5], font=FUENTE, font_size=20, color=BLANCO, weight=BOLD).move_to(icon)
            user_icon = VGroup(icon, label_icon).next_to(rect, LEFT, buff=0.3).shift(UP * 0.5)

            username = Text(f"{titulo_perfil} (T={temp_val})", font=FUENTE, font_size=14, color=MARRON_OSCURO).next_to(rect, UP, aligned_edge=LEFT).shift(UP*0.1)

            contenido = Paragraph(
                texto, font=FUENTE, font_size=22, color=TINTA_NEGRA,
                line_spacing=1.3, alignment="left"
            ).scale_to_fit_width(rect.width - 0.8).move_to(rect)

            bubble_group = VGroup(sombra, rect, user_icon, username, contenido)

            info = Text(f"Generación - Intento #{intento}",
                        font="Monospace", font_size=16, color=color_perfil).next_to(rect, DOWN, buff=0.15, aligned_edge=RIGHT)

            return VGroup(bubble_group, info).next_to(grupo_prompt, DOWN, buff=1.0)

        r_sancho_1 = crear_respuesta("0.1", 1, "\"...de cuyo nombre no quiero acordarme.\"", MARRON_OSCURO, "Sancho")
        r_sancho_2 = crear_respuesta("0.1", 2, "\"...de cuyo nombre no quiero acordarme.\"", MARRON_OSCURO, "Sancho")

        r_quijote_1 = crear_respuesta("1.5", 1, "\"...donde los dragones mecánicos beben aceite de oliva.\"", NARANJA_TERRACOTA, "El Quijote")
        r_quijote_2 = crear_respuesta("1.5", 2, "\"...los molinos me hablan en código binario al amanecer.\"", NARANJA_TERRACOTA, "El Quijote")

        actual = r_sancho_1
        self.play(FadeIn(actual, shift=UP))
        self._siguiente()

        self.play(FadeTransform(actual, r_sancho_2), run_time=1)
        actual = r_sancho_2
        self._siguiente()

        self.play(FadeTransform(actual, r_quijote_1), run_time=1.5)
        actual = r_quijote_1
        self._siguiente()

        self.play(FadeTransform(actual, r_quijote_2), run_time=1)
        self._siguiente()

        self.limpiar_pantalla()
