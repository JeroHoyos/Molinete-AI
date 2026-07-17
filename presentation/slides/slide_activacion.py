import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import numpy as np
import os
from colores import *
from objetos import *


class SlideActivacion:
    def slide_activacion(self):

        sol_fondo = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN*0.2 + LEFT*0.2)
        herradura_fondo = crear_herradura().scale(0.6).to_corner(DL).shift(UP*0.3 + RIGHT*0.3)
        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "Activación GELU",
            palabra_clave="GELU",
            color_clave=NARANJA_TERRACOTA
        )

        self._animar_entrada_slide(titulo, linea, fondo=VGroup(llanuras_fondo, sol_fondo, herradura_fondo))

        formula_principal = MathTex(
            r"\text{GELU}(x) = x \cdot \Phi(x)",
            color=TINTA_NEGRA, font_size=42
        )
        formula_aprox = MathTex(
            r"\approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} (x + 0.044715x^3)\right)\right)",
            color=MARRON_OSCURO, font_size=26
        )

        grupo_formulas = VGroup(formula_principal, formula_aprox).arrange(DOWN, buff=0.4)

        texto_contexto = Text(
            "Apagado de las neuronas",
            font=FUENTE, font_size=20, color=NARANJA_TERRACOTA, line_spacing=1.2
        )

        grupo_izq = VGroup(grupo_formulas, texto_contexto).arrange(DOWN, buff=0.8)
        grupo_izq.to_edge(LEFT, buff=1.0).shift(UP * 1.0)

        self.play(Write(formula_principal))
        self.play(FadeIn(formula_aprox, shift=UP))
        self.play(FadeIn(texto_contexto, shift=RIGHT))

        ejes = Axes(
            x_range=[-3, 3, 1], y_range=[-1, 3, 1],
            x_length=6, y_length=4.5,
            axis_config={"color": MARRON_OSCURO, "include_numbers": True, "font_size": 16}
        ).to_edge(RIGHT, buff=0.8).shift(DOWN * 0.5)

        curva_relu = ejes.plot(lambda x: np.maximum(0, x), color=BEIGE_MEDIO, stroke_width=4)
        lbl_relu = Text("ReLU", font=FUENTE, font_size=26, color=BEIGE_MEDIO).next_to(ejes.c2p(2, 2), UL, buff=0.2)

        curva_gelu = ejes.plot(
            lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))),
            color=NARANJA_TERRACOTA, stroke_width=5
        )
        lbl_gelu = Text("GELU", font=FUENTE, font_size=26, color=NARANJA_TERRACOTA).next_to(ejes.c2p(2.5, 2.5), DR, buff=0.1)

        punto_minimo = ejes.c2p(-0.75, -0.17)
        lbl_suavizado = Text("Apagado suave", font=FUENTE, font_size=16, color=MARRON_QUIJOTE).next_to(punto_minimo, DOWN, buff=0.5).shift(RIGHT * 1)
        flecha_suav = Arrow(lbl_suavizado.get_left(), punto_minimo, buff=0.1, color=MARRON_QUIJOTE, tip_length=0.15)

        self.play(Create(ejes), Write(lbl_relu))
        self.play(Create(curva_relu))
        self._siguiente()

        nodo = Circle(radius=0.7, fill_color=PAPEL_CREMA, fill_opacity=1, stroke_color=BEIGE_MEDIO, stroke_width=4)
        lbl_nodo = Text("ReLU", font=FUENTE, font_size=20, color=BEIGE_MEDIO).move_to(nodo)

        flecha_in = Arrow(ORIGIN, RIGHT * 1.5, buff=0.1, color=MARRON_OSCURO)
        val_in = MathTex("x = -1", font_size=28, color=TINTA_NEGRA).next_to(flecha_in, UP, buff=0.1)
        grupo_in = VGroup(val_in, flecha_in)

        flecha_out = Arrow(ORIGIN, RIGHT * 1.5, buff=0.1, color=BEIGE_MEDIO)
        val_out = MathTex("0", font_size=32, color=BEIGE_MEDIO).next_to(flecha_out, UP, buff=0.1)
        grupo_out = VGroup(val_out, flecha_out)

        diagrama_neurona = VGroup(grupo_in, nodo, grupo_out).arrange(RIGHT, buff=0.1)
        diagrama_neurona.to_corner(DL, buff=1.0).shift(UP * 0.5)

        lbl_nodo.move_to(nodo)

        cruz_muerte = Cross(val_out, stroke_color=RED, stroke_width=5, scale_factor=0.6)

        self.play(FadeIn(nodo), Write(lbl_nodo))
        self.play(GrowArrow(flecha_in), FadeIn(val_in, shift=RIGHT))

        val_in_anim = val_in.copy()
        self.play(val_in_anim.animate.move_to(nodo).scale(0.5).set_opacity(0), run_time=0.8)

        self.play(GrowArrow(flecha_out), FadeIn(val_out, shift=RIGHT))
        self.play(Create(cruz_muerte))
        self._siguiente()

        lbl_nodo_gelu = Text("GELU", font=FUENTE, font_size=20, color=NARANJA_TERRACOTA).move_to(nodo)

        self.play(
            ReplacementTransform(curva_relu, curva_gelu),
            ReplacementTransform(lbl_relu, lbl_gelu),
            nodo.animate.set_stroke(color=NARANJA_TERRACOTA),
            ReplacementTransform(lbl_nodo, lbl_nodo_gelu),
            flecha_out.animate.set_color(NARANJA_TERRACOTA),
            FadeOut(cruz_muerte)
        )

        val_out_gelu = MathTex("-0.15", font_size=32, color=NARANJA_TERRACOTA).next_to(flecha_out, UP, buff=0.1)
        chispa = Star(n=5, outer_radius=0.25, inner_radius=0.12, color=MARRON_QUIJOTE, fill_opacity=1).next_to(val_out_gelu, RIGHT, buff=0.2)

        val_in_anim_2 = val_in.copy()
        self.play(val_in_anim_2.animate.move_to(nodo).scale(0.5).set_opacity(0), run_time=0.8)

        self.play(
            ReplacementTransform(val_out, val_out_gelu),
            Create(chispa)
        )

        self.play(FadeIn(lbl_suavizado, shift=UP), GrowArrow(flecha_suav))
        self._siguiente()

        self.limpiar_pantalla()


