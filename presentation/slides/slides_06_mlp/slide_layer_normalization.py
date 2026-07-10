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


class SlideLayerNormalization:
    def slide_layer_normalization(self):


        def crear_cajita(texto, bg_color, borde_color=MARRON_OSCURO, w=2.6, h=0.7, tam_fuente=20):
            caja = RoundedRectangle(corner_radius=0.1, width=w, height=h,
                                    fill_color=bg_color, fill_opacity=1,
                                    stroke_color=borde_color, stroke_width=2)
            lbl = Text(texto, font_size=tam_fuente, color=TINTA_NEGRA).move_to(caja.get_center())
            return VGroup(caja, lbl)

        def crear_vector_visual(numeros, bg_color, borde_color=MARRON_OSCURO):
            return VGroup(*[
                crear_cajita(num, bg_color, borde_color, w=1.6, h=0.7, tam_fuente=18)
                for num in numeros
            ]).arrange(DOWN, buff=0.1)


        titulo_p1 = Text("Layer ", font_size=42, weight=BOLD, color=TINTA_NEGRA)
        titulo_p2 = Text("Normalization", font_size=42, weight=BOLD, color=NARANJA_TERRACOTA)
        titulo_completo = VGroup(titulo_p1, titulo_p2).arrange(RIGHT, buff=0.1)
        linea = Line(LEFT * 4, RIGHT * 4, color=MARRON_OSCURO).next_to(titulo_completo, DOWN)
        grupo_titulo = VGroup(titulo_completo, linea).to_edge(UP)

        adornos = self._crear_adornos_esquinas()
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo_completo, linea, adornos=adornos, fondo=llanuras_fondo)


        lbl_antes   = Text("Sin norma",     font=FUENTE, font_size=20,
                            color=NARANJA_TERRACOTA, weight=BOLD)
        lbl_despues = Text("Con LayerNorm", font=FUENTE, font_size=20,
                            color=MARRON_OSCURO, weight=BOLD)

        pares = [
            ("8459.1",  "1.34"),
            ("-7302.4", "-1.15"),
            ("0.00001", "0.00"),
            ("5120.9",  "0.89"),
            ("-9999.9", "-1.52"),
        ]
        inestables = [p[0] for p in pares]
        estables   = [p[1] for p in pares]

        vec_inestable = crear_vector_visual(
            inestables, bg_color=SALMON_CLARO, borde_color=NARANJA_TERRACOTA
        )
        vec_estable = crear_vector_visual(
            estables, bg_color=CREMA_CALIDA, borde_color=MARRON_OSCURO
        )


        lbl_antes.next_to(vec_inestable, UP, buff=0.3)
        lbl_despues.next_to(vec_estable, UP, buff=0.3)

        bloque_izq = VGroup(lbl_antes,   vec_inestable)
        bloque_der = VGroup(lbl_despues, vec_estable)

        VGroup(bloque_izq, bloque_der).arrange(RIGHT, buff=1.8).move_to(ORIGIN)

        stats_antes = Text(
            "μ ≈ -1344   σ ≈ 7210",
            font=FUENTE, font_size=17, color=NARANJA_TERRACOTA
        ).next_to(vec_inestable, DOWN, buff=0.3)

        stats_despues = Text(
            "μ = 0   σ = 1",
            font=FUENTE, font_size=17, color=MARRON_OSCURO, weight=BOLD
        ).next_to(vec_estable, DOWN, buff=0.3)


        self.play(
            FadeIn(lbl_antes, shift=UP * 0.1),
            LaggedStart(*[FadeIn(c, shift=UP * 0.15) for c in vec_inestable],
                        lag_ratio=0.1),
            run_time=1.0
        )
        self.play(
            Indicate(vec_inestable, color=NARANJA_TERRACOTA, scale_factor=1.05),
            FadeIn(stats_antes, shift=UP * 0.1)
        )
        self.play(
            LaggedStart(*[
                ReplacementTransform(vec_inestable[i].copy(), vec_estable[i])
                for i in range(len(pares))
            ], lag_ratio=0.15),
            FadeIn(lbl_despues, shift=UP * 0.1),
            run_time=1.4
        )
        self.play(FadeIn(stats_despues, shift=UP * 0.1))
        self._siguiente()

        self.play(
            FadeOut(bloque_izq), FadeOut(bloque_der),
            FadeOut(stats_antes), FadeOut(stats_despues),
            run_time=0.7
        )


        formula = MathTex(
            r"\text{output} = \frac{\text{input} - \mu}{\sqrt{\sigma^2 + \epsilon}} \times \gamma + \beta",
            substrings_to_isolate=[r"\epsilon", r"\times \gamma + \beta"],
            color=TINTA_NEGRA
        ).scale(1.2).move_to(DOWN * 0.5)

        lbl_formula = Text("Por cada token:", font_size=20, weight=BOLD,
                            color=MARRON_OSCURO).next_to(formula, UP, buff=0.5)

        self.play(Write(lbl_formula), FadeIn(formula))

        parte_eps = formula.get_part_by_tex(r"\epsilon")
        caja_eps  = SurroundingRectangle(parte_eps, color=NARANJA_TERRACOTA, buff=0.05)
        nota_eps  = Text("ε previene división por cero", font_size=18,
                          color=NARANJA_TERRACOTA).next_to(caja_eps, DOWN, buff=0.5)

        self.play(Create(caja_eps), FadeIn(nota_eps, shift=UP))
        self.play(FadeOut(caja_eps), FadeOut(nota_eps))

        parte_params = formula.get_part_by_tex(r"\times \gamma + \beta")
        caja_params  = SurroundingRectangle(parte_params, color=MARRON_OSCURO, buff=0.1)
        nota_params  = Text("γ, β: parámetros aprendibles", font_size=18,
                             color=MARRON_OSCURO).next_to(caja_params, DOWN, buff=0.5)

        self.play(Create(caja_params), FadeIn(nota_params, shift=UP))
        self._siguiente()

        self.play(
            *[FadeOut(m) for m in [lbl_formula, formula, caja_params, nota_params]]
        )

        self.limpiar_pantalla()


