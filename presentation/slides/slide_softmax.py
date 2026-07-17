import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import os
from colores import *
from objetos import *


class SlideSoftmax:
    def slide_softmax(self):


        titulo, linea = self.crear_titulo(
            "Softmax",
            palabra_clave="Softmax",
            color_clave=NARANJA_TERRACOTA
        )
        adornos = self._crear_adornos_esquinas()
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, adornos=adornos, fondo=llanuras_fondo)


        ANC = 1.15

        def bloque(v, fondo=FONDO_CAJA, texto=MARRON_OSCURO):
            return self.crear_bloque(v, ancho=ANC, color_fondo=fondo, color_texto=texto)

        def columna(etiqueta, valores, color_etiq=MARRON_OSCURO, **kw_bloque):
            lbl = Text(etiqueta, font=FUENTE, font_size=17,
                       color=color_etiq, weight=BOLD)
            bloques = VGroup(*[bloque(v, **kw_bloque) for v in valores])\
                .arrange(DOWN, buff=0.1)
            return VGroup(lbl, bloques).arrange(DOWN, buff=0.18)

        def conector(operacion, es_tex=False):
            flecha = Arrow(LEFT, RIGHT, color=MARRON_OSCURO,
                           stroke_width=2, max_tip_length_to_length_ratio=0.25).scale(0.5)
            op = (MathTex(operacion, font_size=20, color=NARANJA_TERRACOTA)
                  if es_tex
                  else Text(operacion, font=FUENTE, font_size=16,
                             color=NARANJA_TERRACOTA, weight=BOLD))
            op.next_to(flecha, UP, buff=0.08)
            return VGroup(flecha, op)


        formula_grande = MathTex(
            r"\mathrm{softmax}(x)_i",
            r"=",
            r"\frac{e^{x_i}}{\sum_{j} e^{x_j}}",
            font_size=54, color=TINTA_NEGRA,
        ).move_to(ORIGIN)
        formula_grande[0].set_color(NARANJA_TERRACOTA)
        formula_grande[2][0:4].set_color(NARANJA_TERRACOTA)

        self.play(Write(formula_grande), run_time=1.2)
        self._siguiente()
        self.play(FadeOut(formula_grande), run_time=0.7)


        c1 = columna("Logits",  ["2.0", "1.0", "0.1"])
        k1 = conector("exp(x)")
        c2 = columna("exp(x)", ["7.39", "2.72", "1.10"],
                     fondo=CREMA_CALIDA)
        k2 = conector("/ suma")
        c3 = columna("Prob",   ["66%", "24%", "10%"],
                     color_etiq=NARANJA_TERRACOTA,
                     fondo=NARANJA_TERRACOTA, texto=PAPEL_CREMA)

        flujo1 = VGroup(c1, k1, c2, k2, c3)\
            .arrange(RIGHT, buff=0.55).move_to(DOWN * 0.3)

        suma_lbl = Text("suma = 11.21", font=FUENTE, font_size=15,
                        color=MARRON_OSCURO).next_to(c2, DOWN, buff=0.22)

        barras_prob = VGroup(*[
            RoundedRectangle(
                corner_radius=0.05, width=max(0.15, p * 2.0), height=0.32,
                fill_color=NARANJA_TERRACOTA, fill_opacity=0.9, stroke_width=0,
            ).next_to(c3[1][idx], RIGHT, buff=0.2)
            for idx, p in enumerate([0.66, 0.24, 0.10])
        ])

        self.play(FadeIn(c1, shift=UP * 0.2))
        self.play(
            Write(k1),
            ReplacementTransform(c1[1].copy(), c2[1]),
            FadeIn(c2[0])
        )
        self.play(
            Write(k2),
            FadeIn(suma_lbl, shift=UP * 0.1),
            ReplacementTransform(c2[1].copy(), c3[1]),
            FadeIn(c3[0])
        )
        self.play(
            LaggedStart(*[GrowFromEdge(b, LEFT) for b in barras_prob], lag_ratio=0.2),
            run_time=0.7,
        )
        self._siguiente()

        self.play(FadeOut(flujo1), FadeOut(suma_lbl), FadeOut(barras_prob), run_time=0.7)


        subtit_2 = Text("Problema: overflow", font=FUENTE, font_size=26,
                         color=NARANJA_TERRACOTA).next_to(linea, DOWN, buff=0.4)
        self.play(FadeIn(subtit_2, shift=UP * 0.15))

        c_e1 = columna("Logits", ["800.0", "2.0", "-1.0"])
        c_e1[1][0].set_fill(color=ROJO_TOMATE)
        c_e1[1][0][0].set_color(PAPEL_CREMA)

        k_e1 = conector("exp(x)")

        c_e2 = columna("exp(x)", ["inf", "7.39", "0.37"],
                        fondo=CREMA_CALIDA)
        c_e2[1][0].set_fill(color=ROJO_TOMATE)
        c_e2[1][0][0].set_color(PAPEL_CREMA)

        flujo_err = VGroup(c_e1, k_e1, c_e2)\
            .arrange(RIGHT, buff=0.7).next_to(subtit_2, DOWN, buff=0.5)

        self.play(FadeIn(c_e1, shift=UP * 0.2))
        self.play(Flash(c_e1[1][0], color=NARANJA_TERRACOTA, line_length=0.18))
        self.play(Write(k_e1))
        self.play(ReplacementTransform(c_e1[1].copy(), c_e2[1]), FadeIn(c_e2[0]))
        self.play(Wiggle(c_e2[1][0], scale_value=1.15))

        self._siguiente()
        self.play(FadeOut(flujo_err), FadeOut(subtit_2), run_time=0.7)


        subtit_3 = Text("Fix: restar el máximo", font=FUENTE, font_size=26,
                         color=NARANJA_TERRACOTA).next_to(linea, DOWN, buff=0.4)
        self.play(FadeIn(subtit_3, shift=UP * 0.15))

        c_f1 = columna("Logits",     ["800.0", "2.0", "-1.0"])
        k_f1 = conector("- max(x)")
        c_f2 = columna("Shifted",    ["0.0", "-798.0", "-801.0"],
                        color_etiq=NARANJA_TERRACOTA,
                        fondo=CREMA_CALIDA)
        k_f2 = conector("exp(x)")
        c_f3 = columna("exp seguro", ["1.0", "~0", "~0"],
                        fondo=VERDE_OLIVA, texto=PAPEL_CREMA)
        k_f3 = conector("/ suma")
        c_f4 = columna("Prob",       ["100%", "0%", "0%"],
                        color_etiq=NARANJA_TERRACOTA,
                        fondo=NARANJA_TERRACOTA, texto=PAPEL_CREMA)

        flujo_fix = VGroup(c_f1, k_f1, c_f2, k_f2, c_f3, k_f3, c_f4)\
            .arrange(RIGHT, buff=0.3).next_to(subtit_3, DOWN, buff=0.5)

        self.play(FadeIn(c_f1))
        self.play(Write(k_f1),
                  ReplacementTransform(c_f1[1].copy(), c_f2[1]), FadeIn(c_f2[0]))
        self.play(Write(k_f2),
                  ReplacementTransform(c_f2[1].copy(), c_f3[1]), FadeIn(c_f3[0]))
        self.play(Write(k_f3),
                  ReplacementTransform(c_f3[1].copy(), c_f4[1]), FadeIn(c_f4[0]))

        palomita_ok = VMobject(stroke_color=VERDE_OLIVA, stroke_width=4.5)
        palomita_ok.set_points_as_corners([
            [-0.14, 0.02, 0], [-0.04, -0.1, 0], [0.14, 0.12, 0],
        ])
        palomita_ok.next_to(c_f3[0], RIGHT, buff=0.15)

        self.play(
            Flash(c_f2[1][0], color=NARANJA_TERRACOTA, line_length=0.18),
            Create(palomita_ok),
        )

        self._siguiente()
        self.limpiar_pantalla()

