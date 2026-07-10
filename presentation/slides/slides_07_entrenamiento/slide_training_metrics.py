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


class SlideTrainingMetrics:
    def slide_training_metrics(self):

        llanuras_fondo = crear_llanuras_manchegas()
        adornos = self._crear_adornos_esquinas(escala=0.55, buff=0.5)
        adornos[1].add_updater(lambda m, dt: m.rotate(dt * 0.4))

        titulo, linea = self.crear_titulo(
            "Métricas de Entrenamiento: Loss y Perplejidad",
            palabra_clave="Métricas",
            color_clave=NARANJA_TERRACOTA
        )
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo, adornos=adornos)


        lbl_a1 = Text("Loss = sorpresa del modelo ante la palabra correcta",
                      font=FUENTE, font_size=25, weight=BOLD, color=TINTA_NEGRA
                      ).next_to(linea, DOWN, buff=0.38)
        self.play(FadeIn(lbl_a1, shift=DOWN * 0.1))

        ctx_bg  = RoundedRectangle(corner_radius=0.14, width=7.8, height=0.62,
                                    fill_color=CAJA_INFERIOR, fill_opacity=0.65,
                                    stroke_color=MARRON_OSCURO, stroke_width=1.8)
        ctx_txt = Text('"...Fortuna es una mujer  ___"', font=FUENTE, font_size=20,
                       color=TINTA_NEGRA, t2c={"___": NARANJA_TERRACOTA}).move_to(ctx_bg)
        ctx_grp = VGroup(ctx_bg, ctx_txt).next_to(lbl_a1, DOWN, buff=0.32)
        self.play(FadeIn(ctx_grp))

        palabras_pred = ["borracha", "bella", "rica", "alta"]
        probs_mal     = [0.06, 0.29, 0.22, 0.43]
        probs_bien    = [0.74, 0.12, 0.09, 0.05]
        BAR_W, BAR_MAX_H = 0.52, 1.55
        COL_SEP = BAR_W + 0.32

        def _panel_barras(probs, color_winner, titulo_str, loss_str, loss_color):
            rects, pcts, wrds = [], [], []
            for i, (p, pal) in enumerate(zip(probs, palabras_pred)):
                h = max(0.06, p * BAR_MAX_H)
                rect = Rectangle(width=BAR_W, height=h,
                                 fill_color=color_winner if i == 0 else PAPEL_TAN,
                                 fill_opacity=0.88, stroke_width=1.1,
                                 stroke_color=MARRON_OSCURO)
                rect.move_to(np.array([i * COL_SEP, h / 2, 0]))
                pct = Text(f"{int(p*100)}%", font="Monospace", font_size=13,
                           color=TINTA_NEGRA).move_to(np.array([i * COL_SEP, h + 0.19, 0]))
                wrd = Text(pal, font=FUENTE, font_size=13,
                           color=TINTA_NEGRA).move_to(np.array([i * COL_SEP, -0.25, 0]))
                rects.append(rect); pcts.append(pct); wrds.append(wrd)

            barras = VGroup(*[VGroup(r, p, w) for r, p, w in zip(rects, pcts, wrds)])
            barras.center()
            tit      = Text(titulo_str, font=FUENTE, font_size=17, weight=BOLD,
                            color=color_winner).next_to(barras, UP, buff=0.35)
            loss_lbl = Text(loss_str, font="Monospace", font_size=16, weight=BOLD,
                            color=loss_color).next_to(barras, DOWN, buff=0.28)
            return VGroup(tit, barras, loss_lbl)

        panel_mal  = _panel_barras(probs_mal,  ROJO_TOMATE, "Sin entrenar", "Loss ≈ 8.1", ROJO_TOMATE)
        panel_bien = _panel_barras(probs_bien, VERDE_OLIVA, "Entrenado",   "Loss ≈ 2.5", VERDE_OLIVA)

        comparacion = VGroup(panel_mal, panel_bien).arrange(RIGHT, buff=1.6)
        comparacion.next_to(ctx_grp, DOWN, buff=0.28).set_x(0)

        sep = DashedLine(
            comparacion.get_top() + UP * 0.1,
            comparacion.get_bottom() + DOWN * 0.1,
            color=MARRON_OSCURO, stroke_width=1.2, dash_length=0.10,
        ).set_x(comparacion.get_center()[0])

        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.12) for b in panel_mal[1]], lag_ratio=0.1),
            FadeIn(panel_mal[0]), run_time=0.9,
        )
        self.play(Write(panel_mal[2]))
        self.play(Create(sep))
        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.12) for b in panel_bien[1]], lag_ratio=0.1),
            FadeIn(panel_bien[0]), run_time=0.9,
        )
        self.play(Write(panel_bien[2]))

        formula = MathTex(r"L = -\log P(\text{borracha}\mid\text{contexto})",
                          font_size=22, color=MARRON_OSCURO)
        formula.next_to(comparacion, DOWN, buff=0.25)
        self.play(Write(formula))

        self._siguiente()
        self.play(FadeOut(lbl_a1, ctx_grp, panel_mal, panel_bien, sep, formula))


        lbl_a2 = Text("Perplejidad = ¿cuántas opciones baraja el modelo?",
                      font=FUENTE, font_size=24, weight=BOLD, color=TINTA_NEGRA
                      ).next_to(linea, DOWN, buff=0.38)
        self.play(Write(lbl_a2))

        eq_ppl = MathTex(r"PPL = e^{\,L}", font_size=46, color=NARANJA_TERRACOTA)
        eq_ppl.next_to(lbl_a2, DOWN, buff=0.38)
        self.play(Write(eq_ppl))

        def _tarjeta_ppl(ppl_label, descripcion, n_boxes, color):
            tit = Text(f"PPL {ppl_label}", font="Monospace", font_size=24,
                       weight=BOLD, color=color)
            n_show = min(n_boxes, 7)
            cajas_viz = VGroup(*[
                RoundedRectangle(corner_radius=0.05, width=0.40, height=0.32,
                                 fill_color=color if i == 0 else PAPEL_TAN,
                                 fill_opacity=0.72 if i == 0 else 0.38,
                                 stroke_color=MARRON_OSCURO, stroke_width=1.0)
                for i in range(n_show)
            ]).arrange(RIGHT, buff=0.07)
            if n_boxes > n_show:
                puntos_lbl = Text("···", font="Monospace", font_size=16,
                                  color=MARRON_OSCURO).next_to(cajas_viz, RIGHT, buff=0.1)
                cajas_viz = VGroup(cajas_viz, puntos_lbl)
            desc = Text(descripcion, font=FUENTE, font_size=15,
                        color=MARRON_OSCURO, line_spacing=1.2)
            contenido = VGroup(tit, cajas_viz, desc).arrange(DOWN, buff=0.22)
            fondo = SurroundingRectangle(contenido, color=color,
                                         fill_color=FONDO_CAJA, fill_opacity=1,
                                         buff=0.30, corner_radius=0.16, stroke_width=2.2)
            return VGroup(fondo, contenido)

        card_alto = _tarjeta_ppl("3 360", "Elegir entre\n3360 opciones.", 3360, ROJO_TOMATE)
        card_bajo = _tarjeta_ppl("17", "Elegir entre\n17 opciones.", 17, VERDE_OLIVA)

        tarjetas = VGroup(card_alto, card_bajo).arrange(RIGHT, buff=1.1)
        tarjetas.next_to(eq_ppl, DOWN, buff=0.38).set_x(0)
        if tarjetas.width > 12.8:
            tarjetas.scale(12.8 / tarjetas.width)

        self.play(FadeIn(card_alto, scale=0.92, shift=RIGHT * 0.15), run_time=0.8)
        self.play(FadeIn(card_bajo, scale=0.92, shift=LEFT * 0.15), run_time=0.8)

        flecha_t = Arrow(card_alto.get_right(), card_bajo.get_left(),
                         color=VERDE_OLIVA, stroke_width=3,
                         max_tip_length_to_length_ratio=0.2)
        lbl_flt  = Text("entrenar", font=FUENTE, font_size=14, color=VERDE_OLIVA,
                        slant=ITALIC).next_to(flecha_t, UP, buff=0.1)
        self.play(GrowArrow(flecha_t), FadeIn(lbl_flt, shift=DOWN * 0.1))

        self._siguiente()
        self.play(FadeOut(lbl_a2, eq_ppl, card_alto, card_bajo, flecha_t, lbl_flt))


        lbl_a3 = Text("A medida que el Loss cae, el texto mejora",
                      font=FUENTE, font_size=25, weight=BOLD, color=TINTA_NEGRA
                      ).next_to(linea, DOWN, buff=0.38)
        self.play(Write(lbl_a3))

        def curva_loss(t):
            return 2.85 + 5.27 * np.exp(-t / 4200)

        ax = Axes(
            x_range=[0, 16000, 4000],
            y_range=[0, 9, 3],
            x_length=6.2,
            y_length=3.6,
            axis_config={"include_tip": False, "color": MARRON_OSCURO},
        ).shift(LEFT * 2.0 + DOWN * 0.5)

        x_lbl = Text("Pasos", font=FUENTE, font_size=13,
                     color=MARRON_OSCURO).next_to(ax, DOWN, buff=0.15)
        y_lbl = Text("Loss", font=FUENTE, font_size=13,
                     color=MARRON_OSCURO).next_to(ax, LEFT, buff=0.12)

        curva = ax.plot(curva_loss, x_range=[0, 16000],
                        color=NARANJA_TERRACOTA, stroke_width=4)

        self.play(Create(ax), Write(x_lbl), Write(y_lbl))
        self.play(Create(curva), run_time=2.0)

        checkpoints = [
            (0,     8.12, "3 360", ROJO_TOMATE,
             '"q7e llam8n p0r a#í\nF0rtun4 es una muj8r..."'),
            (4000,  4.45, "85",    PAPEL_TAN,
             '"Esta que llaman por\nahí Fortuna es una mujer..."'),
            (16000, 2.85, "17",    VERDE_OLIVA,
             '"Esta que llaman por ahí\nFortuna es una mujer borracha."'),
        ]

        dot_actual = None
        burbuja_actual = None

        for paso, loss_v, ppl_v, color, texto in checkpoints:
            pos_pt = ax.c2p(paso, curva_loss(paso))
            nuevo_dot = Dot(pos_pt, radius=0.12, color=color,
                            fill_opacity=1, stroke_color=BLANCO, stroke_width=1.5)

            rect_b = RoundedRectangle(corner_radius=0.16, width=4.8, height=1.62,
                                      fill_color=PAPEL_CREMA, fill_opacity=1,
                                      stroke_color=color, stroke_width=2.5)
            txt_b  = Text(texto, font=FUENTE, font_size=17, color=TINTA_NEGRA,
                          line_spacing=1.2)
            txt_b.scale_to_fit_width(rect_b.width - 0.55).move_to(rect_b)
            info_b = Text(f"Paso {paso:,}   Loss {loss_v}   PPL {ppl_v}",
                          font="Monospace", font_size=13, color=color
                          ).next_to(rect_b, DOWN, buff=0.1, aligned_edge=RIGHT)
            nueva_burbuja = VGroup(rect_b, txt_b, info_b).move_to(RIGHT * 3.2 + DOWN * 0.35)

            if dot_actual is None:
                self.play(FadeIn(nuevo_dot, scale=0.5))
                self.play(FadeIn(nueva_burbuja, shift=LEFT * 0.2))
            else:
                self.play(
                    ReplacementTransform(dot_actual, nuevo_dot),
                    FadeTransform(burbuja_actual, nueva_burbuja),
                    run_time=1.1,
                )
            dot_actual = nuevo_dot
            burbuja_actual = nueva_burbuja
            self._siguiente()

        adornos[1].clear_updaters()
        self.play(FadeOut(lbl_a3, ax, x_lbl, y_lbl, curva, dot_actual, burbuja_actual))
        self.limpiar_pantalla()

