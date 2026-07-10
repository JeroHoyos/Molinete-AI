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


class SlideBackpropagation:
    def slide_backpropagation(self):
        titulo, linea = self.crear_titulo(
            "Backpropagation: Regla de la Cadena",
            palabra_clave="Regla de la Cadena",
            color_clave=NARANJA_TERRACOTA
        )
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)

        pregunta = Text(
            "¿Qué parámetro tuvo la culpa?",
            font=FUENTE, font_size=24, weight=BOLD,
            color=TINTA_NEGRA, line_spacing=1.4
        ).move_to(UP * 1.8)
        self.play(Write(pregunta))
        self._siguiente()

        cadena_palabras = ["L", "GELU", "suma", "mul", "w₀"]
        cadena_nodos = VGroup()
        for i, p in enumerate(cadena_palabras):
            rect = RoundedRectangle(corner_radius=0.1, width=1.4, height=0.6,
                                    fill_color=PAPEL_CREMA, fill_opacity=0.9,
                                    stroke_color=MARRON_OSCURO, stroke_width=2)
            lbl = Text(p, font=FUENTE, font_size=20,
                       color=NARANJA_TERRACOTA if i == 0 else TINTA_NEGRA, weight=BOLD)
            lbl.move_to(rect)
            cadena_nodos.add(VGroup(rect, lbl))

        cadena_nodos.arrange(RIGHT, buff=0.55).next_to(pregunta, DOWN, buff=0.55)

        flechas_cadena = VGroup(*[
            Arrow(cadena_nodos[i].get_right(), cadena_nodos[i + 1].get_left(),
                  buff=0.08, color=MARRON_OSCURO,
                  max_tip_length_to_length_ratio=0.3, stroke_width=2.5)
            for i in range(len(cadena_nodos) - 1)
        ])

        lbl_cadena = Text("¿Cuánto contribuyó cada uno?",
                          font=FUENTE, font_size=19, color=MARRON_OSCURO
                          ).next_to(cadena_nodos, DOWN, buff=0.35)

        animaciones_lagged = []
        for i in range(len(cadena_nodos)):
            anims = [FadeIn(cadena_nodos[i], shift=RIGHT * 0.15)]
            if i < len(flechas_cadena):
                anims.append(Create(flechas_cadena[i]))
            animaciones_lagged.append(AnimationGroup(*anims))

        if animaciones_lagged:
            self.play(LaggedStart(*animaciones_lagged, lag_ratio=0.2))

        self.play(Write(lbl_cadena))

        self.play(FadeOut(pregunta), FadeOut(cadena_nodos),
                  FadeOut(flechas_cadena), FadeOut(lbl_cadena))

        lbl_herramienta = Text("Regla de la Cadena",
                               font=FUENTE, font_size=26, weight=BOLD,
                               color=TINTA_NEGRA).move_to(UP * 2.6)
        self.play(Write(lbl_herramienta))

        analogia_lbl = Text(
            "Multiplica las pendientes del camino:",
            font=FUENTE, font_size=20, color=MARRON_OSCURO, line_spacing=1.4
        ).next_to(lbl_herramienta, DOWN, buff=0.4)
        self.play(FadeIn(analogia_lbl, shift=UP * 0.2))
        self._siguiente()

        eq_cadena = MathTex(
            r"\frac{\partial L}{\partial w} = "
            r"\frac{\partial L}{\partial \hat{y}} \cdot "
            r"\frac{\partial \hat{y}}{\partial \text{sum}} \cdot "
            r"\frac{\partial \text{sum}}{\partial \text{mul}} \cdot "
            r"\frac{\partial \text{mul}}{\partial w}",
            font_size=34, color=TINTA_NEGRA
        ).next_to(analogia_lbl, DOWN, buff=0.5)

        self.play(Write(eq_cadena), run_time=2.0)
        self._siguiente()

        caja_eq = SurroundingRectangle(eq_cadena, color=NARANJA_TERRACOTA,
                                       buff=0.2, corner_radius=0.1, stroke_width=2.5)
        nota_local = Text(
            "Gradientes locales · de derecha a izquierda",
            font=FUENTE, font_size=19, color=NARANJA_TERRACOTA, line_spacing=1.3
        ).next_to(eq_cadena, DOWN, buff=0.4)
        self.play(Create(caja_eq))
        self.play(Write(nota_local))

        self.play(FadeOut(lbl_herramienta), FadeOut(analogia_lbl),
                  FadeOut(eq_cadena), FadeOut(caja_eq), FadeOut(nota_local))

        lbl_grafo = Text("forward → backward",
                         font=FUENTE, font_size=24, weight=BOLD,
                         color=TINTA_NEGRA).move_to(UP * 2.7)
        self.play(Write(lbl_grafo))

        EJE_Y = UP * 1.0

        def crear_nodo_op(texto, pos, es_texto=False):
            circ = Circle(radius=0.48, fill_color=FONDO_CAJA, fill_opacity=1,
                          stroke_color=MARRON_OSCURO, stroke_width=3)
            circ.move_to(pos)
            if es_texto:
                etq = Text(texto, font=FUENTE, font_size=15,
                           color=MARRON_OSCURO, weight=BOLD).move_to(circ)
            else:
                etq = MathTex(texto, font_size=38,
                              color=MARRON_OSCURO).move_to(circ)
            return VGroup(circ, etq), circ

        grp_mul, nd_mul = crear_nodo_op(r"\times", LEFT * 3.5 + EJE_Y)
        grp_sum, nd_sum = crear_nodo_op(r"+", LEFT * 0.0 + EJE_Y)
        grp_gelu, nd_gelu = crear_nodo_op("GELU", RIGHT * 3.5 + EJE_Y, es_texto=True)

        tx0 = MathTex("x_0", font_size=30, color=TINTA_NEGRA
                      ).move_to(nd_mul.get_left() + LEFT * 1.4 + UP * 0.55)
        tw0 = MathTex("w_0", font_size=30, color=AZUL_NOCHE
                      ).move_to(nd_mul.get_left() + LEFT * 1.4 + DOWN * 0.55)
        tb = MathTex("b", font_size=30, color=AZUL_NOCHE
                     ).move_to(nd_sum.get_bottom() + DOWN * 1.1)
        ty = MathTex(r"\hat{y}", font_size=34, color=NARANJA_TERRACOTA
                     ).move_to(nd_gelu.get_right() + RIGHT * 1.4)

        def arista(a, b, color=MARRON_OSCURO, sw=2):
            return Line(a, b, color=color, stroke_width=sw, z_index=-1).add_tip(
                tip_length=0.18, tip_width=0.18)

        f_x0 = arista(tx0.get_right(), nd_mul.get_left() + UP * 0.2)
        f_w0 = arista(tw0.get_right(), nd_mul.get_left() + DOWN * 0.2, color=AZUL_NOCHE)
        f_b = arista(tb.get_top(), nd_sum.get_bottom(), color=AZUL_NOCHE)
        f_ms = arista(nd_mul.get_right(), nd_sum.get_left())
        f_sg = arista(nd_sum.get_right(), nd_gelu.get_left())
        f_out = arista(nd_gelu.get_right(), ty.get_left(), color=NARANJA_TERRACOTA)

        lbl_ms = Text("mul", font=FUENTE, font_size=14,
                      color=MARRON_OSCURO).next_to(f_ms, UP, buff=0.08)
        lbl_sg = Text("sum", font=FUENTE, font_size=14,
                      color=MARRON_OSCURO).next_to(f_sg, UP, buff=0.08)

        grafo_fwd = VGroup(grp_mul, grp_sum, grp_gelu,
                           tx0, tw0, tb, ty,
                           f_x0, f_w0, f_b, f_ms, f_sg, f_out,
                           lbl_ms, lbl_sg)

        lbl_fwd = Text("① Forward pass", font=FUENTE, font_size=18,
                       weight=BOLD, color=VERDE_OLIVA).next_to(lbl_grafo, DOWN, buff=0.15)
        self.play(Write(lbl_fwd))
        self.play(LaggedStart(
            AnimationGroup(FadeIn(tx0), FadeIn(tw0)),
            AnimationGroup(Create(f_x0), Create(f_w0)),
            DrawBorderThenFill(grp_mul),
            AnimationGroup(Create(f_ms), FadeIn(lbl_ms), FadeIn(tb), Create(f_b)),
            DrawBorderThenFill(grp_sum),
            AnimationGroup(Create(f_sg), FadeIn(lbl_sg)),
            DrawBorderThenFill(grp_gelu),
            AnimationGroup(Create(f_out), Write(ty)),
            lag_ratio=0.25, run_time=2.5
        ))
        self._siguiente()

        lbl_bwd = Text("② Backward pass  (gradientes de derecha a izquierda)",
                       font=FUENTE, font_size=18, weight=BOLD,
                       color=NARANJA_TERRACOTA).move_to(lbl_fwd)
        self.play(ReplacementTransform(lbl_fwd, lbl_bwd))

        rutas_back = [
            (nd_gelu.get_left(), nd_sum.get_right(), r"\partial\hat{y}/\partial\text{sum}"),
            (nd_sum.get_left(), nd_mul.get_right(), r"\partial\text{sum}/\partial\text{mul}"),
            (nd_mul.get_left() + DOWN * 0.2, tw0.get_right(), r"\partial\text{mul}/\partial w_0"),
            (nd_sum.get_bottom(), tb.get_top(), r"\partial\text{sum}/\partial b"),
        ]

        for p_start, p_end, grad_tex in rutas_back:
            flash_line = Line(p_start, p_end, color=NARANJA_TERRACOTA, stroke_width=5)
            grad_lbl = MathTex(grad_tex, font_size=20, color=NARANJA_TERRACOTA
                               ).move_to(flash_line.get_center() + UP * 0.35)
            self.play(ShowPassingFlash(flash_line, time_width=0.55), run_time=0.7)
            self.play(FadeIn(grad_lbl, shift=UP * 0.1), run_time=0.35)
            self.play(FadeOut(grad_lbl), run_time=0.25)

        caja_w0 = SurroundingRectangle(tw0, color=NARANJA_TERRACOTA,
                                       buff=0.1, corner_radius=0.08, stroke_width=2.5)
        caja_b = SurroundingRectangle(tb, color=NARANJA_TERRACOTA,
                                      buff=0.1, corner_radius=0.08, stroke_width=2.5)
        lbl_upd_w = MathTex(r"-\eta\,\Delta w_0", font_size=20,
                            color=NARANJA_TERRACOTA).next_to(caja_w0, LEFT, buff=0.15)
        lbl_upd_b = MathTex(r"-\eta\,\Delta b", font_size=20,
                            color=NARANJA_TERRACOTA).next_to(caja_b, LEFT, buff=0.15)

        self.play(Create(caja_w0), Create(caja_b))
        self.play(FadeIn(lbl_upd_w, shift=RIGHT * 0.15),
                  FadeIn(lbl_upd_b, shift=RIGHT * 0.15))
        self.play(
            Indicate(lbl_upd_w, color=ORO_VIEJO, scale_factor=1.2),
            Indicate(lbl_upd_b, color=ORO_VIEJO, scale_factor=1.2),
        )

        self.play(FadeOut(grafo_fwd), FadeOut(lbl_bwd),
                  FadeOut(caja_w0), FadeOut(caja_b),
                  FadeOut(lbl_upd_w), FadeOut(lbl_upd_b), FadeOut(lbl_grafo))

        lbl_red = Text("En la red completa: millones de parámetros, mismo principio",
                       font=FUENTE, font_size=22, weight=BOLD,
                       color=TINTA_NEGRA).move_to(UP * 2.7)
        self.play(Write(lbl_red))

        capas_config = [3, 5, 4, 2]
        nodos_red = VGroup()
        for i, n in enumerate(capas_config):
            capa = VGroup(*[
                Circle(radius=0.22, fill_color=FONDO_CAJA, fill_opacity=1,
                       stroke_color=MARRON_OSCURO, stroke_width=2.5)
                for _ in range(n)
            ]).arrange(DOWN, buff=0.45)
            capa.move_to(RIGHT * (i * 2.6 - 3.9) + DOWN * 0.35)
            nodos_red.add(capa)

        conexiones_fwd_red = VGroup()
        conexiones_back_por_capa = []
        for i in range(len(capas_config) - 1):
            grupo_back = VGroup()
            for n1 in nodos_red[i]:
                for n2 in nodos_red[i + 1]:
                    ln_fwd = Line(n1.get_right(), n2.get_left(),
                                  stroke_width=1.5, color=MARRON_OSCURO,
                                  z_index=-1).set_opacity(0.25)
                    conexiones_fwd_red.add(ln_fwd)
                    ln_back = Line(n2.get_left(), n1.get_right(),
                                   stroke_width=4, color=NARANJA_TERRACOTA)
                    grupo_back.add(ln_back)
            conexiones_back_por_capa.append(grupo_back)

        nombres_capas = ["Entrada", "Capa 1", "Capa 2", "Salida"]
        lbls_capas = VGroup(*[
            Text(nombres_capas[i], font=FUENTE, font_size=14, color=MARRON_OSCURO
                 ).next_to(nodos_red[i], DOWN, buff=0.3)
            for i in range(len(capas_config))
        ])

        self.play(LaggedStart(
            *[GrowFromCenter(n) for capa in nodos_red for n in capa],
            lag_ratio=0.06
        ), run_time=1.2)
        self.play(Create(conexiones_fwd_red), FadeIn(lbls_capas), run_time=1.0)

        txt_loss_red = MathTex(r"L", font_size=30,
                               color=ROJO_TOMATE).next_to(nodos_red[-1], RIGHT, buff=0.5)
        self.play(Write(txt_loss_red),
                  Indicate(nodos_red[-1], color=ROJO_TOMATE, scale_factor=1.12))
        self._siguiente()

        lbl_bwd_red = Text("Gradientes fluyendo hacia atrás →",
                           font=FUENTE, font_size=19, weight=BOLD,
                           color=NARANJA_TERRACOTA).next_to(lbl_red, DOWN, buff=0.18)
        self.play(Write(lbl_bwd_red))

        for i in reversed(range(len(capas_config) - 1)):
            destellos = [ShowPassingFlash(l.copy(), time_width=0.45)
                         for l in conexiones_back_por_capa[i]]
            self.play(
                AnimationGroup(*destellos),
                Indicate(nodos_red[i], color=NARANJA_TERRACOTA, scale_factor=1.1),
                run_time=1.0
            )

        conclusion = Text(
            "124M params · un solo paso",
            font=FUENTE, font_size=20, weight=BOLD,
            color=NARANJA_TERRACOTA, line_spacing=1.3
        ).next_to(nodos_red, DOWN, buff=0.55).set_x(0)
        self.play(Write(conclusion))
        self.play(Indicate(conclusion, color=ORO_VIEJO, scale_factor=1.05))
        self._siguiente()

        self.limpiar_pantalla()


