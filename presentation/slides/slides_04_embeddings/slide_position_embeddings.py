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


class SlidePositionEmbeddings:
    def slide_position_embeddings(self) -> None:
        t1 = Text("Embeddings de ", font=FUENTE, font_size=42, weight=BOLD, color=TINTA_NEGRA)
        t2 = Text("Posición", font=FUENTE, font_size=42, weight=BOLD, color=NARANJA_TERRACOTA)
        titulo = VGroup(t1, t2).arrange(RIGHT, buff=0.08).to_edge(UP, buff=0.5)
        linea  = Line(LEFT * 6, RIGHT * 6, color=MARRON_OSCURO, stroke_width=2
                      ).next_to(titulo, DOWN, buff=0.15)

        llanuras = crear_llanuras_manchegas()
        adornos  = self._crear_adornos_esquinas(escala=0.55, buff=0.5)
        adornos[1].add_updater(lambda m, dt: m.rotate(dt * 0.4))
        self._animar_entrada_slide(titulo, linea, fondo=llanuras, adornos=adornos)

        self._acto_lectura_simultanea(linea)
        self._acto_suma_posicion(linea)
        self._acto_ventana_contexto(linea)

        adornos[1].clear_updaters()
        self._siguiente()
        self.limpiar_pantalla()


    @staticmethod
    def _fila_tokens(palabras: list) -> VGroup:
        cajas = VGroup()
        for w in palabras:
            rect = RoundedRectangle(corner_radius=0.1, width=1.5, height=0.6,
                                    fill_color=PAPEL_CREMA, fill_opacity=0.9,
                                    stroke_color=MARRON_OSCURO, stroke_width=2)
            txt = Text(w, font=FUENTE, font_size=22, color=TINTA_NEGRA
                       ).move_to(rect.get_center())
            cajas.add(VGroup(rect, txt))
        return cajas.arrange(RIGHT, buff=0.25)


    def _acto_lectura_simultanea(self, linea: Mobject) -> None:


        frase_a = self._fila_tokens(["El", "perro", "muerde", "al", "hombre"])
        frase_b = self._fila_tokens(["El", "hombre", "muerde", "al", "perro"])
        lbl_a = Text("Frase A:", font=FUENTE, font_size=19, weight=BOLD, color=MARRON_OSCURO)
        lbl_b = Text("Frase B:", font=FUENTE, font_size=19, weight=BOLD, color=MARRON_OSCURO)
        grp_a = VGroup(lbl_a, frase_a).arrange(RIGHT, buff=0.3)
        grp_b = VGroup(lbl_b, frase_b).arrange(RIGHT, buff=0.3)
        frases = VGroup(grp_a, grp_b).arrange(DOWN, buff=0.65).move_to(ORIGIN).shift(UP * 0.3)

        self.play(FadeIn(grp_a, shift=RIGHT * 0.25))
        self.play(FadeIn(grp_b, shift=RIGHT * 0.25))


        self.play(
            frase_a[1][0].animate.set_fill(ROJO_TOMATE, opacity=0.85),
            frase_a[1][1].animate.set_color(BLANCO),
            frase_b[4][0].animate.set_fill(ROJO_TOMATE, opacity=0.85),
            frase_b[4][1].animate.set_color(BLANCO),
            frase_a[4][0].animate.set_fill(VERDE_OLIVA, opacity=0.85),
            frase_a[4][1].animate.set_color(BLANCO),
            frase_b[1][0].animate.set_fill(VERDE_OLIVA, opacity=0.85),
            frase_b[1][1].animate.set_color(BLANCO),
        )


        bag_label = Text("Sin posición, el modelo trata la secuencia como un conjunto sin estructura",
                         font=FUENTE, font_size=21, color=NARANJA_TERRACOTA, weight=BOLD
                         ).next_to(frases, DOWN, buff=0.55)
        self.play(Write(bag_label))

        self._siguiente()
        self.play(FadeOut(grp_a, grp_b, bag_label))

    def _acto_suma_posicion(self, linea: Mobject) -> None:

        def _mini_vec(valores, color_fill, color_stroke=None):
            cs = color_stroke or color_fill
            celdas = VGroup()
            for val in valores:
                es_p = (val == "·")
                rect = RoundedRectangle(
                    corner_radius=0.05,
                    width=0.66,
                    height=0.5,
                    fill_color=color_fill if not es_p else FONDO_CAJA,
                    fill_opacity=0.85 if not es_p else 0
                )
                rect.set_stroke(cs if not es_p else FONDO_CAJA, 1.5)
                txt = Text(val, font="Monospace", font_size=15, color=TINTA_NEGRA).move_to(rect)
                celdas.add(VGroup(rect, txt))
            return celdas.arrange(RIGHT, buff=0.08)

        pos_data = [
            (0, ["0.1", "-0.3", "0.5", "·"], ["0.8", "-0.5", "0.1", "·"], ["0.9", "-0.8", "0.9", "·"]),
            (1, ["0.4", "0.2", "-0.1", "·"], ["0.3", "0.7", "-0.4", "·"], ["0.7", "0.9", "-0.4", "·"]),
            (2, ["-0.8", "0.6", "0.1", "·"], ["0.1", "-0.2", "0.9", "·"], ["0.4", "0.3", "1.0", "·"]),
        ]
        palabras = ["perro", "muerde", "hombre"]

        cols = VGroup()
        for pos_num, tok_vals, pos_vals, sum_vals in pos_data:
            palabra_lbl = Text(palabras[pos_num], font=FUENTE, font_size=22,
                            weight=BOLD, color=TINTA_NEGRA)
            pos_lbl_txt = Text(f"pos {pos_num}", font=FUENTE, font_size=16, color=ACERO)

            v_tok = _mini_vec(tok_vals, PAPEL_TAN)
            v_pos = _mini_vec(pos_vals, NARANJA_TERRACOTA)
            v_sum = _mini_vec(sum_vals, CAJA_INFERIOR)

            lbl_tok = Text("token", font=FUENTE, font_size=13, color=MARRON_OSCURO)
            lbl_pos = Text("posición", font=FUENTE, font_size=13, color=NARANJA_TERRACOTA)
            lbl_sum = Text("final", font=FUENTE, font_size=13, color=TINTA_NEGRA, weight=BOLD)

            mas = Text("+", font=FUENTE, font_size=28, weight=BOLD, color=TINTA_NEGRA)
            sep = Line(LEFT * v_pos.width / 2, RIGHT * v_pos.width / 2,
                    color=MARRON_OSCURO, stroke_width=1.8)

            col_content = VGroup(
                VGroup(lbl_tok, v_tok).arrange(DOWN, buff=0.06),
                mas,
                VGroup(lbl_pos, v_pos).arrange(DOWN, buff=0.06),
                sep,
                VGroup(lbl_sum, v_sum).arrange(DOWN, buff=0.06),
            ).arrange(DOWN, buff=0.18, aligned_edge=LEFT)

            encabezado = VGroup(palabra_lbl, pos_lbl_txt).arrange(DOWN, buff=0.08)
            encabezado.next_to(col_content, UP, buff=0.25)

            caja = RoundedRectangle(
                corner_radius=0.14,
                width=col_content.width + 0.6,
                height=col_content.height + encabezado.height + 0.7,
                fill_color=FONDO_CAJA,
                fill_opacity=1,
                stroke_color=MARRON_OSCURO,
                stroke_width=1.8,
            )

            col_group = VGroup(caja, encabezado, col_content)
            col_content.move_to(caja.get_center() + DOWN * 0.25)
            encabezado.next_to(col_content, UP, buff=0.2)

            cols.add(col_group)

        cols.arrange(RIGHT, buff=0.5).next_to(linea, DOWN, buff=0.9)

        if cols.width > 13.0:
            cols.scale(13.0 / cols.width)

        self.play(
            LaggedStart(*[FadeIn(c, scale=0.92) for c in cols], lag_ratio=0.35),
            run_time=1.2
        )

        puntos_sum = [cols[i][2][-1] for i in range(3)]
        self.play(
            LaggedStart(
                *[Indicate(p, scale_factor=1.12, color=ORO_VIEJO) for p in puntos_sum],
                lag_ratio=0.3,
            ),
            run_time=1.0
        )

        mensaje = Text(
            "Cada token lleva: ¿qué palabra?  +  ¿en qué posición?",
            font=FUENTE,
            font_size=21,
            color=TINTA_NEGRA
        ).next_to(cols, DOWN, buff=0.4)

        self.play(Write(mensaje))

        self._siguiente()
        self.play(FadeOut(cols, mensaje))

    def _acto_ventana_contexto(self, linea: Mobject) -> None:

        titulo = Text(
            "Ventana de contexto",
            font=FUENTE, font_size=26, weight=BOLD, color=TINTA_NEGRA,
        ).next_to(linea, DOWN, buff=0.45)
        self.play(Write(titulo))

        def _celda_pos(num, color):
            celda = RoundedRectangle(
                corner_radius=0.06,
                width=1.2,
                height=0.55,
                fill_color=color,
                fill_opacity=0.7,
            ).set_stroke(MARRON_OSCURO, 1.5)
            etiq = Text(f"{num}", font=FUENTE, font_size=16, color=TINTA_NEGRA)
            etiq.move_to(celda)
            return VGroup(celda, etiq)

        c0    = _celda_pos(0,    PAPEL_TAN)
        c1    = _celda_pos(1,    NARANJA_TERRACOTA)
        c2    = _celda_pos(2,    PAPEL_TAN)
        dots  = Text("···", font_size=30, color=MARRON_OSCURO)
        c_last = _celda_pos(1023, CAJA_INFERIOR)

        tabla = VGroup(c0, c1, c2, dots, c_last).arrange(RIGHT, buff=0.25)

        brace = Brace(tabla, DOWN, color=MARRON_OSCURO)
        lbl = Text(
            "1024 posiciones",
            font=FUENTE,
            font_size=18,
            color=MARRON_OSCURO,
        ).next_to(brace, DOWN, buff=0.2)

        grupo_tabla = VGroup(tabla, brace, lbl)
        grupo_tabla.next_to(titulo, DOWN, buff=1.6).set_x(0).shift(LEFT * 0.8)

        self.play(FadeIn(tabla, shift=UP * 0.2))
        self.play(GrowFromCenter(brace), Write(lbl))


        token_fuera = RoundedRectangle(
            corner_radius=0.06,
            width=1.2,
            height=0.55,
            fill_color=NARANJA_TERRACOTA,
            fill_opacity=0.8,
        ).set_stroke(MARRON_OSCURO, 1.5)

        token_lbl = Text("1024", font=FUENTE, font_size=16, color=TINTA_NEGRA)
        token_lbl.move_to(token_fuera)
        token_group = VGroup(token_fuera, token_lbl)
        token_group.next_to(tabla, RIGHT, buff=0.25)

        cruz = Text("✕", font_size=40, color=NARANJA_TERRACOTA).next_to(token_group, UP, buff=0.15)

        self.play(FadeIn(token_group, shift=RIGHT * 0.3))
        self.play(Write(cruz))

        self.play(Indicate(c_last, color=ORO_VIEJO))