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


class SlideCapaTransformer:
    def slide_capa_transformer(self):

        escala = 0.65

        def crear_nodo(texto, ancho=2.5 * escala, alto=0.8 * escala, resaltado=False):
            bg = NARANJA_TERRACOTA if resaltado else PAPEL_CREMA
            borde = MARRON_OSCURO
            txt_color = BLANCO if resaltado else TINTA_NEGRA
            caja = RoundedRectangle(
                corner_radius=0.15 * escala, width=ancho, height=alto,
                fill_color=bg, fill_opacity=1, stroke_color=borde, stroke_width=2.5 * escala
            )
            txt = Text(texto, font=FUENTE, font_size=20 * escala, color=txt_color)
            return VGroup(caja, txt)

        titulo, linea = self.crear_titulo(
            "Arquitectura: Transformer Layer",
            palabra_clave="Transformer",
            color_clave=NARANJA_TERRACOTA,
        )
        llanuras_fondo = crear_llanuras_manchegas()
        adornos = self._crear_adornos_esquinas()
        self._animar_entrada_slide(
            titulo, linea,
            fondo=VGroup(llanuras_fondo),
            adornos=adornos,
        )


        X_RES  = -3.0
        X_BLOQ =  2.2
        G = 2.0 * escala


        Y_INPUT  =  2.8
        Y_BIF1   =  1.8
        Y_ADD1   =  0.9
        Y_LN1    =  1.8
        Y_ATTN   =  0.9
        Y_BIF2   = -0.1
        Y_ADD2   = -1.0
        Y_LN2    = -0.1
        Y_MLP    = -1.0
        Y_OUTPUT = -2.2


        nodo_input  = crear_nodo("Input") .move_to([X_RES, Y_INPUT,  0])
        nodo_output = crear_nodo("Output").move_to([X_RES, Y_OUTPUT, 0])

        add_1 = VGroup(
            Circle(radius=0.28*escala, fill_color=PAPEL_CREMA, fill_opacity=1,
                stroke_color=MARRON_OSCURO, stroke_width=2*escala),
            Text("+", font_size=24*escala, color=TINTA_NEGRA, weight=BOLD)
        ).move_to([X_RES, Y_ADD1, 0])

        add_2 = VGroup(
            Circle(radius=0.28*escala, fill_color=PAPEL_CREMA, fill_opacity=1,
                stroke_color=MARRON_OSCURO, stroke_width=2*escala),
            Text("+", font_size=24*escala, color=TINTA_NEGRA, weight=BOLD)
        ).move_to([X_RES, Y_ADD2, 0])

        nodo_ln1  = crear_nodo("Layer Norm")                  .move_to([X_BLOQ, Y_LN1,  0])
        nodo_attn = crear_nodo("Self-Attention", resaltado=True).move_to([X_BLOQ, Y_ATTN, 0])
        nodo_ln2  = crear_nodo("Layer Norm")                  .move_to([X_BLOQ, Y_LN2,  0])
        nodo_mlp  = crear_nodo("MLP", resaltado=True)         .move_to([X_BLOQ, Y_MLP,  0])


        seg_input_bif1 = Line(
            nodo_input.get_bottom(), [X_RES, Y_BIF1, 0],
            stroke_color=MARRON_OSCURO, stroke_width=G
        )

        f_bif1_add1 = Arrow(
            [X_RES, Y_BIF1, 0], add_1.get_top(), buff=0,
            color=MARRON_OSCURO, stroke_width=G, max_tip_length_to_length_ratio=0.15
        )

        seg_add1_bif2 = Line(
            add_1.get_bottom(), [X_RES, Y_BIF2, 0],
            stroke_color=MARRON_OSCURO, stroke_width=G
        )
        f_bif2_add2 = Arrow(
            [X_RES, Y_BIF2, 0], add_2.get_top(), buff=0,
            color=MARRON_OSCURO, stroke_width=G, max_tip_length_to_length_ratio=0.15
        )

        f_add2_out = Arrow(
            add_2.get_bottom(), nodo_output.get_top(), buff=0,
            color=MARRON_OSCURO, stroke_width=G, max_tip_length_to_length_ratio=0.15
        )


        f_bif1_ln1 = Arrow(
            [X_RES, Y_BIF1, 0], nodo_ln1.get_left(), buff=0,
            color=MARRON_OSCURO, stroke_width=G, max_tip_length_to_length_ratio=0.15
        )

        f_ln1_attn = Arrow(
            nodo_ln1.get_bottom(), nodo_attn.get_top(), buff=0,
            color=MARRON_OSCURO, stroke_width=G, max_tip_length_to_length_ratio=0.15
        )

        f_attn_add1 = Arrow(
            nodo_attn.get_left(), add_1.get_right(), buff=0,
            color=MARRON_OSCURO, stroke_width=G, max_tip_length_to_length_ratio=0.15
        )


        f_bif2_ln2 = Arrow(
            [X_RES, Y_BIF2, 0], nodo_ln2.get_left(), buff=0,
            color=MARRON_OSCURO, stroke_width=G, max_tip_length_to_length_ratio=0.15
        )
        f_ln2_mlp = Arrow(
            nodo_ln2.get_bottom(), nodo_mlp.get_top(), buff=0,
            color=MARRON_OSCURO, stroke_width=G, max_tip_length_to_length_ratio=0.15
        )
        f_mlp_add2 = Arrow(
            nodo_mlp.get_left(), add_2.get_right(), buff=0,
            color=MARRON_OSCURO, stroke_width=G, max_tip_length_to_length_ratio=0.15
        )


        diagrama = VGroup(
            nodo_input, nodo_output,
            add_1, add_2,
            nodo_ln1, nodo_attn, nodo_ln2, nodo_mlp,
            seg_input_bif1, f_bif1_add1,
            seg_add1_bif2, f_bif2_add2,
            f_add2_out,
            f_bif1_ln1, f_ln1_attn, f_attn_add1,
            f_bif2_ln2, f_ln2_mlp, f_mlp_add2,
        )
        diagrama.move_to(ORIGIN)

        self.play(FadeIn(diagrama))


        caja1 = SurroundingRectangle(
            VGroup(seg_input_bif1, f_bif1_add1, add_1,
                seg_add1_bif2, f_bif2_add2, add_2, f_add2_out),
            color=NARANJA_TERRACOTA, buff=0.2, stroke_width=3
        )
        self.play(Create(caja1))
        self.play(FadeOut(caja1))


        caja2 = VGroup(
            SurroundingRectangle(nodo_ln1, color=MARRON_OSCURO, buff=0.1, stroke_width=3),
            SurroundingRectangle(nodo_ln2, color=MARRON_OSCURO, buff=0.1, stroke_width=3),
        )
        self.play(Create(caja2))
        self.play(FadeOut(caja2))


        caja3 = VGroup(
            SurroundingRectangle(nodo_attn, color=NARANJA_TERRACOTA, buff=0.15, stroke_width=3),
            SurroundingRectangle(nodo_mlp,  color=NARANJA_TERRACOTA, buff=0.15, stroke_width=3),
        )
        self.play(Create(caja3))
        self.play(FadeOut(caja3))

        self._siguiente()
        self.limpiar_pantalla()


