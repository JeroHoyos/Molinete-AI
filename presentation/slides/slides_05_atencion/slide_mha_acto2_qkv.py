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


class SlideMhaActo2Qkv:
    def slide_mha_acto2_qkv(self):
        titulo, linea = self.crear_titulo(
            "Atención: Query, Key, Value.",
            palabra_clave="Atención",
            color_clave=NARANJA_TERRACOTA
        )
        llanuras_fondo = crear_llanuras_manchegas()
        adornos = self._crear_adornos_esquinas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo, adornos=adornos)


        palabras = ["El", "hidalgo", "vio", "al", "gigante,", "pero", "este", "era", "un", "molino."]
        oracion = VGroup(*[
            Text(p, font=FUENTE, font_size=26, color=TINTA_NEGRA) for p in palabras
        ]).arrange(RIGHT, buff=0.18)
        oracion.next_to(linea, DOWN, buff=0.55)

        self.play(
            LaggedStart(*[FadeIn(w, shift=UP * 0.15) for w in oracion], lag_ratio=0.06),
            run_time=1.2
        )


        otros_idx = [i for i in range(len(palabras)) if i != 6]
        self.play(
            oracion[6].animate.set_color(NARANJA_TERRACOTA).scale(1.15),
            AnimationGroup(*[oracion[i].animate.set_opacity(0.18) for i in otros_idx]),
            run_time=0.55
        )

        EMB_Y = 0.3
        emb_caja = RoundedRectangle(
            corner_radius=0.18, width=2.8, height=0.68,
            fill_color=PAPEL_CREMA, fill_opacity=1,
            stroke_color=NARANJA_TERRACOTA, stroke_width=2.2
        ).move_to([0.0, EMB_Y, 0.0])
        emb_lbl = Text(
            "embedding", font=FUENTE, font_size=17, color=NARANJA_TERRACOTA, weight=BOLD
        ).move_to([0.0, EMB_Y, 0.0])
        emb_vec = Text(
            "[0.82, −0.31, 0.57, …]", font=FUENTE, font_size=12, color=MARRON_OSCURO
        ).next_to(emb_caja, DOWN, buff=0.12)

        flecha_emb = Arrow(
            oracion[6].get_bottom(), emb_caja.get_top(),
            buff=0.08, color=NARANJA_TERRACOTA, stroke_width=2.5,
            max_tip_length_to_length_ratio=0.22
        )
        self.play(Create(flecha_emb), FadeIn(emb_caja, emb_lbl), run_time=0.7)
        self.play(FadeIn(emb_vec, shift=DOWN * 0.1))


        ANCHO_TQK  = 2.6
        ALTO_TQK   = 2.1
        X_Q        = -2.2
        X_K        =  2.2
        Y_CAJAS_QK = -1.7

        def hacer_tarjeta_qk(letra, nombre, desc, color, x, y):
            caja = RoundedRectangle(
                corner_radius=0.18, width=ANCHO_TQK, height=ALTO_TQK,
                fill_color=PAPEL_CREMA, fill_opacity=1,
                stroke_color=color, stroke_width=2.5
            )
            header = RoundedRectangle(
                corner_radius=0.14, width=ANCHO_TQK, height=0.54,
                fill_color=color, fill_opacity=1, stroke_width=0
            ).align_to(caja, UP)

            tapa = Rectangle(
                width=ANCHO_TQK, height=0.2,
                fill_color=color, fill_opacity=1, stroke_width=0
            ).next_to(header, DOWN, buff=0)

            letra_txt = Text(letra, font=FUENTE, font_size=28, weight=BOLD, color=PAPEL_CREMA
            ).move_to(header.get_center())

            nombre_txt = Text(nombre, font=FUENTE, font_size=13, weight=BOLD, color=color
            ).next_to(tapa, DOWN, buff=0.18)

            desc_txt = Text(desc, font=FUENTE, font_size=12, color=TINTA_NEGRA, line_spacing=1.3
            ).next_to(nombre_txt, DOWN, buff=0.14)

            tarjeta = VGroup(caja, header, tapa, letra_txt, nombre_txt, desc_txt)
            tarjeta.move_to([x, y, 0.0])
            return tarjeta

        grupo_q = hacer_tarjeta_qk(
            "Q", "Query — ¿Qué busco?",
            "Lo que \"este\" necesita\nsaber del contexto.",
            NARANJA_TERRACOTA, X_Q, Y_CAJAS_QK
        )
        grupo_k = hacer_tarjeta_qk(
            "K", "Key — ¿Qué ofrezco?",
            "La 'etiqueta' que permite\na otras palabras encontrarme.",
            MARRON_OSCURO, X_K, Y_CAJAS_QK
        )

        EMB_BOT_Y = EMB_Y - 0.34
        TOP_QK_Y  = Y_CAJAS_QK + ALTO_TQK / 2

        flecha_a_q = Arrow(
            [0.0, EMB_BOT_Y, 0.0], [X_Q, TOP_QK_Y, 0.0],
            path_arc=0.6,
            buff=0.1, color=NARANJA_TERRACOTA, stroke_width=3,
            max_tip_length_to_length_ratio=0.15
        )
        flecha_a_k = Arrow(
            [0.0, EMB_BOT_Y, 0.0], [X_K, TOP_QK_Y, 0.0],
            path_arc=-0.6,
            buff=0.1, color=MARRON_OSCURO, stroke_width=3,
            max_tip_length_to_length_ratio=0.15
        )

        self.play(FadeOut(emb_vec), run_time=0.4)
        self.play(
            Create(flecha_a_q),
            FadeIn(grupo_q, shift=DOWN * 0.2, scale=0.95),
            run_time=0.9
        )

        self.play(
            Create(flecha_a_k),
            FadeIn(grupo_k, shift=DOWN * 0.2, scale=0.95),
            run_time=0.9
        )
        self._siguiente()


        Y_EMB_P3   =  1.6
        Y_TITULO   =  2.08
        Y_PERILLAS =  0.2
        Y_TRONCO   =  Y_EMB_P3 - 0.34 - 0.30
        Y_CAJAS_P3 = -1.5
        X_P        = [-3.8, 0.0, 3.8]

        self.play(
            FadeOut(VGroup(
                oracion, flecha_emb,
                flecha_a_q, grupo_q,
                flecha_a_k, grupo_k,
            )),
            emb_caja.animate.move_to([0.0, Y_EMB_P3, 0.0]),
            emb_lbl.animate.move_to([0.0, Y_EMB_P3, 0.0]),
            run_time=0.85
        )

        emb_titulo = Text(
            "embedding de \"este\"", font=FUENTE, font_size=16, color=NARANJA_TERRACOTA
        ).move_to([0.0, Y_TITULO, 0.0])
        self.play(FadeIn(emb_titulo))

        nombres_w = [("W_Q", NARANJA_TERRACOTA), ("W_K", MARRON_OSCURO), ("W_V", PAPEL_TAN)]
        perillas = VGroup()
        for i, (nombre, color) in enumerate(nombres_w):
            circ = Circle(
                radius=0.38, fill_color=PAPEL_CREMA, fill_opacity=1,
                stroke_color=color, stroke_width=2.5
            ).move_to([X_P[i], Y_PERILLAS, 0.0])
            etiq = Text(nombre, font=FUENTE, font_size=15, weight=BOLD, color=color
            ).move_to([X_P[i], Y_PERILLAS, 0.0])
            perillas.add(VGroup(circ, etiq))

        flechas_w = VGroup()
        for i, (_, color) in enumerate(nombres_w):
            seg_v = Line(
                [0.0, Y_EMB_P3 - 0.34, 0.0], [0.0, Y_TRONCO, 0.0],
                stroke_color=color, stroke_width=2
            )
            seg_h = Line(
                [0.0, Y_TRONCO, 0.0], [X_P[i], Y_TRONCO, 0.0],
                stroke_color=color, stroke_width=2
            )
            flecha = Arrow(
                [X_P[i], Y_TRONCO, 0.0], [X_P[i], Y_PERILLAS + 0.38, 0.0],
                buff=0.0, color=color, stroke_width=2,
                max_tip_length_to_length_ratio=0.22
            )
            flechas_w.add(VGroup(seg_v, seg_h, flecha))

        self.play(
            LaggedStart(*[
                AnimationGroup(Create(flechas_w[i]), FadeIn(perillas[i], scale=0.9))
                for i in range(3)
            ], lag_ratio=0.28),
            run_time=1.5
        )

        etiquetas_salida = [
            ("Q", "¿Qué busco?",   NARANJA_TERRACOTA),
            ("K", "¿Qué ofrezco?", MARRON_OSCURO),
            ("V", "¿Qué aporto?",  PAPEL_TAN),
        ]
        cajas_salida   = VGroup()
        flechas_salida = VGroup()

        for i, (letra, desc, color) in enumerate(etiquetas_salida):
            caja = RoundedRectangle(
                corner_radius=0.15, width=2.6, height=1.0,
                fill_color=PAPEL_CREMA, fill_opacity=1,
                stroke_color=color, stroke_width=2.2
            ).move_to([X_P[i], Y_CAJAS_P3, 0.0])
            l_letra = Text(letra, font=FUENTE, font_size=26, weight=BOLD, color=color
            ).move_to([X_P[i] - 0.58, Y_CAJAS_P3, 0.0])
            sep = Line(
                [X_P[i] - 0.20, Y_CAJAS_P3 + 0.26, 0.0],
                [X_P[i] - 0.20, Y_CAJAS_P3 - 0.26, 0.0],
                stroke_color=color, stroke_width=1.2, stroke_opacity=0.4
            )
            l_desc = Text(desc, font=FUENTE, font_size=12, color=TINTA_NEGRA, line_spacing=1.2
            ).move_to([X_P[i] + 0.52, Y_CAJAS_P3, 0.0])
            cajas_salida.add(VGroup(caja, l_letra, sep, l_desc))

            flechas_salida.add(Arrow(
                [X_P[i], Y_PERILLAS - 0.38, 0.0],
                [X_P[i], Y_CAJAS_P3 + 0.50, 0.0],
                buff=0.0, color=color, stroke_width=2,
                max_tip_length_to_length_ratio=0.22
            ))

        self.play(
            LaggedStart(*[
                AnimationGroup(
                    Create(flechas_salida[i]),
                    FadeIn(cajas_salida[i], shift=DOWN * 0.15)
                )
                for i in range(3)
            ], lag_ratio=0.28),
            run_time=1.5
        )
        self._siguiente()

        self.limpiar_pantalla()


