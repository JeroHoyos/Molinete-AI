import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


class SlidesAtencion:
    def slide_mha_acto1_intuicion(self):
        titulo, linea = self.crear_titulo(
            "Atención: aquí está la clave del significado.",
            palabra_clave="Atención",
            color_clave=NARANJA_TERRACOTA
        )
        llanuras_fondo = crear_llanuras_manchegas()
        adornos = self._crear_adornos_esquinas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo, adornos=adornos)


        pregunta = Text(
            "¿Cómo sabemos el significado de una palabra?",
            font=FUENTE, font_size=30, color=TINTA_NEGRA
        ).next_to(linea, DOWN, buff=0.9)
        self.play(FadeIn(pregunta, shift=DOWN * 0.3))

        respuesta = Text(
            "Poniendo Atención a las palabras que la rodean.",
            font=FUENTE, font_size=36, weight=BOLD, color=NARANJA_TERRACOTA
        ).next_to(pregunta, DOWN, buff=0.55)
        self.play(Write(respuesta))


        ejemplos_data = [
            ("banco", "Me senté en el banco\ndel parque.",   NARANJA_TERRACOTA),
            ("banco", "Saqué dinero\ndel banco.",             MARRON_OSCURO),
            ("banco", "El banco de peces\npasó nadando.",     PAPEL_TAN),
        ]
        tarjetas = VGroup()
        for palabra, ctx, color in ejemplos_data:
            caja = RoundedRectangle(
                corner_radius=0.2, width=3.4, height=1.9,
                fill_color=PAPEL_CREMA, fill_opacity=1,
                stroke_color=color, stroke_width=2.5
            )
            pal = Text(
                palabra, font=FUENTE, font_size=26, weight=BOLD, color=color
            ).move_to(caja.get_center() + UP * 0.42)
            sep = Line(
                caja.get_center() + UP * 0.18 + LEFT * 1.4,
                caja.get_center() + UP * 0.18 + RIGHT * 1.4,
                stroke_color=color, stroke_width=1.2, stroke_opacity=0.4
            )
            ctx_txt = Text(
                ctx, font=FUENTE, font_size=15, color=TINTA_NEGRA, line_spacing=1.3
            ).move_to(caja.get_center() + DOWN * 0.32)
            tarjetas.add(VGroup(caja, pal, sep, ctx_txt))

        tarjetas.arrange(RIGHT, buff=0.5).next_to(respuesta, DOWN, buff=0.65)
        self.play(
            LaggedStart(*[FadeIn(t, shift=UP * 0.2, scale=0.95) for t in tarjetas],
                        lag_ratio=0.22),
            run_time=1.6
        )
        for t in tarjetas:
            self.play(Indicate(t[1], scale_factor=1.18, color=NARANJA_TERRACOTA), run_time=0.4)
        self._siguiente()

        self.play(FadeOut(VGroup(pregunta, respuesta, tarjetas)))


        palabras = ["El", "hidalgo", "vio", "al", "gigante,", "pero", "este", "era", "un", "molino."]
        oracion = VGroup(*[
            Text(p, font=FUENTE, font_size=30, color=TINTA_NEGRA) for p in palabras
        ]).arrange(RIGHT, buff=0.22)
        oracion.move_to(UP * 1.0)

        self.play(
            LaggedStart(*[FadeIn(w, shift=UP * 0.2) for w in oracion], lag_ratio=0.07),
            run_time=1.3
        )


        self.play(oracion[6].animate.set_color(NARANJA_TERRACOTA).scale(1.2), run_time=0.5)
        signo = Text("?", font=FUENTE, font_size=44, color=NARANJA_TERRACOTA, weight=BOLD
        ).next_to(oracion[6], UP, buff=0.1)
        self.play(FadeIn(signo, scale=0.4))
        otros_idx = [i for i in range(len(palabras)) if i != 6]
        self.play(
            AnimationGroup(*[oracion[i].animate.set_opacity(0.15) for i in otros_idx]),
            run_time=0.5
        )
        self._siguiente()
        self.play(
            AnimationGroup(*[oracion[i].animate.set_opacity(1.0) for i in otros_idx]),
            FadeOut(signo),
            run_time=0.4
        )


        Y_ARRIBA = oracion[4].get_top()[1] + 0.55
        p_gig_top  = np.array([oracion[4].get_center()[0], Y_ARRIBA, 0])
        p_este_top = np.array([oracion[6].get_center()[0], Y_ARRIBA, 0])
        seg_gig_sube   = Line(oracion[6].get_top(), p_este_top,
                              stroke_color=NARANJA_TERRACOTA, stroke_width=3)
        seg_horizontal = Line(p_este_top, p_gig_top,
                              stroke_color=NARANJA_TERRACOTA, stroke_width=3)
        seg_este_baja  = Arrow(p_gig_top, oracion[4].get_top(), buff=0,
                               color=NARANJA_TERRACOTA, stroke_width=3,
                               max_tip_length_to_length_ratio=0.25)
        flecha_gigante = VGroup(seg_gig_sube, seg_horizontal, seg_este_baja)
        peso_gig_lbl = Text(
            "0.85", font=FUENTE, font_size=20, color=NARANJA_TERRACOTA, weight=BOLD
        ).move_to([(oracion[4].get_center()[0] + oracion[6].get_center()[0]) / 2,
                   Y_ARRIBA + 0.28, 0])


        Y_ABAJO = oracion[1].get_bottom()[1] - 0.55
        p_hid_bot  = np.array([oracion[1].get_center()[0], Y_ABAJO, 0])
        p_este_bot = np.array([oracion[6].get_center()[0], Y_ABAJO, 0])
        seg_hid_baja   = Line(oracion[6].get_bottom(), p_este_bot,
                              stroke_color=MARRON_OSCURO, stroke_width=2)
        seg_horiz_bajo = Line(p_este_bot, p_hid_bot,
                              stroke_color=MARRON_OSCURO, stroke_width=2)
        seg_este_sube  = Arrow(p_hid_bot, oracion[1].get_bottom(), buff=0,
                               color=MARRON_OSCURO, stroke_width=2,
                               max_tip_length_to_length_ratio=0.25)
        flecha_hidalgo = VGroup(seg_hid_baja, seg_horiz_bajo, seg_este_sube)
        flecha_hidalgo.set_opacity(0.5)
        peso_hid_lbl = Text(
            "0.08", font=FUENTE, font_size=18, color=MARRON_OSCURO
        ).move_to([(oracion[1].get_center()[0] + oracion[6].get_center()[0]) / 2,
                   Y_ABAJO - 0.28, 0])
        peso_hid_lbl.set_opacity(0.5)


        self.play(
            Create(seg_hid_baja), Create(seg_horiz_bajo), Create(seg_este_sube),
            FadeIn(peso_hid_lbl),
            run_time=0.9
        )
        self.play(oracion[1].animate.set_color(MARRON_OSCURO).set_opacity(0.6))


        self.play(
            Create(seg_gig_sube), Create(seg_horizontal), Create(seg_este_baja),
            FadeIn(peso_gig_lbl),
            run_time=0.9
        )
        self.play(
            oracion[4].animate.set_color(NARANJA_TERRACOTA).scale(1.12),
            Flash(oracion[4], color=NARANJA_TERRACOTA, line_length=0.18, num_lines=9),
            run_time=0.8
        )


        self.play(FadeOut(VGroup(
            flecha_gigante, flecha_hidalgo,
            peso_gig_lbl, peso_hid_lbl
        )))
        self.play(oracion[4].animate.scale(1 / 1.12))

        ANCHO_MAX  = 4.8
        ALTO_BARRA = 0.36
        panel = RoundedRectangle(
            corner_radius=0.18, width=7.2, height=1.55,
            fill_color=PAPEL_CREMA, fill_opacity=1,
            stroke_color=MARRON_OSCURO, stroke_width=1.8
        ).next_to(oracion, DOWN, buff=0.55)

        X_ORIGEN = panel.get_left()[0] + 0.35
        Y_CENTRO  = panel.get_center()[1]

        barra_gig = Rectangle(
            width=ANCHO_MAX * 0.85, height=ALTO_BARRA,
            fill_color=NARANJA_TERRACOTA, fill_opacity=0.9, stroke_width=0
        )
        barra_gig.move_to([X_ORIGEN + (ANCHO_MAX * 0.85) / 2, Y_CENTRO + 0.30, 0])

        barra_hid = Rectangle(
            width=ANCHO_MAX * 0.08, height=ALTO_BARRA,
            fill_color=MARRON_OSCURO, fill_opacity=0.75, stroke_width=0
        )
        barra_hid.move_to([X_ORIGEN + (ANCHO_MAX * 0.08) / 2, Y_CENTRO - 0.30, 0])

        lbl_gig = Text(
            "gigante,  0.85", font=FUENTE, font_size=17,
            color=NARANJA_TERRACOTA, weight=BOLD
        ).next_to(barra_gig, RIGHT, buff=0.18)

        lbl_hid = Text(
            "hidalgo   0.08", font=FUENTE, font_size=17, color=MARRON_OSCURO
        ).next_to(barra_hid, RIGHT, buff=0.18)

        self.play(FadeIn(panel))
        self.play(GrowFromEdge(barra_gig, LEFT), FadeIn(lbl_gig, shift=LEFT * 0.15), run_time=0.75)
        self.play(GrowFromEdge(barra_hid, LEFT), FadeIn(lbl_hid, shift=LEFT * 0.15), run_time=0.5)


        pts_gig = VGroup(*[
            Dot(
                point=oracion[4].get_center() + np.array([
                    np.random.uniform(-0.25, 0.25),
                    np.random.uniform(-0.1, 0.1), 0
                ]),
                radius=0.058, color=NARANJA_TERRACOTA
            ) for _ in range(18)
        ])
        pts_hid = VGroup(*[
            Dot(
                point=oracion[1].get_center() + np.array([
                    np.random.uniform(-0.2, 0.2),
                    np.random.uniform(-0.08, 0.08), 0
                ]),
                radius=0.045, color=MARRON_OSCURO
            ) for _ in range(6)
        ])

        self.play(FadeIn(pts_gig, lag_ratio=0.04), FadeIn(pts_hid, lag_ratio=0.06), run_time=0.45)

        destino = oracion[6].get_center()
        self.play(
            LaggedStart(*[
                p.animate.move_to(destino + np.array([
                    np.random.uniform(-0.08, 0.08),
                    np.random.uniform(-0.08, 0.08), 0
                ]))
                for p in pts_gig
            ], lag_ratio=0.03),
            LaggedStart(*[
                p.animate.move_to(destino + np.array([
                    np.random.uniform(-0.06, 0.06),
                    np.random.uniform(-0.06, 0.06), 0
                ]))
                for p in pts_hid
            ], lag_ratio=0.05),
            oracion[6].animate.scale(1.12),
            run_time=1.5
        )
        self.play(
            FadeOut(pts_gig), FadeOut(pts_hid),
            oracion[6].animate.scale(1 / 1.12),
            run_time=0.4
        )


        nota = Text(
            '"este" absorbió principalmente el significado de "gigante,"',
            font_size=28, color=MARRON_OSCURO
        ).next_to(panel, DOWN, buff=0.65)
        self.play(FadeIn(nota, shift=UP * 0.2))

        self._siguiente()

        self.limpiar_pantalla()


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


    def slide_mha_acto3_formula_y_flujo(self):


        titulo, linea = self.crear_titulo(
            "Atención: Formula y diagrama de flujo.",
            palabra_clave="Atención",
            color_clave=NARANJA_TERRACOTA
        )
        llanuras_fondo = crear_llanuras_manchegas()
        adornos = self._crear_adornos_esquinas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo, adornos=adornos)


        formula = MathTex(
            r"\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V",
            color=TINTA_NEGRA, font_size=62
        ).move_to(ORIGIN)
        self.play(FadeIn(formula, shift=UP, scale=0.9))
        self._siguiente()
        self.play(FadeOut(formula), run_time=0.8)
        self.play(FadeOut(adornos), run_time=0.5)


        SW       = 2.2
        TIP      = 0.22
        C_FLECHA = MARRON_OSCURO

        def arr(p1, p2):
            return Arrow(
                p1, p2, buff=0.07,
                color=C_FLECHA,
                stroke_width=SW,
                max_tip_length_to_length_ratio=TIP
            )

        def seg(p1, p2):
            return Line(p1, p2, stroke_color=C_FLECHA, stroke_width=SW)

        def L_up_right(x0, y0, x1, y1):
            return VGroup(seg([x0, y0, 0], [x0, y1, 0]),
                          arr([x0, y1, 0], [x1, y1, 0]))


        COLOR_W     = PAPEL_TAN
        COLOR_W_STR = MARRON_OSCURO
        W_W, W_H    = 1.15, 0.65

        def caja_W(letra):
            r = RoundedRectangle(
                corner_radius=0.16, width=W_W, height=W_H,
                fill_color=COLOR_W, fill_opacity=1,
                stroke_color=COLOR_W_STR, stroke_width=2.2
            )
            lbl = MathTex(rf"W_{{{letra}}}", font_size=20, color=MARRON_OSCURO)
            lbl.move_to(r)
            return VGroup(r, lbl)


        def mat_bloques(filas, cols, bw, bh, color_fondo=FONDO_CAJA):
            buff_b = 0.04
            mat = VGroup()
            for i in range(filas):
                fila = VGroup()
                for j in range(cols):
                    b = RoundedRectangle(
                        corner_radius=0.07, width=bw, height=bh,
                        fill_color=color_fondo, fill_opacity=1,
                        stroke_color=MARRON_OSCURO, stroke_width=1.4
                    )
                    fila.add(b)
                fila.arrange(RIGHT, buff=buff_b)
                mat.add(fila)
            mat.arrange(DOWN, buff=buff_b)
            return mat

        def etiq_math(texto, obj, dir_, buff=0.08, fs=12, col=TINTA_NEGRA):
            lbl = MathTex(texto, font_size=fs, color=col)
            lbl.next_to(obj, dir_, buff=buff)
            return lbl


        cx_X   = -6.2
        cx_W   = -4.7
        cx_QKV = -2.9
        cx_QKT =  0.2
        cx_AP  =  2.9
        cx_Y   =  5.8

        yq    =  1.5
        yk    =  0.0
        yv    = -1.5
        y_mid =  0.75
        y_Y   =  0.10


        X_mat = mat_bloques(8, 2, 0.24, 0.30, FONDO_CAJA).move_to([cx_X, 0, 0])
        lbl_X = MathTex("X", font_size=22, color=TINTA_NEGRA
                ).next_to(X_mat, DOWN, buff=0.08)

        xbif     = X_mat.get_right()[0] + 0.15
        dot_bif  = Dot(radius=0.08, color=MARRON_OSCURO).move_to([xbif, 0, 0])
        ln_x_dot = seg(X_mat.get_right(), dot_bif.get_center())


        Wq = caja_W("Q").move_to([cx_W, yq, 0])
        Wk = caja_W("K").move_to([cx_W, yk, 0])
        Wv = caja_W("V").move_to([cx_W, yv, 0])


        Q_mat = mat_bloques(4, 6, 0.22, 0.24, FONDO_CAJA  ).move_to([cx_QKV, yq, 0])
        K_mat = mat_bloques(4, 6, 0.22, 0.24, CREMA_CALIDA).move_to([cx_QKV, yk, 0])
        V_mat = mat_bloques(4, 6, 0.22, 0.24, SALMON_CLARO).move_to([cx_QKV, yv, 0])

        eQ = etiq_math(r"Q = XW_Q", Q_mat, DOWN, fs=11)
        eK = etiq_math(r"K = XW_K", K_mat, DOWN, fs=11)
        eV = etiq_math(r"V = XW_V", V_mat, DOWN, fs=11)


        QKT = mat_bloques(6, 6, 0.24, 0.24, FONDO_CAJA).move_to([cx_QKT, y_mid, 0])
        lbl_QKT = etiq_math(r"QK^T", QKT, UP, fs=13)


        AP_n = 6
        AP_celdas = VGroup()
        for i in range(AP_n):
            fila = VGroup()
            for j in range(AP_n):
                mask = j > i
                b = RoundedRectangle(
                    corner_radius=0.07, width=0.28, height=0.28,
                    fill_color=MARRON_OSCURO if mask else FONDO_CAJA,
                    fill_opacity=0.75 if mask else 1.0,
                    stroke_color=MARRON_OSCURO,
                    stroke_width=1.1 if mask else 1.5
                )
                fila.add(b)
            fila.arrange(RIGHT, buff=0.04)
            AP_celdas.add(fila)
        AP_celdas.arrange(DOWN, buff=0.04)
        AP_celdas.move_to([cx_AP, y_mid, 0])


        formula_esquina = MathTex(
            r"\mathrm{Softmax}\!\left(\frac{QK^T}{\sqrt{d}}\right)",
            color=TINTA_NEGRA, font_size=15
        ).next_to(AP_celdas, UP, buff=0.10).align_to(AP_celdas, RIGHT)


        Y_mat    = mat_bloques(4, 6, 0.22, 0.24, FONDO_CAJA).move_to([cx_Y, y_Y, 0])

        lbl_Y_eq = MathTex(
            r"\mathrm{Softmax}\!\left(\frac{QK^T}{\sqrt{d}}\right)V",
            color=TINTA_NEGRA, font_size=13
        ).next_to(Y_mat, DOWN, buff=0.10)


        rt_q = L_up_right(xbif, 0, Wq.get_left()[0] - 0.06, yq)
        rt_k = arr(dot_bif.get_right(), Wk.get_left())
        rt_v = L_up_right(xbif, 0, Wv.get_left()[0] - 0.06, yv)


        a_Wq_Q = arr(Wq.get_right(), Q_mat.get_left())
        a_Wk_K = arr(Wk.get_right(), K_mat.get_left())
        a_Wv_V = arr(Wv.get_right(), V_mat.get_left())


        QK_group = VGroup(Q_mat, K_mat)
        brace_obj = Brace(
            QK_group, direction=RIGHT,
            color=MARRON_OSCURO,
            buff=0.10
        )

        brace_tip = brace_obj.get_tip()
        flecha_qkt = arr(brace_tip, QKT.get_left())

        brace_QK = VGroup(brace_obj, flecha_qkt)


        a_QKT_AP = arr(QKT.get_right(), AP_celdas.get_left())


        x_brace_APV  = AP_celdas.get_right()[0] + 0.22


        y_brace_top = AP_celdas.get_top()[1]
        y_brace_bot = yv

        brace_span = Line(
            [x_brace_APV, y_brace_top, 0],
            [x_brace_APV, y_brace_bot, 0]
        )
        brace_APV = Brace(brace_span, direction=RIGHT, color=MARRON_OSCURO, buff=0.08)


        tip_x        = brace_APV.get_tip()[0]
        tip_actual_y = brace_APV.get_tip()[1]


        flecha_Y  = arr([tip_x, y_Y, 0], Y_mat.get_left())


        bot_x     = brace_APV.get_bottom()[0]
        a_V_brace = Arrow(
            V_mat.get_right(), [bot_x, yv, 0],
            buff=0.07, color=C_FLECHA,
            stroke_width=SW,
            max_tip_length_to_length_ratio=0.04
        )


        self.play(
            FadeIn(X_mat), FadeIn(lbl_X),
            Create(ln_x_dot), FadeIn(dot_bif),
            run_time=0.8
        )
        self.play(
            LaggedStart(Create(rt_q), Create(rt_k), Create(rt_v), lag_ratio=0.15),
            LaggedStart(FadeIn(Wq), FadeIn(Wk), FadeIn(Wv), lag_ratio=0.15),
            run_time=1.0
        )


        self.play(
            LaggedStart(Create(a_Wq_Q), Create(a_Wk_K), Create(a_Wv_V), lag_ratio=0.12),
            LaggedStart(
                AnimationGroup(FadeIn(Q_mat), FadeIn(eQ)),
                AnimationGroup(FadeIn(K_mat), FadeIn(eK)),
                AnimationGroup(FadeIn(V_mat), FadeIn(eV)),
                lag_ratio=0.15
            ),
            run_time=1.3
        )
        self._siguiente()


        self.play(
            GrowFromCenter(brace_obj),
            run_time=0.7
        )
        self.play(
            Create(flecha_qkt),
            run_time=0.5
        )
        self.play(FadeIn(QKT), FadeIn(lbl_QKT), run_time=0.8)


        self.play(Create(a_QKT_AP), run_time=0.6)
        self.play(FadeIn(AP_celdas), run_time=0.9)
        self.play(FadeIn(formula_esquina, scale=0.9), run_time=0.5)
        self._siguiente()


        self.play(GrowFromCenter(brace_APV), run_time=0.7)
        self.play(
            Create(flecha_Y),
            Create(a_V_brace),
            run_time=0.6
        )
        self.play(
            FadeIn(Y_mat), FadeIn(lbl_Y_eq),
            run_time=0.8
        )
        self.play(
            Flash(Y_mat, color=NARANJA_TERRACOTA, line_length=0.22, num_lines=8),
            run_time=0.7
        )
        self._siguiente()

        self.limpiar_pantalla()

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