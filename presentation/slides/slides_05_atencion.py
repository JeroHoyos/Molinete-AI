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

        # ══════════════════════════════════════════════════════════════════════════
        # PARTE 1 — ¿Cómo sabemos el significado de una palabra?
        # ══════════════════════════════════════════════════════════════════════════
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

        # Tres tarjetas con "banco"
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

        # ══════════════════════════════════════════════════════════════════════════
        # PARTE 2 — Análisis de la frase, bien centrada
        # ══════════════════════════════════════════════════════════════════════════
        palabras = ["El", "hidalgo", "vio", "al", "gigante,", "pero", "este", "era", "un", "molino."]
        oracion = VGroup(*[
            Text(p, font=FUENTE, font_size=30, color=TINTA_NEGRA) for p in palabras
        ]).arrange(RIGHT, buff=0.22)
        oracion.move_to(UP * 1.0)

        self.play(
            LaggedStart(*[FadeIn(w, shift=UP * 0.2) for w in oracion], lag_ratio=0.07),
            run_time=1.3
        )

        # Señalar "este"
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

        # ── Flecha ARRIBA hacia gigante (codo recto por arriba) ──────────────────
        # Sale de "este", va horizontal, baja a "gigante"
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

        # ── Flecha ABAJO hacia hidalgo (codo recto por abajo) ────────────────────
        # Sale de "este", va horizontal, sube a "hidalgo"
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

        # Mostrar hidalgo primero (por abajo, secundario)
        self.play(
            Create(seg_hid_baja), Create(seg_horiz_bajo), Create(seg_este_sube),
            FadeIn(peso_hid_lbl),
            run_time=0.9
        )
        self.play(oracion[1].animate.set_color(MARRON_OSCURO).set_opacity(0.6))

        # Luego gigante (por arriba, principal)
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


        # ── Barras comparativas ───────────────────────────────────────────────────
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


        # ── Absorción: partículas de AMBAS fuentes → "este" ──────────────────────
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

        # Nota final
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

        # ══════════════════════════════════════════════════════════════════════════
        # PARTE 1 — El embedding de "este"
        # ══════════════════════════════════════════════════════════════════════════
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


        # ══════════════════════════════════════════════════════════════════════════
        # PARTE 2 — Bifurcación: Q (izquierda) y K (derecha)
        # ══════════════════════════════════════════════════════════════════════════
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

        # ══════════════════════════════════════════════════════════════════════════
        # PARTE 3 — W_Q, W_K, W_V ponderan el embedding → Q, K, V
        # ══════════════════════════════════════════════════════════════════════════
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
        """Fórmula general limpia y luego el diagrama de flujo original más espaciado."""
        # ── 1. TÍTULO Y FONDO ──────────────────────────────────────────────────────
        titulo, linea = self.crear_titulo(
            "Atención: Formula y diagrama de flujo.",
            palabra_clave="Atención",
            color_clave=NARANJA_TERRACOTA
        )
        llanuras_fondo = crear_llanuras_manchegas()
        adornos = self._crear_adornos_esquinas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo, adornos=adornos)

        # ── 2. LA FÓRMULA (Limpia y centrada) ──────────────────────────────────────
        formula = MathTex(
            r"\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V",
            color=TINTA_NEGRA, font_size=62
        ).move_to(ORIGIN)

        self.play(FadeIn(formula, shift=UP, scale=0.9))
        self._siguiente()

        # ── 3. TRANSICIÓN ──────────────────────────────────────────────────────────

        self.play(
            FadeOut(formula),
            run_time=1.0
        )

        # ── 4. DIAGRAMA DE FLUJO (Estilo Manhattan original, más espaciado) ───────
        G = 2.5   # grosor flechas

        def caja(texto, color_bg, ancho=1.4, alto=0.75, math=False):
            r = RoundedRectangle(
                corner_radius=0.12, width=ancho, height=alto,
                fill_color=color_bg, fill_opacity=1,
                stroke_color=MARRON_OSCURO, stroke_width=1.8
            )
            lbl = (MathTex(texto, color=TINTA_NEGRA, font_size=26)
                if math else
                Text(texto, font=FUENTE, font_size=18, color=TINTA_NEGRA))
            lbl.move_to(r.get_center())
            return VGroup(r, lbl)

        def flecha(origen, destino):
            return Arrow(origen, destino, buff=0.06,
                        color=MARRON_OSCURO, stroke_width=G,
                        max_tip_length_to_length_ratio=0.18)

        def linea_bifurc(p1, p2):
            return Line(p1, p2, stroke_color=MARRON_OSCURO, stroke_width=G)

        # ── POSICIONES (AQUÍ ESTÁ LA MAGIA: MUCHO MÁS SEPARADOS)
        # Originales: [-5.8, -4.2, -2.8, -1.4, -0.1, 1.2, 2.6, 4.1]
        # Nuevas: Expandidas para estirar las flechas
        xs = [-6.2, -4.6, -3.0, -1.4, 0.4, 2.4, 4.4, 5.8]
        yq, yk, yv = 1.1, 0.0, -1.1

        # Nodo X
        X_lbl = MathTex("X", color=TINTA_NEGRA, font_size=38).move_to([xs[0], 0, 0])
        dot_x  = Dot(radius=0.07, color=MARRON_OSCURO).next_to(X_lbl, RIGHT, buff=0.12)

        # Matrices W
        Wq = caja(r"W^Q", LAVANDA, math=True).move_to([xs[1], yq, 0])
        Wk = caja(r"W^K", LAVANDA, math=True).move_to([xs[1], yk, 0])
        Wv = caja(r"W^V", LAVANDA, math=True).move_to([xs[1], yv, 0])

        # Vectores Q K V
        Q_lbl = MathTex("Q", color=TINTA_NEGRA, font_size=32).move_to([xs[2], yq, 0])
        K_lbl = MathTex("K", color=TINTA_NEGRA, font_size=32).move_to([xs[2], yk, 0])
        V_lbl = MathTex("V", color=TINTA_NEGRA, font_size=32).move_to([xs[2], yv, 0])

        # Bloques centrales
        ymid = (yq + yk) / 2
        mm1   = caja("mat\nmul", NARANJA_CLARO, ancho=1.1, alto=1.6).move_to([xs[3], ymid, 0])
        sc    = caja("scale",   AMARILLO_PALIDO, ancho=1.1, alto=1.6).move_to([xs[4], ymid, 0])
        sm    = caja("softmax",  MENTA_PALIDA, ancho=1.2, alto=1.6).move_to([xs[5], ymid, 0])
        mm2   = caja("mat\nmul", NARANJA_CLARO, ancho=1.1, alto=2.6).move_to([xs[6], 0,   0])

        Y_lbl = MathTex("Y", color=TINTA_NEGRA, font_size=38).move_to([xs[7], 0, 0])

        # ── CONEXIONES
        ln_x_dot = linea_bifurc(X_lbl.get_right(), dot_x.get_center())

        def ruta_bifurc(from_pt, to_mobj):
            mid = [from_pt[0], to_mobj.get_y(), 0]
            return VGroup(
                linea_bifurc(from_pt, mid),
                flecha(mid, to_mobj.get_left())
            )

        rt_q = ruta_bifurc(dot_x.get_center(), Wq)
        rt_k = flecha(dot_x.get_right(), Wk.get_left())
        rt_v = ruta_bifurc(dot_x.get_center(), Wv)

        a_wq = flecha(Wq.get_right(), Q_lbl.get_left())
        a_wk = flecha(Wk.get_right(), K_lbl.get_left())
        a_wv = flecha(Wv.get_right(), V_lbl.get_left())

        a_q_mm1 = flecha([Q_lbl.get_right()[0], Q_lbl.get_y(), 0],
                        [mm1.get_left()[0],     Q_lbl.get_y(), 0])
        a_k_mm1 = flecha([K_lbl.get_right()[0], K_lbl.get_y(), 0],
                        [mm1.get_left()[0],     K_lbl.get_y(), 0])

        a_mm1_sc = flecha(mm1.get_right(), sc.get_left())
        a_sc_sm  = flecha(sc.get_right(),  sm.get_left())

        a_sm_mm2 = flecha([sm.get_right()[0],    sm.get_y(), 0],
                        [mm2.get_left()[0],     sm.get_y(), 0])
        a_v_mm2  = flecha([V_lbl.get_right()[0], yv, 0],
                        [mm2.get_left()[0],     yv, 0])

        a_mm2_y = flecha(mm2.get_right(), Y_lbl.get_left())

        # ── ANIMACIÓN DEL FLUJO
        # Etapa 1: X → Q, K, V
        self.play(
            FadeIn(X_lbl), Create(ln_x_dot), FadeIn(dot_x),
            run_time=0.6
        )
        self.play(
            LaggedStart(Create(rt_q), Create(rt_k), Create(rt_v), lag_ratio=0.15),
            LaggedStart(FadeIn(Wq), FadeIn(Wk), FadeIn(Wv), lag_ratio=0.15),
            run_time=1.0
        )
        self.play(
            LaggedStart(Create(a_wq), Create(a_wk), Create(a_wv), lag_ratio=0.1),
            LaggedStart(FadeIn(Q_lbl), FadeIn(K_lbl), FadeIn(V_lbl), lag_ratio=0.1),
            run_time=0.9
        )

        # Etapa 2: matmul QKt
        self.play(Create(a_q_mm1), Create(a_k_mm1), FadeIn(mm1), run_time=0.9)

        # Etapa 3: scale + softmax
        self.play(Create(a_mm1_sc), FadeIn(sc), Create(a_sc_sm), FadeIn(sm), run_time=1.0)

        # Etapa 4: matmul V → Y
        self.play(Create(a_sm_mm2), Create(a_v_mm2), FadeIn(mm2), run_time=0.9)
        self.play(Create(a_mm2_y), FadeIn(Y_lbl, scale=1.2), run_time=0.7)
        self.play(Flash(Y_lbl, color=NARANJA_TERRACOTA, line_length=0.25, num_lines=8))
        self._siguiente()

        self.limpiar_pantalla()


    def slide_mha_acto4_multihead(self):
        """¿Por qué múltiples cabezas? Animación centrada demostrando la concatenación y mezcla, ajustada hacia arriba."""
        # ── 1. TÍTULO Y FONDO ──────────────────────────────────────────────────────
        titulo, linea = self.crear_titulo(
            "Multi-Head Self-Attention",
            palabra_clave="Multi-Head",
            color_clave=NARANJA_TERRACOTA
        )
        llanuras_fondo = crear_llanuras_manchegas()
        adornos = self._crear_adornos_esquinas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo, adornos=adornos)

        # ── 2. SUBTÍTULO (Subimos los elementos reduciendo el buff) ────────────────
        subtitulo = Text(
            "¿Por qué multi-head?",
            font=FUENTE, font_size=28, color=MARRON_OSCURO
        ).next_to(linea, DOWN, buff=0.4) 
        
        self.play(FadeIn(subtitulo, shift=DOWN))

        # ── 3. EL VECTOR ORIGINAL (768 dims) ───────────────────────────────────────
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

        # ── 4. DIVISIÓN EN CABEZAS ─────────────────────────────────────────────────
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

        # ── 5. EJEMPLOS LIMPIOS (Solo indicando Q, K, V distintos) ─────────────────
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

        # ── 6. CONCATENACIÓN Y MEZCLA LINEAL ───────────────────────────────────────
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

