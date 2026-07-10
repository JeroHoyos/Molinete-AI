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


class SlideMhaActo1Intuicion:
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


