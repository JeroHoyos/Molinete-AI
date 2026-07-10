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


class SlideBatchedMatmul:
    def slide_batched_matmul(self):
        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "Batched MatMul: Una Llamada Para Ejecutarlos A Todos",
            palabra_clave="Batched",
            color_clave=NARANJA_TERRACOTA
        )
        adornos = self._crear_adornos_esquinas()
        self._animar_entrada_slide(
            titulo, linea,
            fondo=llanuras_fondo,
            adornos=adornos
        )

        Y_BOLSAS = DOWN * 0.80
        Y_ANNOT  = DOWN * 2.80
        BUFF_SUB = 0.55

        colores_cab   = [MARRON_OSCURO, NARANJA_TERRACOTA, MARRON_QUIJOTE]
        offsets_x     = [-3.6, 0.0, 3.6]
        nombres_bolsa = ["Operación A", "Operación B", "Operación C"]
        COLOR_LETRA = WHITE

        unidades = VGroup()
        for i in range(3):
            rect = RoundedRectangle(
                corner_radius=0.22, width=2.45, height=1.55,
                fill_color=PAPEL_CREMA, fill_opacity=1.0,
                stroke_color=colores_cab[i], stroke_width=3.0
            )
            letra = Text(
                ["A", "B", "C"][i], font=FUENTE, font_size=28,
                color=COLOR_LETRA, weight=BOLD
            ).move_to(rect).set_z_index(5)
            unidad = VGroup(rect, letra)
            unidad.move_to(RIGHT * offsets_x[i] + Y_BOLSAS)
            unidad.set_z_index(2)
            unidades.add(unidad)

        labels_bolsa = VGroup(*[
            Text(nombres_bolsa[i], font=FUENTE, font_size=14,
                color=colores_cab[i], weight=BOLD)
            .next_to(unidades[i], UP, buff=0.15)
            .set_z_index(3)
            for i in range(3)
        ])

        sub1 = Text(
            "Tres operaciones independientes · múltiples llamadas al procesador",
            font=FUENTE, font_size=18, color=TINTA_NEGRA
        ).next_to(linea, DOWN, buff=BUFF_SUB)
        self.play(Write(sub1))

        self.play(
            LaggedStart(*[
                AnimationGroup(FadeIn(unidades[i]), FadeIn(labels_bolsa[i]))
                for i in range(3)
            ], lag_ratio=0.28),
            run_time=1.0
        )

        annot_viajes = VGroup(*[
            Text("1 llamada", font=FUENTE, font_size=13,
                color=ROJO_CONTRA, weight=BOLD)
            .next_to(unidades[i], DOWN, buff=0.18)
            .set_z_index(3)
            for i in range(3)
        ])

        for i in range(3):
            self.play(
                unidades[i][0].animate.set_fill(colores_cab[i], opacity=1.0),
                FadeIn(annot_viajes[i], shift=UP * 0.08),
                run_time=0.40
            )

        lbl_costo = Text(
            "3 operaciones · 3 llamadas · acceso a memoria ineficiente",
            font=FUENTE, font_size=16, color=ROJO_CONTRA, weight=BOLD
        ).move_to(Y_ANNOT).set_z_index(3)
        self.play(FadeIn(lbl_costo, shift=UP * 0.10))
        self.play(Wiggle(unidades, scale_value=1.05, rotation_angle=0.02))

        self._siguiente()

        self.play(
            FadeOut(sub1), FadeOut(lbl_costo),
            FadeOut(annot_viajes), FadeOut(labels_bolsa),
            *[unidades[i][0].animate.set_fill(PAPEL_CREMA, opacity=1.0)
            for i in range(3)],
            run_time=0.50
        )

        sub2 = Text(
            "Un batch · una sola llamada · ejecución paralela",
            font=FUENTE, font_size=18, color=NARANJA_TERRACOTA, weight=BOLD
        ).next_to(linea, DOWN, buff=BUFF_SUB)
        self.play(Write(sub2))

        contenedor = RoundedRectangle(
            corner_radius=0.20, width=8.5, height=1.55,
            fill_color=PAPEL_CREMA, fill_opacity=1.0,
            stroke_color=MARRON_OSCURO, stroke_width=3.6
        ).move_to(Y_BOLSAS).set_z_index(1)

        ancho = 8.5
        destinos_x = [-ancho / 3, 0.0, ancho / 3]

        self.play(FadeIn(contenedor), run_time=0.40)

        self.play(
            *[unidades[i].animate
                .move_to(contenedor.get_center() + RIGHT * destinos_x[i])
                .scale(0.68)
            for i in range(3)],
            run_time=0.95
        )

        div_xs = [-ancho / 6, ancho / 6]
        divisores = VGroup(*[
            DashedLine(
                contenedor.get_top() + RIGHT * dx,
                contenedor.get_bottom() + RIGHT * dx,
                color=MARRON_OSCURO, stroke_width=1.6, dash_length=0.11
            ).set_z_index(4)
            for dx in div_xs
        ])
        self.play(Create(divisores), run_time=0.45)

        annot_contiguo = Text(
            "A · B · C se ejecutan juntas en una sola operación",
            font=FUENTE, font_size=14, color=MARRON_OSCURO
        ).move_to(Y_ANNOT).set_z_index(3)
        self.play(FadeIn(annot_contiguo, shift=UP * 0.10))

        self.play(
            *[unidades[i][0].animate.set_fill(colores_cab[i], opacity=1.0)
            for i in range(3)],
            run_time=0.55
        )

        self.play(
            *[Flash(unidades[i].get_center(), color=colores_cab[i],
                    line_length=0.30, num_lines=8)
            for i in range(3)],
            run_time=0.60
        )

        self.play(Indicate(contenedor, color=NARANJA_TERRACOTA, scale_factor=1.03))

        msg_final = Text(
            "Multiplicaciones independientes agrupadas en una sola llamada eficiente",
            font=FUENTE, font_size=16, color=MARRON_OSCURO, weight=BOLD
        ).move_to(Y_ANNOT).set_z_index(3)

        self.play(
            FadeTransform(annot_contiguo, msg_final),
            run_time=0.55
        )

        self._siguiente()
        self.limpiar_pantalla()