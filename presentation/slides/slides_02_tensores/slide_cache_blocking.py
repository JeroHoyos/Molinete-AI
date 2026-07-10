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


class SlideCacheBlocking:
    def slide_cache_blocking(self):
        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "Cache Blocking: El Arte de Encajar",
            palabra_clave="Cache",
            color_clave=NARANJA_TERRACOTA
        )
        adornos = self._crear_adornos_esquinas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo, adornos=adornos)


        ram_box = RoundedRectangle(corner_radius=0.15, height=4.0, width=2.6,
                                   stroke_color=MARRON_OSCURO, stroke_width=3,
                                   fill_color=PAPEL_CREMA, fill_opacity=0.85)
        ram_box.move_to(LEFT * 3.2 + DOWN * 0.5)
        label_ram = Text("RAM", font=FUENTE, font_size=24, color=TINTA_NEGRA,
                         weight=BOLD).move_to(ram_box.get_top() + DOWN * 0.35)
        sub_ram = Text("Memoria principal", font=FUENTE, font_size=13,
                       color=MARRON_OSCURO).next_to(label_ram, DOWN, buff=0.04)


        matriz_ram = VGroup(*[
            Square(side_length=0.33,
                   stroke_color=MARRON_OSCURO, stroke_opacity=0.55,
                   fill_color=BEIGE_MEDIO, fill_opacity=0.65)
            for _ in range(36)
        ]).arrange_in_grid(6, 6, buff=0.03).next_to(sub_ram, DOWN, buff=0.14)


        cpu_box = RoundedRectangle(corner_radius=0.18, height=4.0, width=3.6,
                                   stroke_color=MARRON_OSCURO, stroke_width=3,
                                   fill_color=PAPEL_CREMA, fill_opacity=0.25)
        cpu_box.move_to(RIGHT * 2.6 + DOWN * 0.5)
        label_cpu = Text("CPU", font=FUENTE, font_size=24, color=TINTA_NEGRA,
                         weight=BOLD).move_to(cpu_box.get_top() + DOWN * 0.35)

        cache_box = RoundedRectangle(corner_radius=0.12, height=1.6, width=1.6,
                                     stroke_color=NARANJA_TERRACOTA, stroke_width=3,
                                     fill_color=NARANJA_TERRACOTA, fill_opacity=0.18)
        cache_box.next_to(label_cpu, DOWN, buff=0.40)
        label_cache = Text("Caché L1", font=FUENTE, font_size=14,
                           color=TINTA_NEGRA, weight=BOLD).move_to(cache_box.get_center())

        flecha_bus = Arrow(
            ram_box.get_right(), cpu_box.get_left(),
            color=MARRON_OSCURO, stroke_width=3, buff=0.10
        )
        label_bus = Text("Bus de datos", font=FUENTE, font_size=13,
                         color=MARRON_OSCURO).next_to(flecha_bus, UP, buff=0.08)

        self.play(DrawBorderThenFill(ram_box), Write(label_ram), FadeIn(sub_ram))
        self.play(Create(matriz_ram, lag_ratio=0.04), run_time=1.1)
        self.play(DrawBorderThenFill(cpu_box), Write(label_cpu))
        self.play(GrowArrow(flecha_bus), FadeIn(label_bus))
        self.play(DrawBorderThenFill(cache_box), Write(label_cache))


        texto_prob = Text(
            "MatMul necesita datos que no caben completos en caché",
            font=FUENTE, font_size=19, color=TINTA_NEGRA
        ).next_to(linea, DOWN, buff=0.22)
        self.play(Write(texto_prob))


        col0 = VGroup(*[matriz_ram[i * 6 + 0] for i in range(6)])
        col1 = VGroup(*[matriz_ram[i * 6 + 1] for i in range(6)])
        col0_copy = col0.copy()
        col1_copy = col1.copy()

        self.play(
            col0_copy.animate.set_fill(MARRON_QUIJOTE, opacity=0.90),
            col1_copy.animate.set_fill(NARANJA_TERRACOTA, opacity=0.70),
        )


        grupo_cols = VGroup(col0_copy, col1_copy)
        self.play(
            grupo_cols.animate.move_to(cache_box.get_center()).scale(0.70),
            run_time=1.1
        )


        texto_lento = Text(
            "RAM: ~100 ns de latencia  ·  caché L1: ~1 ns",
            font=FUENTE, font_size=17, color=MARRON_OSCURO
        ).next_to(linea, DOWN, buff=0.22)
        self.play(FadeTransform(texto_prob, texto_lento))

        texto_error = Text(
            "No caben → hay que releer desde RAM en cada bloque",
            font=FUENTE, font_size=18, color=NARANJA_TERRACOTA, weight=BOLD
        ).next_to(texto_lento, DOWN, buff=0.18)
        self.play(FadeIn(texto_error, shift=UP * 0.1))

        self.play(
            Wiggle(cache_box, scale_value=1.12, rotation_angle=0.04),
            Flash(cache_box, color=NARANJA_TERRACOTA, line_length=0.38, num_lines=10)
        )
        self.play(FadeOut(grupo_cols))
        self.wait(1.5)


        self._siguiente()


        texto_sol = Text(
            "Solución: llevar bloques 2×2 de a poco a la caché",
            font=FUENTE, font_size=19, color=TINTA_NEGRA
        ).next_to(linea, DOWN, buff=0.22)
        self.play(
            FadeOut(texto_error),
            FadeTransform(texto_lento, texto_sol)
        )


        bloques_2x2 = [
            [r * 6 + c, r * 6 + c + 1, (r+1) * 6 + c, (r+1) * 6 + c + 1]
            for c in range(0, 6, 2)
            for r in range(0, 6, 2)
        ]
        colores_ciclo = [NARANJA_TERRACOTA, MARRON_QUIJOTE, MARRON_OSCURO]

        for k, indices in enumerate(bloques_2x2):
            color = colores_ciclo[k % len(colores_ciclo)]
            bloque_orig = VGroup(*[matriz_ram[i] for i in indices])
            bloque_copia = bloque_orig.copy()


            self.play(
                bloque_copia.animate.set_fill(color, opacity=0.88),
                run_time=0.22
            )

            self.play(
                bloque_copia.animate.move_to(cache_box.get_center()).scale(0.80),
                run_time=0.45
            )

            self.play(
                Flash(cache_box, color=color, line_length=0.22, num_lines=7),
                run_time=0.30
            )

            self.play(FadeOut(bloque_copia, shift=RIGHT * 0.4), run_time=0.28)

        resumen = Text(
            "Bloque a bloque: la caché nunca se desborda",
            font=FUENTE, font_size=19, color=TINTA_NEGRA, weight=BOLD
        ).next_to(linea, DOWN, buff=0.25)
        self.play(FadeOut(texto_sol), FadeIn(resumen, shift=UP * 0.12))
        self.play(Indicate(cache_box, color=NARANJA_TERRACOTA, scale_factor=1.12))
        self.wait(1.5)


        self._siguiente()
        self.limpiar_pantalla()


