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


class SlideResidual:
    def slide_residual(self):

        escala = 0.85

        sol_fondo = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN*0.2 + LEFT*0.2)
        herradura_fondo = crear_herradura().scale(0.6).to_corner(DL).shift(UP*0.3 + RIGHT*0.3)
        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "Conexiones Residuales: El Atajo de Sancho",
            palabra_clave="Residuales",
            color_clave=NARANJA_TERRACOTA
        )

        self._animar_entrada_slide(titulo, linea, fondo=VGroup(llanuras_fondo, sol_fondo, herradura_fondo))

        pos_x = LEFT * 4
        pos_f = LEFT * 0.5
        pos_add = RIGHT * 2.5
        pos_y = RIGHT * 4.5

        label_x = MathTex("x", font_size=48 * escala, color=TINTA_NEGRA).move_to(pos_x)

        nodo_f = RoundedRectangle(corner_radius=0.2, width=1.5 * escala, height=2.2 * escala, fill_color=MARRON_QUIJOTE, fill_opacity=1, stroke_color=BLANCO, stroke_width=3)
        label_f = MathTex("f", font_size=48 * escala, color=BLANCO).move_to(nodo_f)
        grupo_f = VGroup(nodo_f, label_f).move_to(pos_f)

        nodo_add = Circle(radius=0.4 * escala, fill_color=MARRON_QUIJOTE, fill_opacity=1, stroke_color=BLANCO, stroke_width=3)
        label_add = MathTex("+", font_size=40 * escala, color=BLANCO).move_to(nodo_add)
        grupo_add = VGroup(nodo_add, label_add).move_to(pos_add)

        label_y = MathTex("y", font_size=48 * escala, color=TINTA_NEGRA).move_to(pos_y)

        eq_final = MathTex("y = x + f(x)", font_size=42 * escala, color=TINTA_NEGRA).next_to(grupo_f, DOWN, buff=1.2)

        arrow_x_f = Arrow(label_x.get_right(), grupo_f.get_left(), buff=0.2, color=MARRON_OSCURO, stroke_width=3)
        arrow_f_add = Arrow(grupo_f.get_right(), grupo_add.get_left(), buff=0.2, color=MARRON_OSCURO, stroke_width=3)
        arrow_add_y = Arrow(grupo_add.get_right(), label_y.get_left(), buff=0.2, color=MARRON_OSCURO, stroke_width=3)

        punto_inicio_skip = pos_x + RIGHT * 0.8
        p1 = punto_inicio_skip
        p2 = p1 + UP * 2.2
        p3 = np.array([grupo_add.get_center()[0], p2[1], 0])
        p4 = grupo_add.get_top() + UP * 0.1

        line_up = Line(p1, p2, color=MARRON_OSCURO, stroke_width=3)
        line_across = Line(p2, p3, color=MARRON_OSCURO, stroke_width=3)
        arrow_down = Arrow(p3, p4, buff=0, color=MARRON_OSCURO, stroke_width=3)
        skip_connection = VGroup(line_up, line_across, arrow_down)

        self.play(FadeIn(label_x, shift=RIGHT))
        self.play(GrowArrow(arrow_x_f), FadeIn(grupo_f, shift=RIGHT))
        self.play(GrowArrow(arrow_f_add), FadeIn(grupo_add, scale=0.5))
        self.play(Create(skip_connection))
        self.play(GrowArrow(arrow_add_y), FadeIn(label_y, shift=RIGHT))
        self.play(Write(eq_final))
        self._siguiente()

        pos_texto = DOWN * 3

        txt_desc_1 = Text("La señal viaja...", font=FUENTE, font_size=24 * escala, color=TINTA_NEGRA).move_to(pos_texto)
        txt_desc_2 = Text("f(x) transforma... la señal se desvanece", font=FUENTE, font_size=24 * escala, color=MARRON_QUIJOTE).move_to(pos_texto)
        txt_desc_3 = Text("Sancho lleva la copia por el atajo", font=FUENTE, font_size=24 * escala, color=NARANJA_TERRACOTA).move_to(pos_texto)
        txt_desc_4 = Text("Realidad + visión se suman en +", font=FUENTE, font_size=24 * escala, color=MARRON_OSCURO).move_to(pos_texto)

        def crear_imagen_pixelada(resolucion="alta"):
            cuadros = []
            filas, cols = (6, 6) if resolucion == "alta" else (3, 3)
            lado = 0.15 if resolucion == "alta" else 0.3
            colores = [NARANJA_TERRACOTA, MARRON_OSCURO, MARRON_QUIJOTE, OCRE_CERVANTINO]

            for i in range(filas):
                for j in range(cols):
                    color = colores[(i*j) % len(colores)]
                    cuadro = Square(side_length=lado, fill_color=color, fill_opacity=1, stroke_width=0.5, stroke_color=BLANCO)
                    cuadros.append(cuadro)

            img = VGroup(*cuadros).arrange_in_grid(rows=filas, cols=cols, buff=0)
            if resolucion == "baja":
                img.set_opacity(0.6)
            return img

        img_alta_x = crear_imagen_pixelada("alta").move_to(pos_x).shift(UP*1.2)
        img_baja_f = crear_imagen_pixelada("baja").move_to(pos_f).shift(UP*1.2)
        img_alta_copia = crear_imagen_pixelada("alta").move_to(pos_x).shift(UP*1.2)
        img_final_y = crear_imagen_pixelada("alta").move_to(pos_y).shift(UP*1.2)

        txt_input = Text("Imagen Real", font=FUENTE, font_size=18, color=MARRON_OSCURO).next_to(img_alta_x, UP)
        self.play(FadeIn(img_alta_x, shift=DOWN), FadeIn(txt_input), FadeIn(txt_desc_1))
        self._siguiente()

        self.play(ReplacementTransform(txt_desc_1, txt_desc_2))
        txt_f = Text("Visión Alterada f(x)", font=FUENTE, font_size=18, color=MARRON_QUIJOTE).next_to(img_baja_f, UP)
        self.play(
            ReplacementTransform(img_alta_x.copy(), img_baja_f),
            FadeIn(txt_f)
        )
        self._siguiente()

        self.play(ReplacementTransform(txt_desc_2, txt_desc_3))
        txt_skip = Text("La Copia de Sancho", font=FUENTE, font_size=18, color=NARANJA_TERRACOTA).next_to(line_across, UP)
        self.play(FadeIn(txt_skip))
        self.play(
            img_alta_copia.animate.move_to(p2).shift(UP*0.5),
            run_time=0.8
        )
        self.play(
            img_alta_copia.animate.move_to(p3).shift(UP*0.5),
            run_time=1.5
        )
        self.play(
            img_alta_copia.animate.move_to(pos_add).shift(UP*1.2),
            run_time=0.8
        )
        self._siguiente()

        self.play(ReplacementTransform(txt_desc_3, txt_desc_4))
        txt_output = Text("Realidad Recuperada", font=FUENTE, font_size=18, color=MARRON_OSCURO).next_to(img_final_y, UP)

        self.play(
            FadeOut(img_baja_f, shift=RIGHT),
            img_alta_copia.animate.move_to(pos_y).shift(UP*1.2),
            FadeIn(txt_output)
        )

        caja_eq = SurroundingRectangle(eq_final, color=NARANJA_TERRACOTA, buff=0.2, stroke_width=2)
        self.play(Create(caja_eq))
        self._siguiente()

        self.limpiar_pantalla()
