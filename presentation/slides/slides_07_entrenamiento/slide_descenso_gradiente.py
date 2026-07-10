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


class SlideDescensoGradiente:
    def slide_descenso_gradiente(self):

        titulo, linea = self.crear_titulo(
            "Descenso del Gradiente: Bajando la Colina",
            palabra_clave="Gradiente:",
            color_clave=NARANJA_TERRACOTA
        )
        llanuras_fondo = crear_llanuras_manchegas()
        adornos = self._crear_adornos_esquinas(escala=0.55, buff=0.5)
        adornos[1].add_updater(lambda m, dt: m.rotate(dt * 0.4))
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo, adornos=adornos)

        def func_costo(x):
            return np.sin(x) + 0.3 * (x ** 2) + 2

        def gradiente_costo(x):
            return np.cos(x) + 0.6 * x

        ejes = Axes(
            x_range=[-3, 5, 1],
            y_range=[0, 10, 2],
            x_length=6.5,
            y_length=4.5,
            axis_config={"color": MARRON_OSCURO, "include_numbers": False}
        ).shift(LEFT * 1.8 + DOWN * 0.3)

        lbl_x = ejes.get_x_axis_label(
            Text("Parámetro  w", font=FUENTE, font_size=15, color=MARRON_OSCURO),
            edge=RIGHT, direction=DOWN
        )
        lbl_y = ejes.get_y_axis_label(
            Text("Loss  L", font=FUENTE, font_size=15, color=MARRON_OSCURO),
            edge=UP, direction=UP
        )

        curva = ejes.plot(func_costo, color=NARANJA_TERRACOTA, stroke_width=4)
        area  = ejes.get_area(curva, opacity=0.08, color=NARANJA_TERRACOTA)

        x_minimo = -0.89
        p_minimo = ejes.c2p(x_minimo, func_costo(x_minimo))

        def crear_stickman():
            cabeza  = Circle(radius=0.15, color=MARRON_OSCURO,
                            fill_color=PAPEL_CREMA, fill_opacity=1)
            cuerpo  = Line(cabeza.get_bottom(),
                        cabeza.get_bottom() + DOWN * 0.4,
                        color=MARRON_OSCURO, stroke_width=3)
            brazos  = VGroup(
                Line(cuerpo.get_center() + UP * 0.1,
                    cuerpo.get_center() + UP * 0.3 + LEFT * 0.2,
                    color=MARRON_OSCURO, stroke_width=3),
                Line(cuerpo.get_center() + UP * 0.1,
                    cuerpo.get_center() + UP * 0.3 + RIGHT * 0.2,
                    color=MARRON_OSCURO, stroke_width=3)
            )
            piernas = VGroup(
                Line(cuerpo.get_bottom(),
                    cuerpo.get_bottom() + DOWN * 0.3 + LEFT * 0.2,
                    color=MARRON_OSCURO, stroke_width=3),
                Line(cuerpo.get_bottom(),
                    cuerpo.get_bottom() + DOWN * 0.3 + RIGHT * 0.2,
                    color=MARRON_OSCURO, stroke_width=3)
            )
            mastil  = Line(brazos[0].get_end(),
                        brazos[0].get_end() + UP * 0.5,
                        color=TINTA_NEGRA, stroke_width=2)
            bandera = Polygon(
                mastil.get_end(),
                mastil.get_end() + DOWN * 0.15 + LEFT * 0.3,
                mastil.get_end() + DOWN * 0.3,
                fill_color=NARANJA_TERRACOTA, fill_opacity=1, stroke_width=0
            )
            return VGroup(cabeza, cuerpo, brazos, piernas, mastil, bandera)

        stickman = crear_stickman().scale(0.6).next_to(p_minimo, UP, buff=0)
        lbl_meta = Text("mínimo", font=FUENTE, font_size=13,
                        color=NARANJA_TERRACOTA, weight=BOLD).next_to(stickman, UP, buff=0.08)

        lr = 0.8
        eq_update = MathTex(
            r"w \leftarrow w - \underbrace{\eta}_{\text{lr}} \cdot \underbrace{\nabla L}_{\text{pendiente}}",
            font_size=30, color=MARRON_OSCURO
        )
        eq_frame = SurroundingRectangle(eq_update, color=MARRON_OSCURO,
                                        buff=0.25, corner_radius=0.1)
        eq_grupo = VGroup(eq_frame, eq_update)

        lbl_lr   = Text(f"η = {lr}  (learning rate)", font=FUENTE,
                        font_size=16, color=TINTA_NEGRA)
        lbl_grad = Text("∇L = pendiente en w", font=FUENTE,
                        font_size=16, color=NARANJA_TERRACOTA)

        txt_pos      = Text("w = 4.00", font="Monospace", font_size=16, color=MARRON_OSCURO)
        txt_grad_val = Text("∇L = —",   font="Monospace", font_size=16, color=NARANJA_TERRACOTA)
        txt_loss     = Text("L = —",    font="Monospace", font_size=16, color=ROJO_TOMATE)

        panel = VGroup(eq_grupo, lbl_lr, lbl_grad,
                    txt_pos, txt_grad_val, txt_loss
                    ).arrange(DOWN, buff=0.32, aligned_edge=LEFT
                    ).next_to(ejes, RIGHT, buff=0.55).align_to(ejes, UP).shift(DOWN * 0.3)

        self.play(Create(ejes), Write(lbl_x), Write(lbl_y), run_time=0.9)
        self.play(Create(curva), FadeIn(area), run_time=1.0)
        self.play(FadeIn(stickman, shift=UP * 0.3), Write(lbl_meta), run_time=0.6)
        self.play(FadeIn(panel, shift=LEFT * 0.2), run_time=0.5)
        self._siguiente()

        x_val = 4.0
        punto = Dot(ejes.c2p(x_val, func_costo(x_val)),
                    color=TINTA_NEGRA, radius=0.13)
        self.play(FadeIn(punto, scale=0.5))

        rastros = VGroup()

        for i in range(7):
            grad_actual = gradiente_costo(x_val)
            x_next      = x_val - lr * grad_actual
            x_next      = max(-2.8, min(4.8, x_next))

            p_actual = ejes.c2p(x_val,  func_costo(x_val))
            p_next   = ejes.c2p(x_next, func_costo(x_next))

            nuevo_pos  = Text(f"w = {x_val:.2f}",
                            font="Monospace", font_size=16, color=MARRON_OSCURO
                            ).move_to(txt_pos, aligned_edge=LEFT)
            nuevo_grad = Text(f"∇L = {grad_actual:+.2f}",
                            font="Monospace", font_size=16,
                            color=VERDE_OLIVA if abs(grad_actual) < 0.5 else NARANJA_TERRACOTA
                            ).move_to(txt_grad_val, aligned_edge=LEFT)
            nuevo_loss = Text(f"L = {func_costo(x_val):.2f}",
                            font="Monospace", font_size=16, color=ROJO_TOMATE
                            ).move_to(txt_loss, aligned_edge=LEFT)

            self.play(
                Transform(txt_pos,      nuevo_pos),
                Transform(txt_grad_val, nuevo_grad),
                Transform(txt_loss,     nuevo_loss),
                run_time=0.35
            )

            dx      = 0.55
            t_start = ejes.c2p(x_val - dx, func_costo(x_val) - grad_actual * dx)
            t_end   = ejes.c2p(x_val + dx, func_costo(x_val) + grad_actual * dx)
            tangente = Line(t_start, t_end, color=AZUL_NOCHE,
                            stroke_width=2.5).set_opacity(0.55)
            self.play(Create(tangente), run_time=0.25)

            angulo_salto = -TAU / 6 if grad_actual > 0 else TAU / 6
            flecha_salto = CurvedArrow(p_actual, p_next, angle=angulo_salto,
                                    color=NARANJA_TERRACOTA, stroke_width=2.8)
            self.play(Create(flecha_salto), run_time=0.35)

            rastro = Dot(p_actual, color=MARRON_OSCURO, radius=0.055).set_opacity(0.35)
            rastros.add(rastro)
            self.add(rastro)

            self.play(
                punto.animate.move_to(p_next),
                FadeOut(tangente),
                FadeOut(flecha_salto),
                run_time=0.5
            )
            x_val = x_next

        self.play(
            Wiggle(stickman, scale_value=1.18, rotation_angle=0.04 * TAU),
            Flash(punto, color=NARANJA_TERRACOTA, line_length=0.45,
                flash_radius=0.32, num_lines=12),
            run_time=1.0
        )
        self._siguiente()

        adornos[1].clear_updaters()
        self.play(
            FadeOut(ejes), FadeOut(lbl_x), FadeOut(lbl_y),
            FadeOut(curva), FadeOut(area),
            FadeOut(stickman), FadeOut(lbl_meta),
            FadeOut(punto), FadeOut(rastros),
            FadeOut(panel), FadeOut(adornos)
        )
        self.limpiar_pantalla()


