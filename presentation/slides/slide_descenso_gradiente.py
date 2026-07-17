import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import numpy as np
import os
from colores import *
from objetos import *


class SlideDescensoGradiente:
    def slide_descenso_gradiente(self):

        titulo, linea = self.crear_titulo(
            "Descenso del Gradiente",
            palabra_clave="Gradiente",
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
            x_length=9.4,
            y_length=5.2,
            axis_config={"color": MARRON_OSCURO, "include_numbers": False, "stroke_width": 2.5},
            tips=True,
        ).shift(DOWN * 0.35)

        lbl_x = ejes.get_x_axis_label(
            Text("Parámetro  w", font=FUENTE, font_size=17, color=MARRON_OSCURO),
            edge=RIGHT, direction=DOWN
        )
        lbl_y = ejes.get_y_axis_label(
            Text("Loss  L", font=FUENTE, font_size=17, color=MARRON_OSCURO),
            edge=UP, direction=UP
        )

        # rejilla tenue para dar detalle
        rejilla = VGroup()
        for xv in np.arange(-3, 5.01, 1):
            rejilla.add(Line(ejes.c2p(xv, 0), ejes.c2p(xv, 10),
                             color=MARRON_OSCURO, stroke_width=0.8).set_opacity(0.12))
        for yv in np.arange(0, 10.01, 2):
            rejilla.add(Line(ejes.c2p(-3, yv), ejes.c2p(5, yv),
                             color=MARRON_OSCURO, stroke_width=0.8).set_opacity(0.12))

        curva = ejes.plot(func_costo, color=NARANJA_TERRACOTA, stroke_width=5)
        area  = ejes.get_area(curva, x_range=[-3, 5], color=NARANJA_TERRACOTA, opacity=0.14)

        x_minimo = -0.89
        p_minimo = ejes.c2p(x_minimo, func_costo(x_minimo))

        # marcador del mínimo: el hidalgo cangrejo espera abajo
        linea_min = DashedLine(ejes.c2p(x_minimo, 0), p_minimo,
                               color=VERDE_OLIVA, stroke_width=2).set_opacity(0.55)
        dot_min = Dot(p_minimo, color=VERDE_OLIVA, radius=0.08)
        hidalgo = ImageMobject(
            os.path.join("assets", "quijote_rust.png")
        ).set_height(1.0).next_to(p_minimo, UP, buff=0.05)
        lbl_meta = Text("mínimo", font=FUENTE, font_size=15,
                        color=VERDE_OLIVA, weight=BOLD).next_to(hidalgo, UP, buff=0.08)

        self.play(Create(ejes), FadeIn(rejilla), Write(lbl_x), Write(lbl_y), run_time=1.0)
        self.play(Create(curva), FadeIn(area), run_time=1.1)
        self.play(Create(linea_min), FadeIn(dot_min),
                  FadeIn(hidalgo, shift=UP * 0.3), Write(lbl_meta), run_time=0.7)
        self._siguiente()

        # bola con halo que baja suavemente siguiendo la curva
        lr = 0.8
        x0 = 4.0
        x_tracker = ValueTracker(x0)
        halo = Circle(radius=0.24, color=NARANJA_TERRACOTA, fill_opacity=0.3, stroke_width=0)
        nucleo = Dot(radius=0.12, color=TINTA_NEGRA)
        bola = VGroup(halo, nucleo).move_to(ejes.c2p(x0, func_costo(x0)))
        bola.add_updater(lambda m: m.move_to(
            ejes.c2p(x_tracker.get_value(), func_costo(x_tracker.get_value()))))

        # rastro suave que sigue a la bola por la curva
        rastro = TracedPath(lambda: nucleo.get_center(),
                            stroke_color=NARANJA_TERRACOTA, stroke_width=3.5,
                            stroke_opacity=0.5)
        self.add(rastro)

        lbl_inicio = Text("inicio", font=FUENTE, font_size=14, weight=BOLD,
                          color=MARRON_OSCURO).next_to(bola, UR, buff=0.05)
        self.play(FadeIn(bola, scale=0.5), FadeIn(lbl_inicio))

        # trayectoria del descenso del gradiente
        xs = [x0]
        xv = x0
        for _ in range(9):
            xv = max(-2.8, min(4.8, xv - lr * gradiente_costo(xv)))
            xs.append(xv)

        tangentes = VGroup()
        for i in range(9):
            x_from, x_to = xs[i], xs[i + 1]
            grad_actual = gradiente_costo(x_from)

            dx = 0.7
            t_start = ejes.c2p(x_from - dx, func_costo(x_from) - grad_actual * dx)
            t_end   = ejes.c2p(x_from + dx, func_costo(x_from) + grad_actual * dx)
            tangente = Line(t_start, t_end, color=AZUL_NOCHE, stroke_width=3).set_opacity(0.7)
            tangentes.add(tangente)

            extra = [FadeOut(lbl_inicio)] if i == 0 else []
            self.play(Create(tangente), *extra, run_time=0.22)
            self.play(
                x_tracker.animate.set_value(x_to),
                tangente.animate.set_opacity(0.15),
                run_time=0.75, rate_func=smooth,
            )

        bola.clear_updaters()
        self.play(
            Wiggle(hidalgo, scale_value=1.18, rotation_angle=0.04 * TAU),
            Flash(bola, color=NARANJA_TERRACOTA, line_length=0.45,
                  flash_radius=0.34, num_lines=14),
            run_time=1.0
        )
        self._siguiente()

        adornos[1].clear_updaters()
        self.play(
            FadeOut(ejes), FadeOut(rejilla), FadeOut(lbl_x), FadeOut(lbl_y),
            FadeOut(curva), FadeOut(area), FadeOut(linea_min), FadeOut(dot_min),
            FadeOut(hidalgo), FadeOut(lbl_meta),
            FadeOut(bola), FadeOut(rastro), FadeOut(tangentes),
            FadeOut(adornos)
        )
        self.limpiar_pantalla()


