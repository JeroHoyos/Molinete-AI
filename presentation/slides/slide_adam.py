import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import numpy as np
import os
from colores import *
from objetos import *


class SlideAdam:
    def slide_adam(self):

        titulo_intro, linea_intro = self.crear_titulo(
            "Optimizador Adam",
            palabra_clave="Adam",
            color_clave=NARANJA_TERRACOTA
        )
        self.play(Write(titulo_intro), GrowFromCenter(linea_intro))

        formula_adam = MathTex(
            r"w_{t + 1} \leftarrow w_{t} - ",
            r"\frac{\eta}{\sqrt{v_{t}} + \epsilon} ",
            r"\cdot m_{t}",
            color=TINTA_NEGRA, font_size=60
        ).shift(UP * 2)

        self.play(Write(formula_adam))
        self._siguiente()

        ejes_m = Axes(x_range=[0, 4], y_range=[0, 2], x_length=4, y_length=2).shift(LEFT*3.2 + DOWN*1.5)

        func_m = lambda x: 0.5 * np.cos(3*x) - 0.2 * x + 1.5
        terreno_m = ejes_m.plot(func_m, color=MARRON_OSCURO)

        lbl_m = Text("m: Momentum", font_size=24, color=NARANJA_TERRACOTA).next_to(ejes_m, UP)

        bola_gd = Dot(color=MARRON_OSCURO).move_to(ejes_m.c2p(0.5, func_m(0.5)))
        bola_adam = Dot(color=NARANJA_TERRACOTA).move_to(ejes_m.c2p(0.5, func_m(0.5)))

        self.play(
            formula_adam[1].animate.set_opacity(0.3),
            formula_adam[2].animate.set_color(NARANJA_TERRACOTA).scale(1.2),
            Create(ejes_m), Create(terreno_m), Write(lbl_m),
            FadeIn(bola_gd), FadeIn(bola_adam)
        )

        tracker_gd = ValueTracker(0.5)
        tracker_adam = ValueTracker(0.5)

        bola_gd.add_updater(lambda d: d.move_to(ejes_m.c2p(tracker_gd.get_value(), func_m(tracker_gd.get_value()))))
        bola_adam.add_updater(lambda d: d.move_to(ejes_m.c2p(tracker_adam.get_value(), func_m(tracker_adam.get_value()))))

        self.play(
            tracker_gd.animate.set_value(1.2),
            tracker_adam.animate.set_value(3.2),
            run_time=2.5, rate_func=smooth
        )

        bola_gd.clear_updaters()
        bola_adam.clear_updaters()
        self._siguiente()

        ejes_v = Axes(x_range=[0, 4], y_range=[0, 4], x_length=4, y_length=2).shift(RIGHT*3.2 + DOWN*1.5)
        func_v = lambda x: 0.8 * (x - 2.5)**2 + 0.5
        terreno_v = ejes_v.plot(func_v, color=MARRON_OSCURO)

        lbl_v = Text("v: Varianza", font_size=24, color=MARRON_OSCURO).next_to(ejes_v, UP)

        self.play(
            formula_adam[2].animate.set_color(TINTA_NEGRA).set_opacity(0.3).scale(1/1.2),
            formula_adam[1].animate.set_opacity(1).set_color(NARANJA_TERRACOTA).scale(1.2),
            Create(ejes_v), Create(terreno_v), Write(lbl_v)
        )

        x_inicio = 0.8
        punto_base = Dot(ejes_v.c2p(x_inicio, func_v(x_inicio)), color=TINTA_NEGRA)
        self.play(FadeIn(punto_base))

        x_sobreimpulso = 3.8
        flecha_mala = Arrow(
            start=ejes_v.c2p(x_inicio, func_v(x_inicio)),
            end=ejes_v.c2p(x_sobreimpulso, func_v(x_sobreimpulso)),
            color=MARRON_OSCURO, buff=0.1
        )
        lbl_mala = Text("Gradiente puro: paso gigante", font_size=16, color=MARRON_OSCURO).next_to(flecha_mala, UP)

        self.play(GrowArrow(flecha_mala), Write(lbl_mala))
        self.wait(0.5)

        x_ideal = 2.0
        flecha_buena = Arrow(
            start=ejes_v.c2p(x_inicio, func_v(x_inicio)),
            end=ejes_v.c2p(x_ideal, func_v(x_ideal)),
            color=NARANJA_TERRACOTA, buff=0.1
        )
        lbl_buena = Text("Adam: paso adaptativo", font_size=16, color=NARANJA_TERRACOTA).next_to(flecha_buena, DOWN)

        self.play(
            ReplacementTransform(flecha_mala, flecha_buena),
            ReplacementTransform(lbl_mala, lbl_buena)
        )
        self.wait(0.5)

        tracker_v = ValueTracker(x_inicio)
        punto_v = Dot(color=NARANJA_TERRACOTA)
        punto_v.add_updater(lambda d: d.move_to(ejes_v.c2p(tracker_v.get_value(), func_v(tracker_v.get_value()))))

        self.add(punto_v)
        self.remove(punto_base)

        self.play(
            tracker_v.animate.set_value(x_ideal),
            run_time=2,
            rate_func=smooth
        )
        punto_v.clear_updaters()

        self.play(FadeOut(VGroup(
            ejes_m, terreno_m, lbl_m, bola_gd, bola_adam,
            ejes_v, terreno_v, lbl_v, flecha_buena, lbl_buena, punto_v,
            formula_adam, titulo_intro, linea_intro
        )))

        titulo, linea = self.crear_titulo("ADAM vs GD", color_clave=NARANJA_TERRACOTA)
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)

        def func_costo(x):
            return np.cos(3*x)*0.5 + 0.2 * (x**2) + 2

        ejes_gd = Axes(x_range=[-3, 5, 1], y_range=[0, 8, 2], x_length=5.5, y_length=4,
                       axis_config={"color": MARRON_OSCURO}).shift(LEFT * 3.4 + DOWN * 0.5)
        ejes_adam = Axes(x_range=[-3, 5, 1], y_range=[0, 8, 2], x_length=5.5, y_length=4,
                         axis_config={"color": NARANJA_TERRACOTA}).shift(RIGHT * 3.4 + DOWN * 0.5)

        lbl_titulo_gd = Text("Gradient Descent Normal", font_size=24, color=MARRON_OSCURO).next_to(ejes_gd, UP)
        lbl_titulo_adam = Text("Optimizador ADAM", font_size=24, color=NARANJA_TERRACOTA).next_to(ejes_adam, UP)

        self.play(
            Create(VGroup(ejes_gd, ejes_gd.plot(func_costo, color=MARRON_OSCURO))),
            Create(VGroup(ejes_adam, ejes_adam.plot(func_costo, color=NARANJA_TERRACOTA))),
            Write(lbl_titulo_gd),
            Write(lbl_titulo_adam)
        )

        pasos_gd, pasos_adam = [4.0], [4.0]
        x_g, x_a = 4.0, 4.0
        m, v = 0, 0

        for t in range(1, 60):
            g_a = -np.sin(3*x_a)*1.5 + 0.4*x_a
            m = 0.9 * m + 0.1 * g_a
            v = 0.999 * v + 0.001 * (g_a**2)
            m_hat = m / (1 - 0.9**t)
            v_hat = v / (1 - 0.999**t)

            if abs(x_a) < 0.05 and t > 25:
                x_a = 0.0
            else:
                x_a -= 0.35 * m_hat / (np.sqrt(v_hat) + 1e-8)

            pasos_adam.append(x_a)
            g_g = -np.sin(3*x_g)*1.5 + 0.4*x_g
            x_g -= 0.12 * g_g
            pasos_gd.append(x_g)

        trail_gd_points = [ejes_gd.c2p(x, func_costo(x)) for x in pasos_gd]
        trail_adam_points = [ejes_adam.c2p(x, func_costo(x)) for x in pasos_adam]

        linea_rastro_gd = VMobject(color=MARRON_OSCURO).set_points_smoothly(trail_gd_points)
        linea_rastro_adam = VMobject(color=NARANJA_TERRACOTA).set_points_smoothly(trail_adam_points)

        punto_gd = Dot(trail_gd_points[0], color=MARRON_OSCURO)
        punto_adam = Dot(trail_adam_points[0], color=NARANJA_TERRACOTA)

        self.play(FadeIn(punto_gd), FadeIn(punto_adam))

        self.play(
            MoveAlongPath(punto_gd, linea_rastro_gd),
            Create(linea_rastro_gd),
            MoveAlongPath(punto_adam, linea_rastro_adam),
            Create(linea_rastro_adam),
            run_time=4, rate_func=smooth
        )

        lbl_resultado_gd = Text("Mínimo Local", font_size=20, color=MARRON_OSCURO).next_to(punto_gd, DOWN)
        lbl_resultado_adam = Text("Mínimo Global", font_size=20, color=NARANJA_TERRACOTA).next_to(punto_adam, DOWN)

        self.play(
            Write(lbl_resultado_gd),
            Write(lbl_resultado_adam),
            Flash(punto_adam, color=NARANJA_TERRACOTA)
        )

        self._siguiente()

        panel_textos = VGroup(
            Text("¿Por qué ganó ADAM?", font_size=26, color=TINTA_NEGRA, weight=BOLD),
            Text("• El Momentum saltó los baches locales.", font_size=20, color=MARRON_OSCURO),
            Text("• La Varianza controló los pasos bruscos.", font_size=20, color=MARRON_OSCURO)
        ).arrange(DOWN, aligned_edge=LEFT)

        fondo_panel = Rectangle(
            width=panel_textos.width + 1, height=panel_textos.height + 0.7,
            color=PAPEL_CREMA, fill_color=FONDO_CAJA, fill_opacity=1
        )
        panel_final = VGroup(fondo_panel, panel_textos).to_edge(DOWN, buff=0.2)

        self.play(FadeIn(panel_final, shift=UP))
        self._siguiente()
        self.limpiar_pantalla()


