import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import numpy as np
import os
from colores import *
from objetos import *


class SlideTrainingMetrics:
    def slide_training_metrics(self):

        llanuras_fondo = crear_llanuras_manchegas()
        adornos = self._crear_adornos_esquinas(escala=0.55, buff=0.5)
        adornos[1].add_updater(lambda m, dt: m.rotate(dt * 0.4))

        titulo, linea = self.crear_titulo(
            "Métricas de Entrenamiento",
            palabra_clave="Métricas",
            color_clave=NARANJA_TERRACOTA
        )
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo, adornos=adornos)


        # helpers para los exámenes tipo quiz con barras de seguridad
        BARW = 2.6

        def hacer_quiz(pregunta_txt, opciones):
            preg_bg = RoundedRectangle(corner_radius=0.14, width=6.6, height=0.6,
                                       fill_color=CAJA_INFERIOR, fill_opacity=0.6,
                                       stroke_color=MARRON_OSCURO, stroke_width=1.8)
            preg_tx = Text(pregunta_txt, font=FUENTE, font_size=20, color=TINTA_NEGRA,
                           t2c={"___": NARANJA_TERRACOTA}).move_to(preg_bg)
            pregunta = VGroup(preg_bg, preg_tx)
            filas, cajas, radios, tracks = VGroup(), [], [], []
            for L, w in opciones:
                caja = RoundedRectangle(corner_radius=0.12, width=6.2, height=0.6,
                                        fill_color=PAPEL_TAN, fill_opacity=0.28,
                                        stroke_color=MARRON_OSCURO, stroke_width=1.5)
                radio = Circle(radius=0.11, stroke_color=MARRON_OSCURO, stroke_width=2,
                               fill_color=PAPEL_CREMA, fill_opacity=1)\
                    .move_to(caja.get_left() + RIGHT * 0.35)
                lab = Text(f"{L})  {w}", font=FUENTE, font_size=18, color=TINTA_NEGRA)\
                    .next_to(radio, RIGHT, buff=0.18)
                track = RoundedRectangle(corner_radius=0.15, width=BARW, height=0.3,
                                         fill_color=CREMA_CALIDA, fill_opacity=1,
                                         stroke_color=MARRON_OSCURO, stroke_width=1.2)\
                    .move_to(caja).align_to(caja, RIGHT).shift(LEFT * 0.95)
                filas.add(VGroup(caja, radio, lab, track))
                cajas.append(caja); radios.append(radio); tracks.append(track)
            filas.arrange(DOWN, buff=0.2)
            return pregunta, filas, cajas, radios, tracks

        def barra_conf(track, frac, color):
            w = max(0.12, frac * BARW)
            f = RoundedRectangle(corner_radius=min(0.12, w / 2), width=w, height=0.22,
                                 fill_color=color, fill_opacity=1, stroke_width=0)
            f.move_to(track.get_center()).align_to(track, LEFT).shift(RIGHT * 0.05)
            return f

        def pct_texto(track, frac, color):
            return Text(f"{int(round(frac * 100))}%", font="Monospace", font_size=15,
                        weight=BOLD, color=color).next_to(track, RIGHT, buff=0.16)

        def bocadillo(texto, crab, color):
            txt = Text(texto, font=FUENTE, font_size=18, weight=BOLD, color=TINTA_NEGRA)
            bg = RoundedRectangle(corner_radius=0.18, width=txt.width + 0.5, height=txt.height + 0.42,
                                  fill_color=PAPEL_CREMA, fill_opacity=1,
                                  stroke_color=color, stroke_width=2.2)
            txt.move_to(bg)
            grupo = VGroup(bg, txt).next_to(crab, UP, buff=0.28)
            cola = Polygon(grupo[0].get_bottom() + LEFT * 0.14,
                           grupo[0].get_bottom() + RIGHT * 0.14,
                           crab.get_top() + UP * 0.05,
                           color=color, fill_color=PAPEL_CREMA, fill_opacity=1, stroke_width=2.2)
            return VGroup(cola, bg, txt)

        # ── Acto 1: dos preguntas a Quijote — una falla, otra acierta ──
        crab1 = ImageMobject(os.path.join("assets", "quijote_rust.png")).set_height(1.8)

        # ---- Pregunta 1: Quijote FALLA ----
        preg1, filas1, cajas1, rad1, trk1 = hacer_quiz(
            '"Don Quijote atacó unos  ___"',
            [("A", "molinos"), ("B", "gigantes"), ("C", "ovejas"), ("D", "monjes")],
        )
        preg1.next_to(linea, DOWN, buff=0.9)
        filas1.next_to(preg1, DOWN, buff=0.34)
        crab1.next_to(filas1, LEFT, buff=0.25)

        # termómetro de Loss persistente a la derecha
        MET_H = 2.5
        met_track = RoundedRectangle(corner_radius=0.26, width=0.55, height=MET_H,
                                     fill_color=CREMA_CALIDA, fill_opacity=1,
                                     stroke_color=MARRON_OSCURO, stroke_width=1.5)\
            .move_to([5.05, filas1.get_center()[1], 0])
        met_lbl = Text("Loss", font=FUENTE, font_size=16, weight=BOLD,
                       color=MARRON_OSCURO).next_to(met_track, UP, buff=0.15)

        def met_fill(loss_val, color):
            h = max(0.14, (loss_val / 3.5) * MET_H)
            f = RoundedRectangle(corner_radius=min(0.2, h / 2), width=0.42, height=h,
                                 fill_color=color, fill_opacity=1, stroke_width=0)
            f.move_to(met_track.get_bottom(), aligned_edge=DOWN).shift(UP * 0.06)
            return f

        def met_num(txt, color):
            return Text(txt, font="Monospace", font_size=22, weight=BOLD,
                        color=color).next_to(met_track, DOWN, buff=0.12)

        self.play(FadeIn(preg1, shift=DOWN * 0.1), FadeIn(crab1, shift=RIGHT * 0.2),
                  Create(met_track), FadeIn(met_lbl))
        self.play(LaggedStart(*[FadeIn(f, shift=UP * 0.1) for f in filas1], lag_ratio=0.12),
                  run_time=1.0)

        conf1 = [0.04, 0.88, 0.05, 0.03]
        fills1 = [barra_conf(trk1[i], conf1[i], NARANJA_TERRACOTA if i == 1 else PAPEL_TAN)
                  for i in range(4)]
        pcts1 = [pct_texto(trk1[i], conf1[i], MARRON_OSCURO) for i in range(4)]
        pick1 = Dot(radius=0.07, color=NARANJA_TERRACOTA).move_to(rad1[1])

        self.play(*[GrowFromEdge(fills1[i], LEFT) for i in range(4)],
                  *[FadeIn(pcts1[i]) for i in range(4)], run_time=1.1)
        burb1 = bocadillo("¡Gigantes son, sin duda!", crab1, NARANJA_TERRACOTA)
        self.play(
            cajas1[1].animate.set_fill(NARANJA_TERRACOTA, opacity=0.22).set_stroke(NARANJA_TERRACOTA, width=2.5),
            rad1[1].animate.set_stroke(NARANJA_TERRACOTA, width=2.5),
            FadeIn(pick1, scale=0.5),
            FadeIn(burb1, shift=DOWN * 0.1),
        )
        self._siguiente()

        # revela: molinos era la correcta → falla → Loss alto
        check1 = Dot(radius=0.07, color=VERDE_OLIVA).move_to(rad1[0])
        burb1_fail = bocadillo("¿Molinos? ¡Válgame Dios!", crab1, ROJO_TOMATE)
        lf_fill = met_fill(3.2, ROJO_TOMATE)
        lf_num = met_num("3.2", ROJO_TOMATE)
        self.play(
            cajas1[0].animate.set_fill(VERDE_OLIVA, opacity=0.2).set_stroke(VERDE_OLIVA, width=3),
            fills1[0].animate.set_color(VERDE_OLIVA),
            rad1[0].animate.set_stroke(VERDE_OLIVA, width=2.5),
            FadeIn(check1, scale=0.5),
            cajas1[1].animate.set_fill(ROJO_TOMATE, opacity=0.15).set_stroke(ROJO_TOMATE, width=2.5),
            fills1[1].animate.set_color(ROJO_TOMATE),
            pick1.animate.set_color(ROJO_TOMATE),
            Transform(burb1, burb1_fail),
        )
        self.play(GrowFromEdge(lf_fill, DOWN), FadeIn(lf_num, shift=UP * 0.1),
                  Flash(met_track, color=ROJO_TOMATE, line_length=0.2))
        self._siguiente()

        # limpia la pregunta 1 (quedan crab y medidor)
        self.play(FadeOut(preg1, filas1, *fills1, *pcts1, pick1, check1))

        # ---- Pregunta 2: Quijote ACIERTA ----
        preg1b, filas1b, cajas1b, rad1b, trk1b = hacer_quiz(
            '"El caballo de Quijote es  ___"',
            [("A", "Rocinante"), ("B", "Babieca"), ("C", "Bucéfalo"), ("D", "Pegaso")],
        )
        preg1b.next_to(linea, DOWN, buff=0.9)
        filas1b.next_to(preg1b, DOWN, buff=0.34)

        burb1_preg2 = bocadillo("Esta la sé de sobra...", crab1, MARRON_OSCURO)
        self.play(FadeIn(preg1b, shift=DOWN * 0.1),
                  LaggedStart(*[FadeIn(f, shift=UP * 0.1) for f in filas1b], lag_ratio=0.12),
                  Transform(burb1, burb1_preg2), run_time=1.1)

        conf1b = [0.92, 0.05, 0.02, 0.01]
        fills1b = [barra_conf(trk1b[i], conf1b[i], VERDE_OLIVA if i == 0 else PAPEL_TAN)
                   for i in range(4)]
        pcts1b = [pct_texto(trk1b[i], conf1b[i], MARRON_OSCURO) for i in range(4)]
        pick1b = Dot(radius=0.07, color=VERDE_OLIVA).move_to(rad1b[0])

        self.play(*[GrowFromEdge(fills1b[i], LEFT) for i in range(4)],
                  *[FadeIn(pcts1b[i]) for i in range(4)], run_time=1.1)

        # acierta: Rocinante correcto → el Loss baja
        burb1_ok = bocadillo("¡Rocinante, voto a tal!", crab1, VERDE_OLIVA)
        ls_fill = met_fill(0.1, VERDE_OLIVA)
        ls_num = met_num("0.1", VERDE_OLIVA)
        self.play(
            cajas1b[0].animate.set_fill(VERDE_OLIVA, opacity=0.25).set_stroke(VERDE_OLIVA, width=3),
            rad1b[0].animate.set_stroke(VERDE_OLIVA, width=2.5),
            FadeIn(pick1b, scale=0.5),
            Transform(burb1, burb1_ok),
            Transform(lf_fill, ls_fill),
            Transform(lf_num, ls_num),
        )
        self.play(Flash(met_track, color=VERDE_OLIVA, line_length=0.2),
                  Indicate(lf_num, color=VERDE_OLIVA, scale_factor=1.4))
        self._siguiente()
        self.play(FadeOut(preg1b, filas1b, *fills1b, *pcts1b, pick1b, burb1,
                          met_track, met_lbl, lf_fill, lf_num),
                  FadeOut(crab1))


        # ── Acto 2: otro examen — Sancho y la perplejidad ──────────────
        preg2, filas2, cajas2, rad2, trk2 = hacer_quiz(
            '"Sancho Panza monta un  ___"',
            [("A", "burro"), ("B", "caballo"), ("C", "dragón"), ("D", "barco")],
        )
        preg2.next_to(linea, DOWN, buff=0.9)
        filas2.next_to(preg2, DOWN, buff=0.34)

        crab2 = ImageMobject(os.path.join("assets", "sancho_rust.png")).set_height(1.8)
        crab2.next_to(filas2, LEFT, buff=0.25)

        # termómetro de Perplejidad a la derecha
        PPL_H = 2.5
        ppl_track = RoundedRectangle(corner_radius=0.26, width=0.55, height=PPL_H,
                                     fill_color=CREMA_CALIDA, fill_opacity=1,
                                     stroke_color=MARRON_OSCURO, stroke_width=1.5)\
            .move_to([5.05, filas2.get_center()[1], 0])
        ppl_meter_lbl = Text("Perplejidad", font=FUENTE, font_size=14, weight=BOLD,
                             color=MARRON_OSCURO).next_to(ppl_track, UP, buff=0.15)

        def ppl_fill(frac, color):
            h = max(0.14, frac * PPL_H)
            f = RoundedRectangle(corner_radius=min(0.2, h / 2), width=0.42, height=h,
                                 fill_color=color, fill_opacity=1, stroke_width=0)
            f.move_to(ppl_track.get_bottom(), aligned_edge=DOWN).shift(UP * 0.06)
            return f

        def ppl_num(txt, color):
            return Text(txt, font="Monospace", font_size=20, weight=BOLD,
                        color=color).next_to(ppl_track, DOWN, buff=0.12)

        self.play(FadeIn(preg2, shift=DOWN * 0.1), FadeIn(crab2, shift=RIGHT * 0.2),
                  Create(ppl_track), FadeIn(ppl_meter_lbl))
        self.play(LaggedStart(*[FadeIn(f, shift=UP * 0.1) for f in filas2], lag_ratio=0.12),
                  run_time=1.0)

        conf_duda = [0.30, 0.27, 0.24, 0.19]
        fills2 = [barra_conf(trk2[i], conf_duda[i], NARANJA_TERRACOTA) for i in range(4)]
        pcts2 = [pct_texto(trk2[i], conf_duda[i], MARRON_OSCURO) for i in range(4)]
        burb2 = bocadillo("Pardiez, no sé cuál...", crab2, ROJO_TOMATE)
        pf_fill = ppl_fill(0.9, ROJO_TOMATE)
        pf_num = ppl_num("3360", ROJO_TOMATE)

        # seguridad repartida = mucha duda → perplejidad alta
        self.play(*[GrowFromEdge(fills2[i], LEFT) for i in range(4)],
                  *[FadeIn(pcts2[i]) for i in range(4)], run_time=1.1)
        self.play(GrowFromEdge(pf_fill, DOWN), FadeIn(pf_num, shift=UP * 0.1),
                  FadeIn(burb2, shift=DOWN * 0.1),
                  Flash(ppl_track, color=ROJO_TOMATE, line_length=0.2))
        self._siguiente()

        # se decide por burro → perplejidad baja
        conf_seg = [0.88, 0.06, 0.04, 0.02]
        fills2b = [barra_conf(trk2[i], conf_seg[i], VERDE_OLIVA if i == 0 else PAPEL_TAN)
                   for i in range(4)]
        pcts2b = [pct_texto(trk2[i], conf_seg[i], MARRON_OSCURO) for i in range(4)]
        pick2 = Dot(radius=0.07, color=VERDE_OLIVA).move_to(rad2[0])
        ps_fill = ppl_fill(0.13, VERDE_OLIVA)
        ps_num = ppl_num("17", VERDE_OLIVA)
        burb2_ok = bocadillo("¡Un burro, voto a tal!", crab2, VERDE_OLIVA)

        self.play(
            *[Transform(fills2[i], fills2b[i]) for i in range(4)],
            *[Transform(pcts2[i], pcts2b[i]) for i in range(4)],
            cajas2[0].animate.set_fill(VERDE_OLIVA, opacity=0.2).set_stroke(VERDE_OLIVA, width=2.5),
            rad2[0].animate.set_stroke(VERDE_OLIVA, width=2.5),
            FadeIn(pick2, scale=0.5),
            Transform(pf_fill, ps_fill),
            Transform(pf_num, ps_num),
            Transform(burb2, burb2_ok),
            run_time=1.4,
        )
        self.play(Flash(ppl_track, color=VERDE_OLIVA, line_length=0.2))
        self._siguiente()
        self.play(FadeOut(preg2, filas2, *fills2, *pcts2, pick2, burb2,
                          ppl_track, ppl_meter_lbl, pf_fill, pf_num),
                  FadeOut(crab2))


        def curva_loss(t):
            return 2.85 + 5.27 * np.exp(-t / 4200)

        ax = Axes(
            x_range=[0, 16000, 4000],
            y_range=[0, 9, 3],
            x_length=6.2,
            y_length=3.6,
            axis_config={"include_tip": False, "color": MARRON_OSCURO},
        ).shift(LEFT * 2.0 + DOWN * 0.5)

        x_lbl = Text("Pasos", font=FUENTE, font_size=13,
                     color=MARRON_OSCURO).next_to(ax, DOWN, buff=0.15)
        y_lbl = Text("Loss", font=FUENTE, font_size=13,
                     color=MARRON_OSCURO).next_to(ax, LEFT, buff=0.12)

        curva = ax.plot(curva_loss, x_range=[0, 16000],
                        color=NARANJA_TERRACOTA, stroke_width=4)

        self.play(Create(ax), Write(x_lbl), Write(y_lbl))
        self.play(Create(curva), run_time=2.0)

        checkpoints = [
            (0,     8.12, "3 360", ROJO_TOMATE,
             '"q7e llam8n p0r a#í\nF0rtun4 es una muj8r..."'),
            (4000,  4.45, "85",    PAPEL_TAN,
             '"Esta que llaman por\nahí Fortuna es una mujer..."'),
            (16000, 2.85, "17",    VERDE_OLIVA,
             '"Esta que llaman por ahí\nFortuna es una mujer borracha."'),
        ]

        dot_actual = None
        burbuja_actual = None

        for paso, loss_v, ppl_v, color, texto in checkpoints:
            pos_pt = ax.c2p(paso, curva_loss(paso))
            nuevo_dot = Dot(pos_pt, radius=0.12, color=color,
                            fill_opacity=1, stroke_color=BLANCO, stroke_width=1.5)

            rect_b = RoundedRectangle(corner_radius=0.16, width=4.8, height=1.62,
                                      fill_color=PAPEL_CREMA, fill_opacity=1,
                                      stroke_color=color, stroke_width=2.5)
            txt_b  = Text(texto, font=FUENTE, font_size=17, color=TINTA_NEGRA,
                          line_spacing=1.2)
            txt_b.scale_to_fit_width(rect_b.width - 0.55).move_to(rect_b)
            info_b = Text(f"Paso {paso:,}   Loss {loss_v}   PPL {ppl_v}",
                          font="Monospace", font_size=13, color=color
                          ).next_to(rect_b, DOWN, buff=0.1, aligned_edge=RIGHT)
            nueva_burbuja = VGroup(rect_b, txt_b, info_b).move_to(RIGHT * 3.2 + UP * 1.5)

            if dot_actual is None:
                self.play(FadeIn(nuevo_dot, scale=0.5))
                self.play(FadeIn(nueva_burbuja, shift=LEFT * 0.2))
            else:
                self.play(
                    ReplacementTransform(dot_actual, nuevo_dot),
                    FadeTransform(burbuja_actual, nueva_burbuja),
                    run_time=1.1,
                )
            dot_actual = nuevo_dot
            burbuja_actual = nueva_burbuja
            self._siguiente()

        adornos[1].clear_updaters()
        self.play(FadeOut(ax, x_lbl, y_lbl, curva, dot_actual, burbuja_actual))
        self.limpiar_pantalla()

