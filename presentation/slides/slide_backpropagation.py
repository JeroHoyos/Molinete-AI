import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import numpy as np
import os
from colores import *
from objetos import *


class SlideBackpropagation:
    def slide_backpropagation(self):
        titulo, linea = self.crear_titulo(
            "Backpropagation",
            palabra_clave="Backpropagation",
            color_clave=NARANJA_TERRACOTA
        )
        llanuras_fondo = crear_llanuras_manchegas()
        adornos = self._crear_adornos_esquinas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo, adornos=adornos)

        # ══ ACTO 0: el objetivo, un gradiente por parametro ═════════════
        grads_fila = MathTex(
            r"\frac{\partial L}{\partial w_1},\;\;"
            r"\frac{\partial L}{\partial w_2},\;\;"
            r"\frac{\partial L}{\partial w_3},\;\;\dots,\;\;"
            r"\frac{\partial L}{\partial w_n}",
            font_size=44, color=TINTA_NEGRA,
        ).move_to(UP * 0.7)

        self.play(Write(grads_fila), run_time=1.2)
        self._siguiente()

        # ── de golpe: una mega ecuacion tachada ─────────────────────────
        mega = MathTex(
            r"\frac{\partial L}{\partial w}"
            r"=\frac{\partial}{\partial w}\,"
            r"\ell\Big(f_3\big(f_2\left(f_1(x \cdot w + b)\right)\big)\Big)",
            font_size=42, color=TINTA_NEGRA,
        ).move_to(DOWN * 0.6)

        self.play(TransformMatchingShapes(grads_fila, mega), run_time=0.9)
        self.play(Wiggle(mega, scale_value=1.06), run_time=0.9)

        tachon = Line(
            mega.get_corner(DL) + DL * 0.15,
            mega.get_corner(UR) + UR * 0.15,
            stroke_color=ROJO_TOMATE, stroke_width=6,
        )
        self.play(Create(tachon), run_time=0.7)
        self._siguiente()

        # ══ ACTO 1: paso a paso con la regla de la cadena ═══════════════
        self.play(FadeOut(mega), FadeOut(tachon), run_time=0.6)

        def _caja_pipeline(tex, color_borde=MARRON_OSCURO):
            caja = RoundedRectangle(
                corner_radius=0.14, width=1.1, height=0.85,
                fill_color=FONDO_CAJA, fill_opacity=1,
                stroke_color=color_borde, stroke_width=2.2,
            )
            lbl = MathTex(tex, font_size=34, color=TINTA_NEGRA).move_to(caja)
            return VGroup(caja, lbl)

        def _flecha_pipeline():
            return Arrow(
                ORIGIN, RIGHT * 0.65, buff=0,
                color=MARRON_OSCURO, stroke_width=3,
                max_tip_length_to_length_ratio=0.35,
            )

        x_in      = MathTex("x", font_size=38, color=TINTA_NEGRA)
        caja_f1   = _caja_pipeline("f_1")
        caja_f2   = _caja_pipeline("f_2")
        caja_f3   = _caja_pipeline("f_3")
        caja_loss = _caja_pipeline(r"\ell", color_borde=ROJO_TOMATE)
        L_out     = MathTex("L", font_size=42, color=ROJO_TOMATE)

        fl0, fl1, fl2, fl3, fl4 = [_flecha_pipeline() for _ in range(5)]

        fila_pipe = VGroup(
            x_in, fl0, caja_f1, fl1, caja_f2, fl2, caja_f3, fl3, caja_loss, fl4, L_out
        ).arrange(RIGHT, buff=0.28).move_to(UP * 1.4)

        h_lbls = VGroup(*[
            MathTex(h, font_size=26, color=MARRON_OSCURO).next_to(fl, UP, buff=0.12)
            for h, fl in [("h_1", fl1), ("h_2", fl2), ("h_3", fl3)]
        ])

        # el peso w vive dentro de f1
        w_circulo = Circle(radius=0.2, fill_color=NARANJA_CLARO, fill_opacity=1,
                           stroke_color=NARANJA_TERRACOTA, stroke_width=2.2)
        w_lbl  = MathTex("w", font_size=28, color=NARANJA_TERRACOTA).move_to(w_circulo)
        w_chip = VGroup(w_circulo, w_lbl).next_to(caja_f1, DOWN, buff=0.22)

        self.play(
            LaggedStart(*[
                FadeIn(m, shift=RIGHT * 0.15)
                for m in [x_in, fl0, caja_f1, fl1, caja_f2, fl2, caja_f3, fl3, caja_loss]
            ], lag_ratio=0.12),
            FadeIn(h_lbls),
            FadeIn(w_chip, shift=UP * 0.1),
            run_time=1.3,
        )

        # forward: la señal viaja hacia la derecha en verde
        segmentos_fwd = [(fl0, caja_f1), (fl1, caja_f2), (fl2, caja_f3), (fl3, caja_loss)]
        for fl, caja in segmentos_fwd:
            self.play(
                ShowPassingFlash(
                    Line(fl.get_start(), fl.get_end()).set_stroke(VERDE_OLIVA, 5),
                    time_width=0.8,
                ),
                caja[0].animate.set_fill(MENTA_PALIDA, 0.7),
                run_time=0.45,
            )
        self.play(
            GrowArrow(fl4),
            Write(L_out),
            caja_loss[0].animate.set_fill(SALMON_CLARO, 0.8),
            Flash(L_out, color=ROJO_TOMATE, line_length=0.25, num_lines=10),
            run_time=0.8,
        )
        self.play(
            Circumscribe(w_chip, color=NARANJA_TERRACOTA, buff=0.08),
            Indicate(L_out, color=ROJO_TOMATE, scale_factor=1.25),
        )
        self._siguiente()

        # la regla de la cadena se construye al paso del gradiente
        eq = MathTex(
            r"\frac{\partial L}{\partial w}", "=",
            r"\frac{\partial L}{\partial h_3}", r"\cdot",
            r"\frac{\partial h_3}{\partial h_2}", r"\cdot",
            r"\frac{\partial h_2}{\partial h_1}", r"\cdot",
            r"\frac{\partial h_1}{\partial w}",
            font_size=38, color=TINTA_NEGRA,
        ).move_to(DOWN * 0.9)
        eq[0].set_color(NARANJA_TERRACOTA)

        marcas = VGroup()
        for paso, idx in enumerate([2, 4, 6, 8], start=1):
            c = Circle(radius=0.16, color=NARANJA_TERRACOTA, fill_color=PAPEL_CREMA,
                       fill_opacity=1, stroke_width=2)
            num = Text(str(paso), font=FUENTE, font_size=13, weight=BOLD,
                       color=NARANJA_TERRACOTA).move_to(c)
            marcas.add(VGroup(c, num).next_to(eq[idx], DOWN, buff=0.32))

        self.play(Write(eq[0]), Write(eq[1]), run_time=0.7)

        # el pulso terracota recorre el pipeline hacia atras y cada caja
        # que cruza deja su derivada local en la formula
        pasos_bwd = [
            (fl3, caja_loss, 2),
            (fl2, caja_f3,   4),
            (fl1, caja_f2,   6),
            (fl0, caja_f1,   8),
        ]
        for k, (fl, caja, idx) in enumerate(pasos_bwd):
            anims = [
                ShowPassingFlash(
                    Line(fl.get_end(), fl.get_start()).set_stroke(NARANJA_TERRACOTA, 6),
                    time_width=0.8,
                ),
                caja[0].animate.set_stroke(NARANJA_TERRACOTA, 3),
                TransformFromCopy(caja, eq[idx], path_arc=0.4),
                FadeIn(marcas[k], shift=UP * 0.1),
            ]
            if k > 0:
                anims.append(FadeIn(eq[idx - 1]))
            self.play(*anims, run_time=0.85)
        self._siguiente()

        self.play(
            FadeOut(fila_pipe), FadeOut(h_lbls), FadeOut(w_chip),
            FadeOut(eq), FadeOut(marcas),
            run_time=0.7,
        )

        # ══ ACTO 2: la red completa y el zoom a una neurona ═════════════
        capas  = [3, 4, 4, 2]
        xs     = [-4.7, -1.57, 1.57, 4.7]
        base_y = -0.2

        nodos = []
        for li, n in enumerate(capas):
            col = VGroup(*[
                Circle(radius=0.27, fill_color=FONDO_CAJA, fill_opacity=1,
                       stroke_color=MARRON_OSCURO, stroke_width=2.5)
                for _ in range(n)
            ]).arrange(DOWN, buff=0.42).move_to([xs[li], base_y, 0])
            nodos.append(col)
        todos_nodos = [nd for col in nodos for nd in col]

        edges_fwd = []
        for li in range(len(capas) - 1):
            ef = VGroup()
            for a in nodos[li]:
                for b in nodos[li + 1]:
                    ef.add(Line(a.get_right(), b.get_left(), stroke_width=1.6,
                                color=MARRON_OSCURO, z_index=-1).set_opacity(0.25))
            edges_fwd.append(ef)
        edges_fwd_all = VGroup(*edges_fwd)

        lbl_in = Text("entrada", font=FUENTE, font_size=15, weight=BOLD,
                      color=MARRON_OSCURO).next_to(nodos[0], DOWN, buff=0.3)
        lbl_out = Text("salida", font=FUENTE, font_size=15, weight=BOLD,
                       color=MARRON_OSCURO).next_to(nodos[-1], DOWN, buff=0.3)

        self.play(LaggedStart(*[GrowFromCenter(nd) for nd in todos_nodos],
                              lag_ratio=0.04), run_time=1.1)
        self.play(Create(edges_fwd_all), FadeIn(lbl_in), FadeIn(lbl_out), run_time=0.9)

        # forward: verde, capa a capa
        self.play(*[n.animate.set_fill(VERDE_OLIVA, 0.55) for n in nodos[0]], run_time=0.35)
        for i in range(len(capas) - 1):
            self.play(*[ShowPassingFlash(e.copy().set_stroke(VERDE_OLIVA, 4.5), time_width=0.6)
                        for e in edges_fwd[i]], run_time=0.55)
            self.play(*[n.animate.set_fill(VERDE_OLIVA, 0.55) for n in nodos[i + 1]],
                      run_time=0.3)

        # prediccion erronea: el termometro de Loss se dispara
        MET_H = 2.2
        term_track = RoundedRectangle(corner_radius=0.26, width=0.55, height=MET_H,
                                      fill_color=CREMA_CALIDA, fill_opacity=1,
                                      stroke_color=MARRON_OSCURO, stroke_width=1.5)\
            .next_to(nodos[-1], RIGHT, buff=0.6)
        term_lbl = Text("Loss", font=FUENTE, font_size=16, weight=BOLD,
                        color=MARRON_OSCURO).next_to(term_track, UP, buff=0.15)
        termometro = VGroup(term_track, term_lbl)

        def term_fill(frac, color):
            h = max(0.14, frac * (MET_H - 0.12))
            f = RoundedRectangle(corner_radius=min(0.2, h / 2), width=0.42, height=h,
                                 fill_color=color, fill_opacity=1, stroke_width=0)
            f.move_to(term_track.get_bottom(), aligned_edge=DOWN).shift(UP * 0.06)
            return f

        fill_loss = term_fill(0.9, ROJO_TOMATE)

        self.play(Create(term_track), FadeIn(term_lbl), run_time=0.6)
        self.play(
            *[Indicate(n, color=ROJO_TOMATE, scale_factor=1.15) for n in nodos[-1]],
            GrowFromEdge(fill_loss, DOWN),
        )
        self._siguiente()

        # ── zoom: una neurona, sin marco, como acercando la camara ──────
        foco = nodos[1][1]
        n_dest = len(nodos[1])
        edges_foco = VGroup(*[edges_fwd[0][a * n_dest + 1] for a in range(len(nodos[0]))])
        resto_edges = VGroup(*[e for capa in edges_fwd for e in capa
                               if e not in edges_foco.submobjects])
        resto_nodos = VGroup(*[n for n in todos_nodos if n is not foco])

        self.play(
            resto_edges.animate.set_opacity(0.08),
            resto_nodos.animate.set_opacity(0.18),
            VGroup(lbl_in, lbl_out).animate.set_opacity(0.2),
            VGroup(termometro, fill_loss).animate.set_opacity(0.25),
            edges_foco.animate.set_stroke(NARANJA_TERRACOTA, 3).set_opacity(0.9),
            foco.animate.set_stroke(NARANJA_TERRACOTA, 3.5),
            run_time=0.8,
        )
        self.play(Indicate(foco, color=NARANJA_TERRACOTA, scale_factor=1.35),
                  run_time=0.6)

        # ── el grafo conceptual: entradas, perillas, x, +, ReLU ─────────
        def _perilla(nombre, angulo0=0.0):
            radio = 0.27
            cuerpo = Circle(radius=radio, fill_color=NARANJA_CLARO, fill_opacity=0.6,
                            stroke_color=NARANJA_TERRACOTA, stroke_width=2.6)
            muescas = VGroup()
            for ang in np.linspace(210, -30, 7) * DEGREES:
                d = np.array([np.cos(ang), np.sin(ang), 0])
                muescas.add(Line(d * radio * 1.1, d * radio * 1.28,
                                 stroke_width=1.6, stroke_color=NARANJA_TERRACOTA))
            aguja = Line(ORIGIN, UP * radio * 0.78, stroke_width=3.4,
                         color=MARRON_OSCURO).rotate(angulo0, about_point=ORIGIN)
            eje = Dot(radius=0.04, color=MARRON_OSCURO)
            knob = VGroup(cuerpo, muescas, aguja, eje)
            lbl = MathTex(nombre, font_size=28, color=NARANJA_TERRACOTA)\
                .next_to(knob, LEFT, buff=0.14)
            return VGroup(knob, lbl), aguja, cuerpo

        def _nodo_op(simbolo, radio=0.28):
            circ = Circle(radius=radio, fill_color=FONDO_CAJA, fill_opacity=1,
                          stroke_color=MARRON_OSCURO, stroke_width=2.4)
            s = MathTex(simbolo, font_size=34, color=ROJO_TOMATE).move_to(circ)
            return VGroup(circ, s)

        def _linea_hacia(circ, p_src):
            centro = circ.get_center()
            v = centro - p_src
            v = v / np.linalg.norm(v)
            return Line(p_src, centro - v * (circ.radius + 0.04),
                        stroke_width=2.0, color=MARRON_OSCURO)

        filas_y = [1.05, -0.3, -1.65]
        X_IN, X_KNOB, X_MULT = -5.2, -4.15, -2.4

        xs_lbls, perillas, agujas, cuerpos, mults = [], [], [], [], []
        lineas_in = VGroup()
        angulos0 = [-35 * DEGREES, 20 * DEGREES, 50 * DEGREES]
        for i, fy in enumerate(filas_y):
            x_lbl = MathTex(f"x_{i + 1}", font_size=34, color=TINTA_NEGRA)\
                .move_to([X_IN, fy + 0.42, 0])
            per, aguja, cuerpo = _perilla(f"w_{i + 1}", angulos0[i])
            per[0].move_to([X_KNOB, fy - 0.32, 0])
            per[1].next_to(per[0], LEFT, buff=0.14)
            mult = _nodo_op(r"\times").move_to([X_MULT, fy, 0])
            lineas_in.add(_linea_hacia(mult[0], x_lbl.get_right() + RIGHT * 0.08))
            lineas_in.add(_linea_hacia(mult[0], cuerpo.get_center()
                                       + RIGHT * (cuerpo.radius + 0.05)))
            xs_lbls.append(x_lbl)
            perillas.append(per)
            agujas.append(aguja)
            cuerpos.append(cuerpo)
            mults.append(mult)

        suma = _nodo_op("+", radio=0.32).move_to([-0.5, -0.3, 0])
        lineas_suma = VGroup(*[
            _linea_hacia(suma[0], m.get_center()) for m in mults
        ])

        per_b, aguja_b, cuerpo_b = _perilla("b", 0.0)
        per_b[0].move_to([-0.5, -2.0, 0])
        per_b[1].next_to(per_b[0], LEFT, buff=0.14)
        linea_b = _linea_hacia(suma[0], cuerpo_b.get_center() + UP * (cuerpo_b.radius + 0.05))

        relu_bg = RoundedRectangle(corner_radius=0.12, width=1.35, height=0.68,
                                   fill_color=FONDO_CAJA, fill_opacity=1,
                                   stroke_color=MARRON_OSCURO, stroke_width=2.4)\
            .move_to([1.9, -0.3, 0])
        relu_txt = Text("ReLU", font=FUENTE, font_size=19, weight=BOLD,
                        color=TINTA_NEGRA).move_to(relu_bg)
        relu = VGroup(relu_bg, relu_txt)

        chip_L_bg = RoundedRectangle(corner_radius=0.14, width=1.3, height=0.95,
                                     fill_color=FONDO_CAJA, fill_opacity=1,
                                     stroke_color=ROJO_TOMATE, stroke_width=2.5)\
            .move_to([4.7, -0.3, 0])
        chip_L_txt = MathTex("L", font_size=44, color=ROJO_TOMATE).move_to(chip_L_bg)
        chip_L = VGroup(chip_L_bg, chip_L_txt)

        fl_relu = Arrow(suma.get_right(), relu_bg.get_left(), buff=0.08,
                        color=MARRON_OSCURO, stroke_width=3,
                        max_tip_length_to_length_ratio=0.18)
        fl_L = Arrow(relu_bg.get_right(), chip_L_bg.get_left(), buff=0.08,
                     color=MARRON_OSCURO, stroke_width=3,
                     max_tip_length_to_length_ratio=0.12)

        detalle = VGroup(
            lineas_in, lineas_suma, linea_b,
            *xs_lbls, *perillas, per_b, *mults, suma, relu, chip_L, fl_relu, fl_L,
        )

        # zoom de camara: la red se agranda hacia el espectador y se funde
        # con el detalle de la neurona, que crece desde el punto focal
        pz = VGroup(nodos[0], foco).get_center()
        red_todo = VGroup(edges_fwd_all, *nodos, lbl_in, lbl_out,
                          termometro, fill_loss)
        self.play(
            red_todo.animate.scale(3.4, about_point=pz).fade(1),
            FadeIn(detalle, scale=0.25, shift=detalle.get_center() - pz),
            run_time=1.3,
        )
        self.remove(edges_fwd_all, *todos_nodos, lbl_in, lbl_out,
                    term_track, term_lbl, fill_loss)
        red_todo.scale(1 / 3.4, about_point=pz)

        # las perillas se presentan girando
        todas_agujas  = [*agujas, aguja_b]
        todos_cuerpos = [*cuerpos, cuerpo_b]
        self.play(
            *[Rotate(a, 30 * DEGREES, about_point=c.get_center(),
                     rate_func=rate_functions.wiggle)
              for a, c in zip(todas_agujas, todos_cuerpos)],
            run_time=1.1,
        )

        # forward rapido en verde para orientar
        todas_lineas = [*lineas_in, *lineas_suma, linea_b, fl_relu, fl_L]
        self.play(
            *[ShowPassingFlash(
                Line(l.get_start(), l.get_end()).set_stroke(VERDE_OLIVA, 4.5),
                time_width=0.7)
              for l in todas_lineas],
            run_time=1.0,
        )
        self.play(Flash(chip_L, color=ROJO_TOMATE, line_length=0.3, num_lines=12),
                  run_time=0.5)
        self._siguiente()

        # ── backward: el gradiente va hacia atras y, al pasar por cada
        # parametro, saca su derivada y ajusta la perilla ────────────────
        self.play(Flash(chip_L, color=NARANJA_TERRACOTA, line_length=0.35, num_lines=14),
                  run_time=0.5)

        def _flash_atras(linea):
            return ShowPassingFlash(
                Line(linea.get_end(), linea.get_start())
                .set_stroke(NARANJA_TERRACOTA, 5.5),
                time_width=0.7,
            )

        def _tag_grad(tex, color_borde=NARANJA_TERRACOTA):
            t = MathTex(*tex, font_size=24, color=NARANJA_TERRACOTA)
            bg = SurroundingRectangle(
                t, corner_radius=0.08, buff=0.09,
                fill_color=FONDO_CAJA, fill_opacity=0.96,
                stroke_color=color_borde, stroke_width=1.5,
            )
            return VGroup(bg, t)

        # tramo compartido: de L a la suma
        self.play(_flash_atras(fl_L),
                  relu_bg.animate.set_stroke(NARANJA_TERRACOTA, 3), run_time=0.55)
        self.play(_flash_atras(fl_relu),
                  suma[0].animate.set_stroke(NARANJA_TERRACOTA, 3), run_time=0.55)

        giros = [-80 * DEGREES, 60 * DEGREES, -45 * DEGREES, 40 * DEGREES]
        nombres = ["w_1", "w_2", "w_3", "b"]

        def _saca_y_ajusta(tag_pos, nom, aguja, cuerpo, giro, flashes, marcas_nodo):
            # la derivada aparece al paso del pulso
            tag = _tag_grad([rf"\frac{{\partial L}}{{\partial {nom}}}"])\
                .move_to(tag_pos)
            self.play(
                LaggedStart(*flashes, lag_ratio=0.45),
                *marcas_nodo,
                run_time=0.7,
            )
            self.play(FadeIn(tag, scale=0.6), run_time=0.35)

            # se vuelve negativa y se absorbe en la perilla, que gira
            tag_menos = _tag_grad(
                ["-", rf"\frac{{\partial L}}{{\partial {nom}}}"],
                color_borde=ROJO_TOMATE,
            )
            tag_menos[1][0].set_color(ROJO_TOMATE).scale(1.4)
            tag_menos.move_to(tag)
            self.play(Transform(tag, tag_menos), run_time=0.3)
            self.play(
                tag.animate.scale(0.15).move_to(cuerpo.get_center()).fade(1),
                Rotate(aguja, giro, about_point=cuerpo.get_center(),
                       rate_func=rate_functions.ease_out_back),
                cuerpo.animate.set_stroke(VERDE_OLIVA, 2.8),
                run_time=0.6,
            )
            self.remove(tag)

        for i in range(3):
            linea_w = lineas_in[2 * i + 1]
            medio = (perillas[i][0].get_center() + mults[i].get_center()) / 2
            _saca_y_ajusta(
                medio + DOWN * 0.42, nombres[i],
                agujas[i], cuerpos[i], giros[i],
                [_flash_atras(lineas_suma[i]), _flash_atras(linea_w)],
                [mults[i][0].animate.set_stroke(NARANJA_TERRACOTA, 3)],
            )
        _saca_y_ajusta(
            cuerpo_b.get_center() + RIGHT * 1.15, "b",
            aguja_b, cuerpo_b, giros[3],
            [_flash_atras(linea_b)],
            [],
        )

        self.play(
            *[Flash(c, color=VERDE_OLIVA, line_length=0.18, num_lines=8)
              for c in todos_cuerpos],
            run_time=0.6,
        )
        self._siguiente()

        # ── zoom out: la red vuelve, el Loss sigue alto ─────────────────
        for e in [*resto_edges, *edges_foco]:
            e.set_stroke(MARRON_OSCURO, width=1.6).set_opacity(0.25)
        for n in todos_nodos:
            n.set_stroke(MARRON_OSCURO, 2.5, opacity=1).set_fill(VERDE_OLIVA, 0.55)
        VGroup(lbl_in, lbl_out).set_opacity(1)
        termometro.set_opacity(1)
        fill_loss.set_opacity(1)

        self.play(FadeOut(detalle), run_time=0.6)
        self.play(
            FadeIn(resto_edges), FadeIn(edges_foco),
            *[FadeIn(n) for n in todos_nodos],
            FadeIn(lbl_in), FadeIn(lbl_out),
            FadeIn(termometro), FadeIn(fill_loss),
            run_time=0.9,
        )
        self._siguiente()

        # update capa por capa, de atras hacia adelante: restar el gradiente
        for i in reversed(range(len(capas) - 1)):
            chip_nabla = MathTex(r"-\nabla w", font_size=32,
                                 color=NARANJA_TERRACOTA)\
                .move_to([(xs[i] + xs[i + 1]) / 2, 2.0, 0])
            self.play(
                *[ShowPassingFlash(
                    Line(e.get_end(), e.get_start())
                    .set_stroke(NARANJA_TERRACOTA, 4),
                    time_width=0.6)
                  for e in edges_fwd[i]],
                *[e.animate.set_stroke(VERDE_OLIVA, 2.0).set_opacity(0.45)
                  for e in edges_fwd[i]],
                FadeIn(chip_nabla, shift=DOWN * 0.2),
                run_time=0.7,
            )
            self.play(FadeOut(chip_nabla), run_time=0.2)

        # forward otra vez: ahora la prediccion mejora y el Loss cae
        for i in range(len(capas) - 1):
            self.play(
                *[ShowPassingFlash(e.copy().set_stroke(VERDE_OLIVA, 4.5),
                                   time_width=0.6)
                  for e in edges_fwd[i]],
                run_time=0.45,
            )
        fill_bajo = term_fill(0.16, VERDE_OLIVA)
        self.play(
            Transform(fill_loss, fill_bajo),
            *[Indicate(n, color=VERDE_OLIVA, scale_factor=1.15) for n in nodos[-1]],
            run_time=1.0,
        )
        self.play(Flash(term_track, color=VERDE_OLIVA, line_length=0.3, num_lines=12),
                  run_time=0.5)
        self._siguiente()

        self.limpiar_pantalla()
