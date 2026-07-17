import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import numpy as np
import os
from colores import *
from objetos import *


class SlideMhaActo1Intuicion:
    def slide_mha_acto1_intuicion(self):
        titulo, linea = self.crear_titulo(
            "Atención",
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


        # ── dibujitos: la misma palabra "banco" según el contexto ──────
        def _arbol():
            tronco = Rectangle(width=0.09, height=0.24, fill_color=BARRO_MANCHEGO,
                               fill_opacity=1, stroke_width=0)
            copa = Circle(radius=0.25, fill_color=VERDE_OLIVA, fill_opacity=1,
                          stroke_color=MARRON_OSCURO, stroke_width=1.4)\
                .move_to(tronco.get_top() + UP * 0.17)
            return VGroup(tronco, copa)

        def _banquito():
            asiento = RoundedRectangle(corner_radius=0.03, width=0.5, height=0.07,
                                       fill_color=BARRO_MANCHEGO, fill_opacity=1,
                                       stroke_color=MARRON_OSCURO, stroke_width=1)
            respaldo = asiento.copy().shift(UP * 0.14)
            p1 = Rectangle(width=0.05, height=0.16, fill_color=BARRO_MANCHEGO,
                           fill_opacity=1, stroke_width=0)\
                .next_to(asiento, DOWN, buff=-0.02).align_to(asiento, LEFT).shift(RIGHT * 0.04)
            p2 = p1.copy().align_to(asiento, RIGHT).shift(LEFT * 0.04)
            return VGroup(respaldo, asiento, p1, p2)

        def icono_parque():
            arbol = _arbol()
            banco = _banquito().next_to(arbol, RIGHT, buff=0.12).align_to(arbol, DOWN)
            return VGroup(arbol, banco)

        def icono_dinero():
            monedas = VGroup()
            for k in range(3):
                c = Circle(radius=0.2, fill_color=ORO_VIEJO, fill_opacity=1,
                           stroke_color=MARRON_OSCURO, stroke_width=1.4)
                s = Text("$", font=FUENTE, font_size=16, color=MARRON_OSCURO,
                         weight=BOLD).move_to(c)
                monedas.add(VGroup(c, s).shift(UP * 0.12 * k + RIGHT * 0.04 * k))
            return monedas

        def _pez(col):
            cuerpo = Ellipse(width=0.5, height=0.26, fill_color=col, fill_opacity=1,
                             stroke_color=MARRON_OSCURO, stroke_width=1.4)
            cola = Polygon([0, 0, 0], [0.16, 0.12, 0], [0.16, -0.12, 0],
                           fill_color=col, fill_opacity=1,
                           stroke_color=MARRON_OSCURO, stroke_width=1.4)\
                .next_to(cuerpo, LEFT, buff=-0.03)
            ojo = Dot(radius=0.028, color=MARRON_OSCURO)\
                .move_to(cuerpo.get_center() + RIGHT * 0.13)
            return VGroup(cola, cuerpo, ojo)

        def icono_peces():
            p1 = _pez(NARANJA_TERRACOTA)
            p2 = _pez(PAPEL_TAN).scale(0.78).next_to(p1, UR, buff=-0.02)
            p3 = _pez(MARRON_OSCURO).scale(0.68).next_to(p1, DR, buff=0.02)
            return VGroup(p1, p2, p3)

        ejemplos = [
            ("parque", NARANJA_TERRACOTA, icono_parque),
            ("dinero", MARRON_OSCURO, icono_dinero),
            ("peces", VERDE_OLIVA, icono_peces),
        ]
        tarjetas = VGroup()
        for etiqueta, color, mk_icon in ejemplos:
            caja = RoundedRectangle(
                corner_radius=0.2, width=3.0, height=2.4,
                fill_color=PAPEL_CREMA, fill_opacity=1,
                stroke_color=color, stroke_width=2.5
            )
            pal = Text("banco", font=FUENTE, font_size=26, weight=BOLD, color=color)\
                .move_to(caja.get_top() + DOWN * 0.42)
            icon = mk_icon()
            icon.scale_to_fit_height(0.95).move_to(caja.get_center() + DOWN * 0.02)
            etq = Text(etiqueta, font=FUENTE, font_size=17, color=TINTA_NEGRA)\
                .move_to(caja.get_bottom() + UP * 0.32)
            tarjetas.add(VGroup(caja, pal, icon, etq))

        tarjetas.arrange(RIGHT, buff=0.5).next_to(respuesta, DOWN, buff=0.55)
        self.play(
            LaggedStart(*[FadeIn(t, shift=UP * 0.2, scale=0.95) for t in tarjetas],
                        lag_ratio=0.22),
            run_time=1.6
        )
        for t in tarjetas:
            self.play(Indicate(t[2], scale_factor=1.2, color=NARANJA_TERRACOTA), run_time=0.4)
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


        # ── arcos de atención: "este" señala a hidalgo y, sobre todo, a "gigante,"
        pesos = {1: 0.08, 4: 0.85}  # hidalgo, gigante
        este_top = oracion[6].get_top() + UP * 0.06

        def arco_atencion(idx, w, fuerte):
            dst = oracion[idx].get_top() + UP * 0.06
            dx = abs(dst[0] - este_top[0])
            mid = (este_top + dst) / 2 + UP * (0.55 + 0.13 * dx)
            m = VMobject(
                stroke_color=NARANJA_TERRACOTA if fuerte else PAPEL_TAN,
                stroke_width=2 + w * 11,
                stroke_opacity=0.9 if fuerte else 0.5,
            )
            m.set_points_smoothly([este_top, mid, dst])
            return m

        arcos_debiles = VGroup(*[arco_atencion(i, w, False)
                                 for i, w in pesos.items() if i != 4])
        arco_fuerte = arco_atencion(4, pesos[4], True)
        peso_gig_lbl = Text("0.85", font=FUENTE, font_size=22, color=NARANJA_TERRACOTA,
                            weight=BOLD).move_to(arco_fuerte.get_top() + UP * 0.2)

        # primero, atención tenue repartida entre varias palabras
        self.play(LaggedStart(*[Create(a) for a in arcos_debiles], lag_ratio=0.18),
                  run_time=1.1)
        # luego, la atención se concentra en "gigante,"
        self.play(Create(arco_fuerte), run_time=0.7)
        self.play(
            FadeIn(peso_gig_lbl, shift=UP * 0.1),
            oracion[4].animate.set_color(NARANJA_TERRACOTA).scale(1.12),
            Flash(oracion[4], color=NARANJA_TERRACOTA, line_length=0.18, num_lines=9),
            run_time=0.8,
        )
        self._siguiente()

        self.play(FadeOut(VGroup(arcos_debiles, arco_fuerte, peso_gig_lbl)))
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


