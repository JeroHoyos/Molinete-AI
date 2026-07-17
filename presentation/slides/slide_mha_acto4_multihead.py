import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import numpy as np
import os
from colores import *
from objetos import *


class SlideMhaActo4Multihead:
    def slide_mha_acto4_multihead(self):

        titulo, linea = self.crear_titulo(
            "Multi-Head Self-Attention",
            palabra_clave="Multi-Head",
            color_clave=NARANJA_TERRACOTA
        )
        llanuras_fondo = crear_llanuras_manchegas()
        adornos = self._crear_adornos_esquinas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo, adornos=adornos)


        subtitulo = Text(
            "¿Por qué multi-head?",
            font=FUENTE, font_size=28, color=MARRON_OSCURO
        ).next_to(linea, DOWN, buff=0.4)

        self.play(FadeIn(subtitulo, shift=DOWN))


        etiqueta_vec = Text(
            "Vector de Embedding (768 dimensiones)",
            font=FUENTE, font_size=22, color=MARRON_OSCURO
        ).next_to(subtitulo, DOWN, buff=0.4)

        vector = Rectangle(
            width=10, height=0.75,
            fill_color=PAPEL_CREMA, fill_opacity=1,
            stroke_color=MARRON_OSCURO, stroke_width=2
        ).next_to(etiqueta_vec, DOWN, buff=0.2)

        self.play(FadeIn(etiqueta_vec), FadeIn(vector))


        colores_h = [NARANJA_TERRACOTA, MARRON_OSCURO, PAPEL_TAN, NARANJA_CLARO] * 3

        cabezas = VGroup(*[
            Rectangle(
                width=10/12, height=1.1,
                fill_color=colores_h[i], fill_opacity=0.9,
                stroke_color=PAPEL_CREMA, stroke_width=1.5
            )
            for i in range(12)
        ]).arrange(RIGHT, buff=0).move_to(vector.get_center())

        etiqueta_h = Text(
            "12 cabezas independientes",
            font=FUENTE, font_size=22, color=MARRON_OSCURO
        ).move_to(etiqueta_vec.get_center())

        self.play(ReplacementTransform(vector, cabezas), Transform(etiqueta_vec, etiqueta_h))
        self.play(cabezas.animate.arrange(RIGHT, buff=0.1).move_to(cabezas.get_center()))
        self.play(
            LaggedStart(*[Indicate(c, scale_factor=1.1, color=PAPEL_CREMA) for c in cabezas], lag_ratio=0.08),
            run_time=1.5
        )


        # ── cada cabeza propone su propia reorganización del espacio ────
        self.play(
            FadeOut(subtitulo), FadeOut(etiqueta_vec),
            cabezas.animate.scale(0.62).next_to(linea, DOWN, buff=0.3),
            run_time=0.8,
        )
        cap1 = Text("Cada cabeza propone su propia reorganización del espacio",
                    font=FUENTE, font_size=23, color=MARRON_OSCURO,
                    t2c={"propia reorganización": NARANJA_TERRACOTA})\
            .next_to(cabezas, DOWN, buff=0.3)
        self.play(FadeIn(cap1, shift=DOWN * 0.1))

        SC = 0.6
        # tokens (palabras) en coordenadas del espacio; cada cabeza los recoloca
        tokens = [(np.array([0.6, 0.7]), NARANJA_TERRACOTA),
                  (np.array([1.9, 1.5]), VERDE_OLIVA),
                  (np.array([1.3, 2.1]), AZUL_NOCHE)]

        def T(M, p, ctr):
            p = np.asarray(p, dtype=float)
            q = SC * np.array([M[0, 0] * p[0] + M[0, 1] * p[1],
                               M[1, 0] * p[0] + M[1, 1] * p[1], 0.0])
            return ctr + q

        def head_space(M, tint, ctr):
            corners = [[0, 0], [2.4, 0], [2.4, 2.4], [0, 2.4]]
            poly = Polygon(*[T(M, c, ctr) for c in corners],
                           fill_color=tint, fill_opacity=0.16,
                           stroke_color=tint, stroke_width=1.6)
            grid = VGroup()
            for u in (0.8, 1.6):
                grid.add(Line(T(M, [u, 0], ctr), T(M, [u, 2.4], ctr),
                              stroke_color=PAPEL_TAN, stroke_width=0.8, stroke_opacity=0.55))
            for v in (0.8, 1.6):
                grid.add(Line(T(M, [0, v], ctr), T(M, [2.4, v], ctr),
                              stroke_color=PAPEL_TAN, stroke_width=0.8, stroke_opacity=0.55))
            ax = Arrow(T(M, [0, 0], ctr), T(M, [2.75, 0], ctr), buff=0, color=MARRON_OSCURO,
                       stroke_width=2.4, max_tip_length_to_length_ratio=0.06)
            ay = Arrow(T(M, [0, 0], ctr), T(M, [0, 2.75], ctr), buff=0, color=MARRON_OSCURO,
                       stroke_width=2.4, max_tip_length_to_length_ratio=0.06)
            toks = VGroup(*[
                Dot(T(M, pos, ctr), radius=0.075, color=c).set_stroke(MARRON_OSCURO, width=1.2)
                for pos, c in tokens
            ])
            return VGroup(poly, grid, ax, ay, toks)

        Ms = [
            np.array([[1.02, 0.28], [0.05, 0.95]]),
            np.array([[0.95, -0.36], [0.24, 1.02]]),
            np.array([[0.80, 0.46], [-0.30, 0.90]]),
        ]
        tints = [NARANJA_TERRACOTA, VERDE_OLIVA, AZUL_NOCHE]
        centers = [np.array([-4.2, -1.3, 0]), np.array([-0.6, -1.3, 0]), np.array([3.0, -1.3, 0])]
        labels_txt = ["Cabeza 1", "Cabeza 2", "Cabeza h"]

        heads = [head_space(M, t, c) for M, t, c in zip(Ms, tints, centers)]
        head_lbls = VGroup(*[
            Text(labels_txt[k], font=FUENTE, font_size=18, color=tints[k], weight=BOLD)
            .next_to(heads[k], DOWN, buff=0.18) for k in range(3)
        ])
        dots3 = Text("· · ·", font=FUENTE, font_size=34, color=MARRON_OSCURO)\
            .move_to((heads[1].get_center() + heads[2].get_center()) / 2)

        self.play(LaggedStart(*[Create(h) for h in heads], lag_ratio=0.3), run_time=2.2)
        self.play(FadeIn(head_lbls), FadeIn(dots3), run_time=0.6)
        # las mismas palabras quedan en posiciones distintas: opiniones en paralelo
        self.play(LaggedStart(*[Indicate(h[4], color=BLANCO, scale_factor=1.2)
                                for h in heads], lag_ratio=0.2), run_time=1.1)
        self._siguiente()

        # ── se combinan todas en una representación compuesta ───────────
        self.play(FadeOut(cap1), FadeOut(head_lbls), FadeOut(dots3), run_time=0.4)
        cap2 = Text("Se combinan todas en una representación compuesta",
                    font=FUENTE, font_size=23, color=MARRON_OSCURO,
                    t2c={"compuesta": NARANJA_TERRACOTA}).next_to(cabezas, DOWN, buff=0.3)
        self.play(FadeIn(cap2, shift=DOWN * 0.1))

        # el espacio compuesto (promedio de las propuestas)
        Mc = sum(Ms) / len(Ms)
        compuesto = head_space(Mc, NARANJA_TERRACOTA, np.array([0.0, 0.0, 0]))
        compuesto.scale(1.5).move_to(DOWN * 0.35)
        marco_c = SurroundingRectangle(compuesto, color=NARANJA_TERRACOTA, buff=0.25,
                                       stroke_width=3, corner_radius=0.16)
        centro = compuesto.get_center()

        # las tres propuestas se deslizan y se superponen en el centro...
        self.play(
            LaggedStart(*[
                heads[k].animate.scale(1.15).move_to(centro).set_opacity(0.4)
                for k in range(3)
            ], lag_ratio=0.14),
            run_time=1.5,
        )
        # ...y de la superposición emerge una sola representación compuesta
        self.play(
            ReplacementTransform(VGroup(*heads), compuesto),
            Create(marco_c),
            run_time=1.1,
        )
        self.play(
            Indicate(compuesto[4], color=NARANJA_TERRACOTA, scale_factor=1.25),
            Flash(centro, color=NARANJA_TERRACOTA, line_length=0.35, num_lines=14),
            run_time=0.9,
        )
        self._siguiente()
        self.limpiar_pantalla()