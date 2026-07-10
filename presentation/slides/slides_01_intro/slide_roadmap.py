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


def _icono_tensores():
    def capa(color_fondo, opacidad):
        return VGroup(*[
            Square(0.26, fill_color=color_fondo, fill_opacity=opacidad,
                   stroke_color=MARRON_OSCURO, stroke_width=1.6)
            for _ in range(9)
        ]).arrange_in_grid(3, 3, buff=0.045)

    atras  = capa(PAPEL_TAN, 0.9).shift(UP * 0.14 + RIGHT * 0.14)
    frente = capa(FONDO_CAJA, 1.0)
    return VGroup(atras, frente)


def _icono_tokenizacion():
    palabra = RoundedRectangle(
        corner_radius=0.06, width=1.05, height=0.3,
        fill_color=PERGAMINO, fill_opacity=1,
        stroke_color=MARRON_OSCURO, stroke_width=1.6,
    )
    cortes = VGroup(*[
        DashedLine(UP * 0.2, DOWN * 0.2, dash_length=0.05,
                   stroke_color=ROJO_TOMATE, stroke_width=2.2).move_to(palabra).shift(RIGHT * dx)
        for dx in (-0.14, 0.24)
    ])
    anchos  = [0.34, 0.42, 0.25]
    colores = [SALMON_ATENCION, ARENA_DORADA, CAJA_INFERIOR]
    chips = VGroup(*[
        RoundedRectangle(corner_radius=0.06, width=w, height=0.3,
                         fill_color=c, fill_opacity=1,
                         stroke_color=MARRON_OSCURO, stroke_width=1.6)
        for w, c in zip(anchos, colores)
    ]).arrange(RIGHT, buff=0.08).next_to(palabra, DOWN, buff=0.16)
    return VGroup(palabra, cortes, chips)


def _icono_embeddings():
    ejes = VGroup(
        Line(LEFT * 0.42, RIGHT * 0.42, stroke_color=MARRON_OSCURO,
             stroke_width=2, stroke_opacity=0.65),
        Line(DOWN * 0.42, UP * 0.42, stroke_color=MARRON_OSCURO,
             stroke_width=2, stroke_opacity=0.65),
    )
    vectores = VGroup()
    for angulo, color in ((35, NARANJA_TERRACOTA), (120, VERDE_OLIVA), (-40, PAPEL_TAN)):
        punta = np.array([
            0.38 * math.cos(math.radians(angulo)),
            0.38 * math.sin(math.radians(angulo)),
            0,
        ])
        vectores.add(VGroup(
            Line(ORIGIN, punta, stroke_color=color, stroke_width=3),
            Dot(punta, radius=0.045, color=color),
        ))
    return VGroup(ejes, vectores)


def _icono_arquitectura():
    bloques = VGroup(*[
        RoundedRectangle(corner_radius=0.04, width=0.68, height=0.18,
                         fill_color=c, fill_opacity=1,
                         stroke_color=MARRON_OSCURO, stroke_width=1.4)
        for c in (SALMON_ATENCION, CELESTE_PALIDO, AMARILLO_PALIDO)
    ]).arrange(UP, buff=0.06)
    marco = RoundedRectangle(
        corner_radius=0.09, width=0.9, height=0.85,
        fill_color=CREMA_CALIDA, fill_opacity=0.6,
        stroke_color=MARRON_OSCURO, stroke_width=1.6,
    ).move_to(bloques)
    flecha = Arrow(
        marco.get_top(), marco.get_top() + UP * 0.26, buff=0,
        color=MARRON_OSCURO, stroke_width=2.5,
        max_tip_length_to_length_ratio=0.6,
    )
    return VGroup(marco, bloques, flecha)


def _icono_entrenamiento():
    eje_y = Line(ORIGIN, UP * 0.8, stroke_color=MARRON_OSCURO, stroke_width=2.2)
    eje_x = Line(ORIGIN, RIGHT * 1.0, stroke_color=MARRON_OSCURO, stroke_width=2.2)
    curva = VMobject(stroke_color=NARANJA_TERRACOTA, stroke_width=3.5)
    curva.set_points_smoothly([
        np.array([0.07, 0.7, 0]), np.array([0.3, 0.32, 0]),
        np.array([0.55, 0.16, 0]), np.array([0.95, 0.07, 0]),
    ])
    punto = Dot(np.array([0.95, 0.07, 0]), radius=0.045, color=ROJO_TOMATE)
    return VGroup(eje_y, eje_x, curva, punto)


def _icono_demo():
    cola = Polygon(
        [-0.25, -0.28, 0], [-0.05, -0.28, 0], [-0.3, -0.52, 0],
        fill_color=FONDO_CAJA, fill_opacity=1,
        stroke_color=MARRON_OSCURO, stroke_width=1.8,
    )
    globo = RoundedRectangle(
        corner_radius=0.16, width=1.0, height=0.6,
        fill_color=FONDO_CAJA, fill_opacity=1,
        stroke_color=MARRON_OSCURO, stroke_width=1.8,
    )
    puntos = VGroup(*[
        Dot(radius=0.05, color=NARANJA_TERRACOTA) for _ in range(3)
    ]).arrange(RIGHT, buff=0.14).move_to(globo)
    return VGroup(cola, globo, puntos)


class SlideRoadmap:
    def slide_roadmap(self):
        llanuras_fondo = crear_llanuras_manchegas()
        adornos = self._crear_adornos_esquinas(escala=0.55, buff=0.55)
        adornos[0][-1].add_updater(lambda m, dt: m.rotate(-dt * 0.5))

        titulo, linea = self.crear_titulo(
            "Hoja de Ruta",
            palabra_clave="Ruta",
            color_clave=NARANJA_TERRACOTA,
        )
        self._animar_entrada_slide(titulo, linea, adornos=adornos, fondo=llanuras_fondo)

        pasos = [
            ("Tensores",      NARANJA_TERRACOTA, _icono_tensores,      1.00),
            ("Tokenización",  MARRON_OSCURO,     _icono_tokenizacion,  1.00),
            ("Embeddings",    NARANJA_TERRACOTA, _icono_embeddings,    1.05),
            ("Arquitectura",  MARRON_OSCURO,     _icono_arquitectura,  0.95),
            ("Entrenamiento", NARANJA_TERRACOTA, _icono_entrenamiento, 1.00),
            ("Demo",          MARRON_OSCURO,     _icono_demo,          1.00),
        ]
        XS = [-4.9, -2.94, -0.98, 0.98, 2.94, 4.9]
        YS = [-0.60, -0.30, -0.55, -0.30, -0.55, -0.60]
        Y_ETIQUETAS = -2.7

        nodos, iconos, textos = VGroup(), VGroup(), VGroup()
        for i, ((etiqueta, color, fn_icono, escala), x, y) in enumerate(zip(pasos, XS, YS)):
            anillo = Circle(
                radius=0.55,
                stroke_color=color, stroke_width=5,
                fill_color=PAPEL_CREMA, fill_opacity=0.95,
            )
            numero = Text(str(i + 1), font=FUENTE, font_size=34,
                          color=color, weight=BOLD).move_to(anillo)
            nodos.add(VGroup(anillo, numero).move_to([x, y, 0]))

            icono = fn_icono().scale(escala).next_to(nodos[i], DOWN, buff=0.28)
            iconos.add(icono)
            textos.add(Text(etiqueta, font=FUENTE, font_size=20,
                            color=MARRON_OSCURO, weight=BOLD).move_to([x, Y_ETIQUETAS, 0]))

        def tramo_camino(p_a, p_b):
            curva = CubicBezier(
                p_a, p_a + RIGHT * 0.7 + DOWN * 0.22,
                p_b + LEFT * 0.7 + DOWN * 0.22, p_b,
            )
            return DashedVMobject(curva, num_dashes=9, dashed_ratio=0.55)\
                .set_stroke(color=MARRON_OSCURO, width=3.5)

        tramos = VGroup(
            tramo_camino(np.array([-6.35, -0.5, 0]), nodos[0][0].get_left())
        )
        for i in range(len(pasos) - 1):
            tramos.add(tramo_camino(nodos[i][0].get_right(), nodos[i + 1][0].get_left()))

        quijote = ImageMobject(os.path.join("assets", "quijote_rust.png")).set_height(1.0)
        quijote.move_to([-6.15, 0.3, 0])

        self.play(FadeIn(quijote, shift=RIGHT * 0.3), run_time=0.5)

        for i in range(len(pasos)):
            llegada = nodos[i][0].get_top() + UP * 0.5
            self.play(
                Create(tramos[i]),
                quijote.animate(path_arc=-0.5).move_to(llegada),
                run_time=0.55,
            )
            self.play(
                DrawBorderThenFill(nodos[i][0]),
                Write(nodos[i][1]),
                FadeIn(iconos[i], scale=0.6),
                Write(textos[i]),
                run_time=0.55,
            )

        self.play(
            LaggedStart(
                *[Indicate(n, scale_factor=1.12, color=NARANJA_TERRACOTA) for n in nodos],
                lag_ratio=0.12,
            ),
            run_time=1.2,
        )

        self._siguiente()
        self.limpiar_pantalla()

