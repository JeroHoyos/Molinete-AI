import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import math
import os
from colores import *
from objetos import *


class SlideStrides:
    def slide_strides(self):

        titulo, linea = self.crear_titulo(
            "Strides",
            palabra_clave="Strides",
            color_clave=NARANJA_TERRACOTA
        )

        camino_mancha = FunctionGraph(lambda x: 0.5 * math.sin(x) - 0.5, color=MARRON_OSCURO).set_opacity(0.15)
        camino_punteado = DashedVMobject(camino_mancha, num_dashes=45, dashed_ratio=0.5)

        lanza_fondo = Line(LEFT * 7 + DOWN * 2, RIGHT * 7 + UP * 2, color=NARANJA_TERRACOTA, stroke_width=2).set_opacity(0.15)

        decoracion_quijote = VGroup(camino_punteado, lanza_fondo).set_z_index(-2)

        adornos = self._crear_adornos_esquinas()
        adornos[1].add_updater(lambda m, dt: m.rotate(dt * 0.15))

        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, adornos=adornos, fondo=VGroup(llanuras_fondo, decoracion_quijote))

        arr_1d = VGroup(*[self.crear_bloque(str(i)) for i in range(6)])
        arr_1d.arrange(RIGHT, buff=0.1).shift(UP * 1.5)

        lbl_1d = Text("RAM (1D)", font=FUENTE, font_size=20, color=MARRON_OSCURO, weight=BOLD).next_to(arr_1d, UP, buff=0.3)

        self.play(FadeIn(arr_1d, shift=UP*0.2), FadeIn(lbl_1d, shift=UP*0.2))
        self._siguiente()

        fila1 = VGroup(*[arr_1d[i].copy() for i in range(3)]).arrange(RIGHT, buff=0.1)
        fila2 = VGroup(*[arr_1d[i].copy() for i in range(3, 6)]).arrange(RIGHT, buff=0.1)

        mat_shape = VGroup(fila1, fila2).arrange(DOWN, buff=0.1).shift(DOWN * 0.55)

        def _chip_strides(texto_chip):
            t = Text(texto_chip, font=FUENTE, font_size=17,
                     color=MARRON_OSCURO, weight=BOLD)
            caja = RoundedRectangle(
                corner_radius=0.1, width=t.width + 0.35, height=t.height + 0.22,
                fill_color=CAJA_INFERIOR, fill_opacity=0.9,
                stroke_color=MARRON_OSCURO, stroke_width=1.3,
            ).move_to(t)
            return VGroup(caja, t)

        lbl_2d = Text("Shape lógica (2×3)", font=FUENTE, font_size=20,
                      color=MARRON_OSCURO, weight=BOLD)
        chip_strides = _chip_strides("strides: [3, 1]")
        pie_matriz = VGroup(lbl_2d, chip_strides).arrange(RIGHT, buff=0.35)
        pie_matriz.move_to([0, -2.15, 0])

        self.play(
            TransformFromCopy(VGroup(*arr_1d[0:3]), fila1),
            TransformFromCopy(VGroup(*arr_1d[3:6]), fila2),
            FadeIn(lbl_2d, shift=DOWN*0.2),
            FadeIn(chip_strides, shift=DOWN*0.2),
            run_time=1.5
        )

        self.play(
            arr_1d[0][0].animate.set_fill(PAPEL_TAN, opacity=0.8),
            arr_1d[3][0].animate.set_fill(PAPEL_TAN, opacity=0.8),
            mat_shape[0][0][0].animate.set_fill(PAPEL_TAN, opacity=0.8),
            mat_shape[1][0][0].animate.set_fill(PAPEL_TAN, opacity=0.8),
        )

        arco_fila = CurvedArrow(arr_1d[0].get_top(), arr_1d[3].get_top(),
                                angle=-PI/2, color=NARANJA_TERRACOTA)
        txt_fila = Text("stride fila = 3", font=FUENTE, font_size=19,
                        color=NARANJA_TERRACOTA, weight=BOLD).next_to(arco_fila, UP, buff=0.08)

        arco_col = CurvedArrow(arr_1d[0].get_bottom(), arr_1d[1].get_bottom(),
                               angle=PI/2, color=VERDE_OLIVA)
        txt_col = Text("stride col = 1", font=FUENTE, font_size=16,
                       color=VERDE_OLIVA, weight=BOLD).next_to(arco_col, DOWN, buff=0.08)

        arco_2d_fila = CurvedArrow(mat_shape[0][0].get_left(), mat_shape[1][0].get_left(),
                                   angle=PI/2, color=NARANJA_TERRACOTA).shift(LEFT*0.1)
        txt_2d_fila = Text("+1 fila", font=FUENTE, font_size=16,
                           color=NARANJA_TERRACOTA).next_to(arco_2d_fila, LEFT, buff=0.1)

        arco_2d_col = CurvedArrow(mat_shape[1][1].get_bottom(), mat_shape[1][2].get_bottom(),
                                  angle=PI/2, color=VERDE_OLIVA)
        txt_2d_col = Text("+1 columna", font=FUENTE, font_size=16,
                          color=VERDE_OLIVA).next_to(arco_2d_col, RIGHT, buff=0.12)

        self.play(
            Create(arco_fila), Write(txt_fila),
            Create(arco_col), Write(txt_col),
        )
        self.play(
            Create(arco_2d_fila), Write(txt_2d_fila),
            Create(arco_2d_col), Write(txt_2d_col),
        )

        self._siguiente()

        adornos[1].clear_updaters()
        self.limpiar_pantalla()


