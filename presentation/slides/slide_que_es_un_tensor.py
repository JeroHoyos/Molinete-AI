import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import os
from colores import *
from objetos import *


class SlideQueEsUnTensor:
    def slide_que_es_un_tensor(self):

        titulo, linea = self.crear_titulo(
            "¿Qué es un Tensor?",
            palabra_clave="Tensor?",
            color_clave=NARANJA_TERRACOTA
        )
        llanuras_fondo = crear_llanuras_manchegas()
        adornos = self._crear_adornos_esquinas()
        adornos[1].add_updater(lambda m, dt: m.rotate(dt * 0.15))

        self._animar_entrada_slide(titulo, linea, adornos=adornos, fondo=llanuras_fondo)


        lbl_0d = Text("Escalar", font=FUENTE, font_size=22,
                    color=MARRON_OSCURO, weight=BOLD)
        lbl_1d = Text("Vector",  font=FUENTE, font_size=22,
                    color=MARRON_OSCURO, weight=BOLD)
        lbl_2d = Text("Matriz",  font=FUENTE, font_size=22,
                    color=MARRON_OSCURO, weight=BOLD)
        lbl_3d = Text("Tensor",  font=FUENTE, font_size=22,
                    color=MARRON_OSCURO, weight=BOLD)


        escalar = self.crear_bloque("7", ancho=1.0, alto=1.0)

        vector = VGroup(*[
            self.crear_bloque(v, ancho=0.9, alto=0.55)
            for v in ["1", "5", "9", "2"]
        ]).arrange(DOWN, buff=0.08)

        valores_mat = ["3","1","4", "2","5","9", "2","6","5", "3","5","8"]
        filas_2d = VGroup(*[
            VGroup(*[
                self.crear_bloque(valores_mat[r*3 + c], ancho=0.7, alto=0.55)
                for c in range(3)
            ]).arrange(RIGHT, buff=0.06)
            for r in range(4)
        ]).arrange(DOWN, buff=0.06)

        def hacer_capa(valores, color_fondo, opacidad):
            capa = VGroup(*[
                VGroup(*[
                    self.crear_bloque(
                        valores[r*3 + c],
                        ancho=0.58, alto=0.48,
                        color_fondo=color_fondo
                    )
                    for c in range(3)
                ]).arrange(RIGHT, buff=0.05)
                for r in range(3)
            ]).arrange(DOWN, buff=0.05)
            capa.set_opacity(opacidad)
            return capa

        capa_back = hacer_capa(["1","2","3","4","5","6","7","8","9"], PAPEL_CREMA,       0.30)
        capa_mid  = hacer_capa(["9","8","7","6","5","4","3","2","1"], PAPEL_TAN,         0.60)
        capa_top  = hacer_capa(["2","4","6","8","0","2","4","6","8"], NARANJA_TERRACOTA, 1.00)
        capa_mid.shift(RIGHT * 0.18 + UP * 0.18)
        capa_top.shift(RIGHT * 0.36 + UP * 0.36)
        tensor_3d = VGroup(capa_back, capa_mid, capa_top)


        forma_0d = Text("forma: []",      font=FUENTE, font_size=17, color=PAPEL_TAN)
        forma_1d = Text("forma: [4]",     font=FUENTE, font_size=17, color=PAPEL_TAN)
        forma_2d = Text("forma: [4, 3]",  font=FUENTE, font_size=17, color=PAPEL_TAN)
        forma_3d = Text("forma: [3,3,3]", font=FUENTE, font_size=17, color=PAPEL_TAN)


        contenidos = VGroup(escalar, vector, filas_2d, tensor_3d)
        contenidos.arrange(RIGHT, buff=1.2)


        ZONA_TOP = linea.get_bottom()[1] - 0.5
        ZONA_BOT = -2.8
        ZONA_MID_Y = (ZONA_TOP + ZONA_BOT) / 2


        for mob in contenidos:
            mob.set_y(ZONA_MID_Y)


        for lbl, mob in zip([lbl_0d, lbl_1d, lbl_2d, lbl_3d], contenidos):
            lbl.next_to(mob, UP, buff=0.3).set_x(mob.get_x())


        etiquetas = [lbl_0d, lbl_1d, lbl_2d, lbl_3d]
        formas    = [forma_0d, forma_1d, forma_2d, forma_3d]

        lbl_top_y = max(lbl.get_top()[1] for lbl in etiquetas)
        for lbl in etiquetas:
            lbl.set_y(lbl_top_y - lbl.height / 2)

        y_formas = min(mob.get_bottom()[1] for mob in contenidos) - 0.42
        for forma, mob in zip(formas, contenidos):
            forma.move_to([mob.get_x(), y_formas, 0])

        def _pill_dim(texto_dim):
            fondo = RoundedRectangle(
                corner_radius=0.12, width=0.62, height=0.34,
                fill_color=CAJA_INFERIOR, fill_opacity=1,
                stroke_color=MARRON_OSCURO, stroke_width=1.2,
            )
            t = Text(texto_dim, font=FUENTE, font_size=15,
                     color=MARRON_OSCURO, weight=BOLD).move_to(fondo)
            return VGroup(fondo, t)

        pills = VGroup(*[_pill_dim(t) for t in ("0D", "1D", "2D", "3D")])
        for pill, lbl in zip(pills, etiquetas):
            pill.next_to(lbl, UP, buff=0.16).set_x(lbl.get_x())


        separadores = VGroup()
        pares = [(escalar, vector), (vector, filas_2d), (filas_2d, tensor_3d)]
        for izq, der in pares:
            x_sep = (izq.get_right()[0] + der.get_left()[0]) / 2
            sep = DashedLine(
                UP * 2.8, DOWN * 2.8,
                color=MARRON_OSCURO, stroke_width=1.0,
                dash_length=0.12, dashed_ratio=0.4,
            ).set_x(x_sep)
            separadores.add(sep)


        anims_bloques = [
            GrowFromCenter(escalar),
            LaggedStart(*[FadeIn(b, shift=UP*0.15) for b in vector],   lag_ratio=0.08),
            LaggedStart(*[FadeIn(f, shift=UP*0.1)  for f in filas_2d], lag_ratio=0.08),
            AnimationGroup(
                FadeIn(capa_back, shift=UP*0.1),
                FadeIn(capa_mid,  shift=UP*0.1),
                FadeIn(capa_top,  shift=UP*0.1),
                lag_ratio=0.15,
            ),
        ]

        for i in range(4):
            anims = [
                FadeIn(pills[i], shift=DOWN * 0.1),
                Write(etiquetas[i]),
                anims_bloques[i],
                FadeIn(formas[i], shift=UP * 0.1),
            ]
            if i > 0:
                anims.append(Create(separadores[i - 1]))
            self.play(*anims, run_time=0.75)

        self._siguiente()

        adornos[1].clear_updaters()
        self.limpiar_pantalla()


