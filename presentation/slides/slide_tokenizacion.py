import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import os
from colores import *
from objetos import *


class SlideTokenizacion:
    def slide_tokenizacion(self):

        llanuras_fondo = crear_llanuras_manchegas()
        adornos = self._crear_adornos_esquinas(escala=0.55, buff=0.5)

        titulo, linea = self.crear_titulo(
            "La Tokenización",
            palabra_clave="Tokenización",
            color_clave=NARANJA_TERRACOTA
        )
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo, adornos=adornos)

        frase = "Confía en el tiempo que suele dar dulces salidas"


        tokens_palabra  = frase.split(" ")
        tokens_caracter = [c if c != ' ' else '·' for c in frase]
        tokens_bpe      = [
            "Con", "f", "ía", "en", "el", "tiem", "po",
            "que", "su", "ele", "dar", "dul", "ces", "sali", "das"
        ]

        def hacer_bloques(tokens, ancho_fn, buff_b=0.1,
                          color_fondo=FONDO_CAJA, color_texto=MARRON_OSCURO):
            g = VGroup(*[
                self.crear_bloque(t, ancho=ancho_fn(t),
                                  color_fondo=color_fondo, color_texto=color_texto)
                for t in tokens
            ]).arrange(RIGHT, buff=buff_b)
            if g.width > 11.5:
                g.scale_to_fit_width(11.5)
            return g

        bloques_1 = hacer_bloques(tokens_palabra,  lambda t: max(0.65, len(t) * 0.22))
        bloques_2 = hacer_bloques(tokens_caracter, lambda t: 0.28, buff_b=0.04)
        bloques_3 = hacer_bloques(tokens_bpe, lambda t: max(0.5, len(t) * 0.25))


        lbl_1 = Text(
            "1. Por palabra",
            font=FUENTE, font_size=19, color=TINTA_NEGRA, weight=BOLD,
            t2c={"Por palabra": NARANJA_TERRACOTA}
        )
        lbl_2 = Text(
            "2. Por carácter",
            font=FUENTE, font_size=19, color=TINTA_NEGRA, weight=BOLD,
            t2c={"Por carácter": NARANJA_TERRACOTA}
        )
        lbl_3 = Text(
            "3. BPE",
            font=FUENTE, font_size=19, color=TINTA_NEGRA, weight=BOLD,
            t2c={"BPE": NARANJA_TERRACOTA}
        )


        grupo_1 = VGroup(lbl_1, bloques_1).arrange(DOWN, buff=0.15)
        grupo_2 = VGroup(lbl_2, bloques_2).arrange(DOWN, buff=0.15)
        grupo_3 = VGroup(lbl_3, bloques_3).arrange(DOWN, buff=0.15)

        todos = VGroup(grupo_1, grupo_2, grupo_3)\
            .arrange(DOWN, buff=0.32)\
            .next_to(linea, DOWN, buff=0.35)\
            .move_to(UP * 0.1)


        self.play(FadeIn(lbl_1, shift=RIGHT * 0.15))
        self.play(
            LaggedStart(*[GrowFromCenter(b) for b in bloques_1], lag_ratio=0.07),
            run_time=1.1
        )

        self.play(FadeIn(lbl_2, shift=RIGHT * 0.15))
        self.play(
            LaggedStart(*[GrowFromCenter(b) for b in bloques_2], lag_ratio=0.01),
            run_time=1.5
        )

        self.play(FadeIn(lbl_3, shift=RIGHT * 0.15))
        self.play(
            LaggedStart(*[GrowFromCenter(b) for b in bloques_3], lag_ratio=0.07),
            run_time=1.2
        )
        self._siguiente()

        self.limpiar_pantalla()
