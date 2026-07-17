import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import os
from colores import *
from objetos import *


class SlideProntoIniciamos:
    def slide_pronto_iniciamos(self):

        gato_caballero = ImageMobject(os.path.join("assets", "gato_armadura.png")).scale(0.5)
        texto_inicio = Text("Pronto iniciamos", font=FUENTE, font_size=50, weight=BOLD, color=MARRON_OSCURO)

        cat_and_text = Group(gato_caballero, texto_inicio).arrange(DOWN, buff=0.8)
        cat_and_text.move_to(ORIGIN)

        pantalla_completa = Group(cat_and_text)

        self.play(FadeIn(pantalla_completa, shift=UP))

        self._siguiente()
        self.limpiar_pantalla()

