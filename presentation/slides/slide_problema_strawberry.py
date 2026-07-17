import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import os
from colores import *
from objetos import *


class SlideProblemaStrawberry:
    def slide_problema_strawberry(self):

        titulo, linea = self.crear_titulo(
            "¿Por qué los LLM no saben 'leer'?",
            palabra_clave="'leer'?",
            color_clave=NARANJA_TERRACOTA
        )
        llanuras_fondo = crear_llanuras_manchegas()
        sol_fondo = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN * 0.2 + LEFT * 0.2)

        burbuja_pregunta = self._crear_burbuja_chat(
            "¿Cuántas letras 'r' hay en 'strawberry'?",
            color_fondo=MARRON_OSCURO,
            color_texto=PAPEL_CREMA, es_usuario=True
        )

        burbuja_respuesta = self._crear_burbuja_chat(
            "Hay 2 letras 'r' en 'strawberry'.",
            color_fondo=FONDO_CAJA,
            color_texto=TINTA_NEGRA, es_usuario=False,
            t2c_dict={"2": NARANJA_TERRACOTA}
        )

        grupo_chat = VGroup(burbuja_pregunta, burbuja_respuesta).arrange(DOWN, buff=0.5)
        burbuja_pregunta.shift(RIGHT * 1.5)
        burbuja_respuesta.shift(LEFT * 1.5)
        grupo_chat.next_to(linea, DOWN, buff=0.35)

        fresa_der = self._crear_fresa().to_corner(DR).shift(UP * 0.3 + LEFT * 0.3)
        fresa_izq = self._crear_fresa().to_corner(DL).shift(UP * 0.3 + RIGHT * 0.3)

        token1 = self.crear_bloque("str", ancho=1.2)
        token2 = self.crear_bloque("aw", ancho=1.2)
        token3 = self.crear_bloque("berry", ancho=1.6)
        tokens_straw = VGroup(token1, token2, token3).arrange(RIGHT, buff=0.15)
        tokens_straw.move_to(DOWN * 2.62)

        letras = VGroup(*[
            Text(c, font=FUENTE, font_size=44, color=TINTA_NEGRA, weight=BOLD)
            for c in "strawberry"
        ]).arrange(RIGHT, buff=0.12).move_to(DOWN * 1.2)

        INDICES_R = [2, 7, 8]
        chips_r = VGroup()
        for n, idx in enumerate(INDICES_R):
            circulo = Circle(radius=0.16, fill_color=ROJO_TOMATE,
                             fill_opacity=1, stroke_width=0)
            numero = Text(str(n + 1), font=FUENTE, font_size=14,
                          color=BLANCO, weight=BOLD).move_to(circulo)
            chips_r.add(VGroup(circulo, numero).next_to(letras[idx], UP, buff=0.18))

        caption_tokens = Text(
            "El modelo no ve letras: ve fichas",
            font=FUENTE, font_size=22, color=TINTA_NEGRA,
            t2c={"fichas": NARANJA_TERRACOTA},
        ).move_to(DOWN * 1.95)

        self._animar_entrada_slide(titulo, linea, fondo=VGroup(llanuras_fondo, sol_fondo))

        self.add(fresa_der, fresa_izq)
        self.play(FadeIn(burbuja_pregunta, shift=UP * 0.2, scale=0.9))
        self.wait(0.5)
        self.play(FadeIn(burbuja_respuesta, shift=UP * 0.2, scale=0.9))

        self.play(
            LaggedStart(*[FadeIn(l, shift=UP * 0.1) for l in letras], lag_ratio=0.05),
            run_time=0.7,
        )
        for chip, idx in zip(chips_r, INDICES_R):
            self.play(
                letras[idx].animate.set_color(ROJO_TOMATE),
                FadeIn(chip, scale=0.6),
                run_time=0.35,
            )
        self.play(Wiggle(burbuja_respuesta, scale_value=1.06, rotation_angle=0.02))

        self._siguiente()

        grupos_letras = [letras[0:3], letras[3:5], letras[5:10]]
        self.play(FadeIn(caption_tokens, shift=UP * 0.1))
        self.play(
            LaggedStart(*[
                TransformFromCopy(VGroup(*grupo), token)
                for grupo, token in zip(grupos_letras, [token1, token2, token3])
            ], lag_ratio=0.25),
            run_time=1.3,
        )

        caption_final = Text(
            "Las tres r quedan escondidas dentro de las fichas",
            font=FUENTE, font_size=17, color=ROJO_CONTRA, weight=BOLD
        ).next_to(tokens_straw, DOWN, buff=0.24)
        self.play(
            FadeIn(caption_final, shift=UP * 0.1),
            *[chip.animate.set_opacity(0.25) for chip in chips_r],
            *[letras[idx].animate.set_opacity(0.3) for idx in INDICES_R],
        )

        self._siguiente()

        self.limpiar_pantalla()


