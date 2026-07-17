import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import numpy as np
import os
from colores import *
from objetos import *


class SlideCapaTransformer:
    def slide_capa_transformer(self):
        titulo, linea = self.crear_titulo(
            "La Capa Transformer",
            palabra_clave="Transformer",
            color_clave=NARANJA_TERRACOTA,
        )
        llanuras_fondo = crear_llanuras_manchegas()
        adornos = self._crear_adornos_esquinas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo, adornos=adornos)

        def card(texto, fill, txt_color, w=1.7, h=0.85, fs=19):
            sombra = RoundedRectangle(corner_radius=0.15, width=w, height=h,
                                      fill_color=MARRON_OSCURO, fill_opacity=0.13,
                                      stroke_width=0).shift(DOWN * 0.05 + RIGHT * 0.05)
            caja = RoundedRectangle(corner_radius=0.15, width=w, height=h,
                                    fill_color=fill, fill_opacity=1,
                                    stroke_color=MARRON_OSCURO, stroke_width=2.5)
            t = Text(texto, font=FUENTE, font_size=fs, color=txt_color).move_to(caja)
            if t.width > w - 0.24:
                t.scale((w - 0.24) / t.width)
            return VGroup(sombra, caja, t)

        def suma(pos):
            c = Circle(radius=0.27, fill_color=FONDO_CAJA, fill_opacity=1,
                       stroke_color=NARANJA_TERRACOTA, stroke_width=3)
            p = MathTex(r"+", font_size=34, color=NARANJA_TERRACOTA).move_to(c)
            return VGroup(c, p).move_to(pos)

        n_in = card("Input", CAJA_INFERIOR, TINTA_NEGRA, w=1.6)
        attn = card("Self-Attention", NARANJA_TERRACOTA, BLANCO, w=2.3)
        add1 = suma(ORIGIN)
        mlp = card("MLP", VERDE_OLIVA, BLANCO, w=1.5)
        add2 = suma(ORIGIN)
        n_out = card("Output", CAJA_INFERIOR, TINTA_NEGRA, w=1.7)

        # separación uniforme entre bordes -> todas las flechas del mismo tamaño
        GAP = 0.62
        VGroup(n_in, attn, add1, mlp, add2, n_out).arrange(RIGHT, buff=GAP)
        YR = n_in.get_center()[1]

        def flecha(a, b):
            return Arrow(a, b, buff=0.06, color=MARRON_OSCURO, stroke_width=3,
                         max_tip_length_to_length_ratio=0.22)

        f1 = flecha(n_in.get_right(), attn.get_left())
        f2 = flecha(attn.get_right(), add1.get_left())
        f3 = flecha(add1.get_right(), mlp.get_left())
        f4 = flecha(mlp.get_right(), add2.get_left())
        f5 = flecha(add2.get_right(), n_out.get_left())

        # atajos residuales: pasan por arriba rodeando cada sub-bloque
        YT = YR + 1.35

        def atajo(x_start, add):
            pa = np.array([x_start, YR, 0])
            pb = np.array([x_start, YT, 0])
            pc = np.array([add.get_center()[0], YT, 0])
            return VGroup(
                Line(pa, pb, color=NARANJA_TERRACOTA, stroke_width=3),
                Line(pb, pc, color=NARANJA_TERRACOTA, stroke_width=3),
                Arrow(pc, add.get_top(), buff=0, color=NARANJA_TERRACOTA,
                      stroke_width=3, max_tip_length_to_length_ratio=0.4),
            )

        skip1 = atajo(f1.get_center()[0], add1)
        skip2 = atajo(f3.get_center()[0], add2)

        diagrama = VGroup(n_in, attn, add1, mlp, add2, n_out,
                          f1, f2, f3, f4, f5, skip1, skip2).move_to(UP * 0.1)

        # ── animación ───────────────────────────────────────────────────
        self.play(FadeIn(n_in, shift=RIGHT * 0.15))
        self.play(GrowArrow(f1), FadeIn(attn, shift=RIGHT * 0.1))
        self.play(GrowArrow(f2), FadeIn(add1, scale=0.6),
                  Create(skip1))
        self.play(GrowArrow(f3), FadeIn(mlp, shift=RIGHT * 0.1))
        self.play(GrowArrow(f4), FadeIn(add2, scale=0.6), Create(skip2))
        self.play(GrowArrow(f5), FadeIn(n_out, shift=RIGHT * 0.15))

        # los dos sub-bloques, cada uno con su atajo
        self.play(
            LaggedStart(Indicate(add1, color=NARANJA_TERRACOTA, scale_factor=1.35),
                        Indicate(add2, color=NARANJA_TERRACOTA, scale_factor=1.35),
                        lag_ratio=0.4),
        )

        llave = Brace(VGroup(attn, add1, mlp, add2), DOWN, color=MARRON_OSCURO, buff=0.25)
        lbl_n = Text("×  N capas", font=FUENTE, font_size=20, color=MARRON_OSCURO,
                     weight=BOLD).next_to(llave, DOWN, buff=0.15)
        self.play(GrowFromCenter(llave), FadeIn(lbl_n, shift=DOWN * 0.1))

        self._siguiente()
        self.limpiar_pantalla()
