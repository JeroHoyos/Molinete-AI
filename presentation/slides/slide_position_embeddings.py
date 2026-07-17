import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import numpy as np
import os
from colores import *
from objetos import *


class SlidePositionEmbeddings:
    def slide_position_embeddings(self) -> None:
        t1 = Text("Embeddings de ", font=FUENTE, font_size=42, weight=BOLD, color=TINTA_NEGRA)
        t2 = Text("Posición", font=FUENTE, font_size=42, weight=BOLD, color=NARANJA_TERRACOTA)
        titulo = VGroup(t1, t2).arrange(RIGHT, buff=0.08).to_edge(UP, buff=0.5)
        linea  = Line(LEFT * 6, RIGHT * 6, color=MARRON_OSCURO, stroke_width=2
                      ).next_to(titulo, DOWN, buff=0.15)

        llanuras = crear_llanuras_manchegas()
        adornos  = self._crear_adornos_esquinas(escala=0.55, buff=0.5)
        adornos[1].add_updater(lambda m, dt: m.rotate(dt * 0.4))
        self._animar_entrada_slide(titulo, linea, fondo=llanuras, adornos=adornos)

        self._acto_lectura_simultanea(linea)
        self._acto_suma_posicion(linea)
        self._acto_ventana_contexto(linea)

        adornos[1].clear_updaters()
        self._siguiente()
        self.limpiar_pantalla()


    @staticmethod
    def _fila_tokens(palabras: list) -> VGroup:
        cajas = VGroup()
        for w in palabras:
            rect = RoundedRectangle(corner_radius=0.1, width=1.5, height=0.6,
                                    fill_color=PAPEL_CREMA, fill_opacity=0.9,
                                    stroke_color=MARRON_OSCURO, stroke_width=2)
            txt = Text(w, font=FUENTE, font_size=22, color=TINTA_NEGRA
                       ).move_to(rect.get_center())
            cajas.add(VGroup(rect, txt))
        return cajas.arrange(RIGHT, buff=0.25)


    def _acto_lectura_simultanea(self, linea: Mobject) -> None:


        frase_a = self._fila_tokens(["El", "perro", "muerde", "al", "hombre"])
        frase_b = self._fila_tokens(["El", "hombre", "muerde", "al", "perro"])
        lbl_a = Text("Frase A:", font=FUENTE, font_size=19, weight=BOLD, color=MARRON_OSCURO)
        lbl_b = Text("Frase B:", font=FUENTE, font_size=19, weight=BOLD, color=MARRON_OSCURO)
        grp_a = VGroup(lbl_a, frase_a).arrange(RIGHT, buff=0.3)
        grp_b = VGroup(lbl_b, frase_b).arrange(RIGHT, buff=0.3)
        frases = VGroup(grp_a, grp_b).arrange(DOWN, buff=0.65).move_to(ORIGIN).shift(UP * 0.3)

        self.play(FadeIn(grp_a, shift=RIGHT * 0.25))
        self.play(FadeIn(grp_b, shift=RIGHT * 0.25))


        self.play(
            frase_a[1][0].animate.set_fill(ROJO_TOMATE, opacity=0.85),
            frase_a[1][1].animate.set_color(BLANCO),
            frase_b[4][0].animate.set_fill(ROJO_TOMATE, opacity=0.85),
            frase_b[4][1].animate.set_color(BLANCO),
            frase_a[4][0].animate.set_fill(VERDE_OLIVA, opacity=0.85),
            frase_a[4][1].animate.set_color(BLANCO),
            frase_b[1][0].animate.set_fill(VERDE_OLIVA, opacity=0.85),
            frase_b[1][1].animate.set_color(BLANCO),
        )


        bag_label = Text("Sin posición, el modelo trata la secuencia como un conjunto sin estructura",
                         font=FUENTE, font_size=21, color=NARANJA_TERRACOTA, weight=BOLD
                         ).next_to(frases, DOWN, buff=0.55)
        self.play(Write(bag_label))

        self._siguiente()
        self.play(FadeOut(grp_a, grp_b, bag_label))

    def _acto_suma_posicion(self, linea: Mobject) -> None:
        # espacio vectorial: token + posición = embedding final
        centro = np.array([-1.5, -0.55, 0.0])
        grid = self._emb_grid(centro, unidad=0.8, n=3)
        self.play(Create(grid), run_time=0.8)

        # el token "perro" es un vector base (misma palabra siempre)
        B = centro + np.array([1.3, 1.4, 0])
        v_tok = self._emb_vec(centro, B, PAPEL_TAN, width=5)
        lbl_tok = Text("perro", font=FUENTE, font_size=20, color=MARRON_OSCURO,
                       weight=BOLD).set_background_stroke(color=PAPEL_CREMA, width=5)\
            .next_to(B, UP, buff=0.12)
        self.play(GrowArrow(v_tok), FadeIn(lbl_tok, shift=UP * 0.1))

        # sumar el vector de posición lo desplaza a un punto distinto por posición
        offs = {0: np.array([1.05, 0.35, 0]),
                1: np.array([1.4, -0.55, 0]),
                2: np.array([0.55, -1.35, 0])}
        cols_pos = {0: NARANJA_TERRACOTA, 1: VERDE_OLIVA, 2: AZUL_NOCHE}
        dirs_pos = {0: UR, 1: RIGHT, 2: DR}
        finales = VGroup()
        for p in (0, 1, 2):
            fin = B + offs[p]
            arr = self._emb_vec(B, fin, cols_pos[p], width=3.2, opacity=0.9)
            pt = self._emb_punto(fin, cols_pos[p], f"pos {p}", dirs_pos[p], fs=17)
            self.play(GrowArrow(arr), Create(pt), run_time=0.6)
            finales.add(arr, pt)

        # panel lateral: token + posición = final
        suma = VGroup(
            Text("perro", font=FUENTE, font_size=22, color=PAPEL_TAN, weight=BOLD),
            Text("+", font=FUENTE, font_size=24, color=TINTA_NEGRA),
            Text("posición", font=FUENTE, font_size=22, color=NARANJA_TERRACOTA, weight=BOLD),
        ).arrange(RIGHT, buff=0.2)
        igual = Text("= embedding final", font=FUENTE, font_size=22,
                     color=TINTA_NEGRA, weight=BOLD)
        panel = VGroup(suma, igual).arrange(DOWN, buff=0.25)\
            .to_edge(RIGHT, buff=0.9)
        self.play(FadeIn(panel, shift=LEFT * 0.15))

        mensaje = Text("El mismo token cae en un punto distinto según su posición",
                       font=FUENTE, font_size=21, color=MARRON_OSCURO)\
            .to_edge(DOWN, buff=0.5)
        self.play(Write(mensaje))

        self._siguiente()
        self.play(FadeOut(grid, v_tok, lbl_tok, finales, panel, mensaje))

    def _acto_ventana_contexto(self, linea: Mobject) -> None:

        titulo = Text(
            "Ventana de contexto",
            font=FUENTE, font_size=26, weight=BOLD, color=TINTA_NEGRA,
        ).next_to(linea, DOWN, buff=0.45)
        self.play(Write(titulo))

        def _celda_pos(num, color):
            celda = RoundedRectangle(
                corner_radius=0.06,
                width=1.2,
                height=0.55,
                fill_color=color,
                fill_opacity=0.7,
            ).set_stroke(MARRON_OSCURO, 1.5)
            etiq = Text(f"{num}", font=FUENTE, font_size=16, color=TINTA_NEGRA)
            etiq.move_to(celda)
            return VGroup(celda, etiq)

        c0    = _celda_pos(0,    PAPEL_TAN)
        c1    = _celda_pos(1,    NARANJA_TERRACOTA)
        c2    = _celda_pos(2,    PAPEL_TAN)
        dots  = Text("···", font_size=30, color=MARRON_OSCURO)
        c_last = _celda_pos(1023, CAJA_INFERIOR)

        tabla = VGroup(c0, c1, c2, dots, c_last).arrange(RIGHT, buff=0.25)

        brace = Brace(tabla, DOWN, color=MARRON_OSCURO)
        lbl = Text(
            "1024 posiciones",
            font=FUENTE,
            font_size=18,
            color=MARRON_OSCURO,
        ).next_to(brace, DOWN, buff=0.2)

        grupo_tabla = VGroup(tabla, brace, lbl)
        grupo_tabla.next_to(titulo, DOWN, buff=1.6).set_x(0).shift(LEFT * 0.8)

        self.play(FadeIn(tabla, shift=UP * 0.2))
        self.play(GrowFromCenter(brace), Write(lbl))


        token_fuera = RoundedRectangle(
            corner_radius=0.06,
            width=1.2,
            height=0.55,
            fill_color=NARANJA_TERRACOTA,
            fill_opacity=0.8,
        ).set_stroke(MARRON_OSCURO, 1.5)

        token_lbl = Text("1024", font=FUENTE, font_size=16, color=TINTA_NEGRA)
        token_lbl.move_to(token_fuera)
        token_group = VGroup(token_fuera, token_lbl)
        token_group.next_to(tabla, RIGHT, buff=0.25)

        cruz = Text("✕", font_size=40, color=NARANJA_TERRACOTA).next_to(token_group, UP, buff=0.15)

        self.play(FadeIn(token_group, shift=RIGHT * 0.3))
        self.play(Write(cruz))

        self.play(Indicate(c_last, color=ORO_VIEJO))