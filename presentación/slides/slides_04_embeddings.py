import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


class SlidesEmbeddings:
    _MAGNITUD_BASE = 2.3
    _VALORES_EMBEDDING = ["0.9", "-0.1", "0.4", "0.6", "...", "0.7"]
    _FILA_ACTIVA_W     = 3

    class _TokenVector:
        def __init__(self, nombre, angulo_deg, magnitud, color, dir_etiqueta):
            self.nombre       = nombre
            self.angulo_deg   = angulo_deg
            self.magnitud     = magnitud
            self.color        = color
            self.dir_etiqueta = dir_etiqueta

        @property
        def angulo_rad(self):
            return math.radians(self.angulo_deg)

        @property
        def coordenada(self):
            return np.array([
                self.magnitud * math.cos(self.angulo_rad),
                self.magnitud * math.sin(self.angulo_rad),
                0.0,
            ])

    _TOKENS = {
        "Rey":    _TokenVector("Rey",    38.0,   _MAGNITUD_BASE, NARANJA_TERRACOTA, UR),
        "Reina":  _TokenVector("Reina",  142.0,  _MAGNITUD_BASE, VERDE_OLIVA,       UL),
        "Hombre": _TokenVector("Hombre", -38.0,  _MAGNITUD_BASE, NARANJA_TERRACOTA, DR),
        "Mujer":  _TokenVector("Mujer",  -142.0, _MAGNITUD_BASE, VERDE_OLIVA,       DL),
    }


    @staticmethod
    def _factory_nodo_pipeline(etiqueta, subtexto="", color_borde=NARANJA_TERRACOTA, ancho=2.8):
        bg     = RoundedRectangle(corner_radius=0.16, width=ancho, height=0.74,
                                  fill_color=FONDO_CAJA, fill_opacity=1,
                                  stroke_color=color_borde, stroke_width=2.5)
        titulo = Text(etiqueta, font=FUENTE, font_size=22,
                      color=color_borde, weight=BOLD).move_to(bg)
        if subtexto:
            sub = Text(subtexto, font=FUENTE, font_size=13,
                       color=MARRON_OSCURO, slant=ITALIC)
            VGroup(titulo, sub).arrange(DOWN, buff=0.06).move_to(bg)
            return VGroup(bg, titulo, sub)
        return VGroup(bg, titulo)

    @staticmethod
    def _factory_matriz_W(n_filas=7, n_cols=5, fila_activa=3):
        filas = VGroup()
        for i in range(n_filas):
            es_activa = (i == fila_activa)
            fila = VGroup(*[
                RoundedRectangle(corner_radius=0.06, width=0.55, height=0.42,
                                 fill_color=NARANJA_TERRACOTA if es_activa else PAPEL_CREMA,
                                 fill_opacity=1,
                                 stroke_color=MARRON_OSCURO, stroke_width=1.2
                                 ).scale(0.50)
                for _ in range(n_cols)
            ]).arrange(RIGHT, buff=0.04)
            filas.add(fila)
        filas.arrange(DOWN, buff=0.04)
        puntos_r = Text("...", font="Monospace", font_size=15, color=MARRON_OSCURO
                        ).next_to(filas, RIGHT, buff=0.08)
        puntos_b = Text("...", font="Monospace", font_size=15, color=MARRON_OSCURO
                        ).next_to(filas, DOWN, buff=0.05)
        lbl_W    = Text("W  (vocab × 768)", font=FUENTE, font_size=13,
                        color=PAPEL_TAN, weight=BOLD).next_to(filas, UP, buff=0.18)
        return VGroup(lbl_W, filas, puntos_r, puntos_b), filas

    @staticmethod
    def _factory_vector_resultado(valores):
        celdas = VGroup()
        for val in valores:
            es_puntos = (val == "···")
            celda = RoundedRectangle(corner_radius=0.06, width=1.0, height=0.48,
                                     fill_color=VERDE_OLIVA if not es_puntos else PAPEL_CREMA,
                                     fill_opacity=1 if not es_puntos else 0,
                                     stroke_color=VERDE_OLIVA if not es_puntos else FONDO_CAJA,
                                     stroke_width=1.5).scale(0.78)
            txt = Text(val, font="Monospace", font_size=14,
                       color=PAPEL_CREMA if not es_puntos else MARRON_OSCURO
                       ).move_to(celda)
            celdas.add(VGroup(celda, txt))
        celdas.arrange(DOWN, buff=0.05)
        lbl = Text("embedding\n(768 dims)", font=FUENTE, font_size=13,
                   color=VERDE_OLIVA, weight=BOLD).next_to(celdas, UP, buff=0.16)
        return VGroup(lbl, celdas), celdas

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


    def slide_embeddings(self) -> None:
        t1 = Text("Embeddings: ", font=FUENTE, font_size=40, weight=BOLD, color=TINTA_NEGRA)
        t2 = Text("El Mapa del Significado", font=FUENTE, font_size=40,
                  weight=BOLD, color=NARANJA_TERRACOTA)
        titulo = VGroup(t1, t2).arrange(RIGHT, buff=0.08).to_edge(UP, buff=0.5)
        linea  = Line(LEFT * 6, RIGHT * 6, color=MARRON_OSCURO, stroke_width=2
                      ).next_to(titulo, DOWN, buff=0.15)

        llanuras = crear_llanuras_manchegas()
        adornos  = self._crear_adornos_esquinas(escala=0.55, buff=0.5)
        adornos[1].add_updater(lambda m, dt: m.rotate(dt * 0.4))
        self._animar_entrada_slide(titulo, linea, fondo=llanuras, adornos=adornos)

        self._acto_ids_ciegos(linea)
        self._acto_mapa_semantico(linea)
        self._acto_lookup_simple(linea)

        adornos[1].clear_updaters()
        self._siguiente()
        self.limpiar_pantalla()


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


    def _acto_ids_ciegos(self, linea: Mobject) -> None:
        pregunta = Text("¿Qué recibe el modelo tras tokenizar?",
                        font=FUENTE, font_size=27, weight=BOLD, color=TINTA_NEGRA
                        ).next_to(linea, DOWN, buff=0.45)
        self.play(FadeIn(pregunta, shift=DOWN * 0.15))

        datos = [("quijote", "1605"), ("caballero", "8472"), ("hidalgo", "23198")]
        filas = VGroup()
        for word, id_ in datos:
            w_box = RoundedRectangle(corner_radius=0.12, width=2.4, height=0.62,
                                     fill_color=PAPEL_TAN, fill_opacity=0.85,
                                     stroke_color=MARRON_OSCURO, stroke_width=2)
            w_txt = Text(word, font=FUENTE, font_size=22, color=TINTA_NEGRA).move_to(w_box)
            arrow = Arrow(ORIGIN, RIGHT * 1.1, color=MARRON_OSCURO, stroke_width=2,
                          max_tip_length_to_length_ratio=0.18)
            i_box = RoundedRectangle(corner_radius=0.12, width=1.8, height=0.62,
                                     fill_color=FONDO_CAJA, fill_opacity=1,
                                     stroke_color=ACERO, stroke_width=2)
            i_txt = Text(id_, font="Monospace", font_size=22, color=ACERO).move_to(i_box)
            fila  = VGroup(VGroup(w_box, w_txt), arrow, VGroup(i_box, i_txt)
                           ).arrange(RIGHT, buff=0.35)
            filas.add(fila)

        filas.arrange(DOWN, buff=0.45).next_to(pregunta, DOWN, buff=0.4)
        self.play(LaggedStart(*[FadeIn(f, shift=RIGHT * 0.2) for f in filas], lag_ratio=0.4),
                  run_time=1.2)

        nota = Text("Números arbitrarios — sin relación semántica visible",
                    font=FUENTE, font_size=21, color=NARANJA_TERRACOTA, weight=BOLD
                    ).next_to(filas, DOWN, buff=0.4)
        self.play(Write(nota))

        self._siguiente()
        self.play(FadeOut(pregunta, filas, nota))

    def _acto_mapa_semantico(self, linea: Mobject) -> None:

        ejes = Axes(
            x_range=[-3.8, 3.8, 1], y_range=[-2.8, 2.8, 1],
            x_length=7.0, y_length=3.6,
            axis_config={"color": MARRON_OSCURO, "stroke_width": 1.5,
                         "include_ticks": False, "stroke_opacity": 0.45},
        ).move_to(ORIGIN).shift(UP * 0.3)


        lbl_x = Text("← femenino  |  masculino →", font="Monospace",
                     font_size=12, color=AZUL_NOCHE
                     ).next_to(ejes.x_axis, DOWN, buff=0.45)
        lbl_y = Text("↑ nobleza", font="Monospace", font_size=12, color=AZUL_NOCHE
                     ).next_to(ejes.y_axis.get_top(), LEFT, buff=0.12)

        self.play(Create(ejes), run_time=0.7)
        self.play(Write(lbl_x), Write(lbl_y))


        coords = {k: ejes.c2p(*v.coordenada[:2]) for k, v in self._TOKENS.items()}

        def _punto(nombre, color, dir_lbl):
            pos = coords[nombre]
            dot = Dot(pos, radius=0.14, color=color, fill_opacity=1)
            lbl = Text(nombre, font=FUENTE, font_size=19, color=color, weight=BOLD)
            lbl.set_background_stroke(color=PAPEL_CREMA, width=5)
            lbl.next_to(pos, dir_lbl, buff=0.16)
            return VGroup(dot, lbl)

        puntos = {
            "Rey":    _punto("Rey",    NARANJA_TERRACOTA, UR),
            "Reina":  _punto("Reina",  VERDE_OLIVA,       UL),
            "Hombre": _punto("Hombre", NARANJA_TERRACOTA, DR),
            "Mujer":  _punto("Mujer",  VERDE_OLIVA,       DL),
        }


        self.play(
            LaggedStart(Create(puntos["Hombre"]), Create(puntos["Rey"]), lag_ratio=0.45),
            run_time=0.9,
        )
        self.play(
            LaggedStart(Create(puntos["Mujer"]), Create(puntos["Reina"]), lag_ratio=0.45),
            run_time=0.9,
        )


        flecha_g1 = Arrow(coords["Rey"],    coords["Reina"],  color=LAVANDA, stroke_width=2.8,
                          max_tip_length_to_length_ratio=0.10, buff=0.18)
        flecha_g2 = Arrow(coords["Hombre"], coords["Mujer"],  color=LAVANDA, stroke_width=2.8,
                          max_tip_length_to_length_ratio=0.10, buff=0.18)

        self.play(
            LaggedStart(Create(flecha_g1), Create(flecha_g2), lag_ratio=0.35),
            run_time=0.9,
        )


        formula_lhs = MathTex(r"\vec{\text{Rey}} - \vec{\text{Hombre}} + \vec{\text{Mujer}}",
                              font_size=30, color=TINTA_NEGRA)
        formula_rhs = MathTex(r"\approx \vec{\text{Reina}}", font_size=30, color=VERDE_OLIVA)
        formula     = VGroup(formula_lhs, formula_rhs).arrange(RIGHT, buff=0.15)
        fondo_f     = SurroundingRectangle(formula, color=NARANJA_TERRACOTA,
                                           fill_color=PAPEL_CREMA, fill_opacity=0.95,
                                           buff=0.2, corner_radius=0.12, stroke_width=2.2)
        formula_grp = VGroup(fondo_f, formula).next_to(ejes, DOWN, buff=0.25)

        self.play(FadeIn(fondo_f, scale=0.88), Write(formula_lhs), run_time=0.9)
        self.play(Write(formula_rhs))
        self.play(
            Indicate(puntos["Reina"], scale_factor=1.35, color=ORO_VIEJO),
            Flash(coords["Reina"], color=ORO_VIEJO, line_length=0.5, num_lines=14),
        )

        self._siguiente()
        self.play(FadeOut(
            ejes, lbl_x, lbl_y,
            *puntos.values(), flecha_g1, flecha_g2,
            formula_grp,
        ))

    def _acto_lookup_simple(self, linea: Mobject) -> None:
        label = Text("¿Cómo funciona en la práctica? Un lookup en la tabla W.",
                     font=FUENTE, font_size=25, weight=BOLD, color=MARRON_OSCURO
                     ).next_to(linea, DOWN, buff=0.45)
        self.play(FadeIn(label, shift=DOWN * 0.1))


        nodo_token = self._factory_nodo_pipeline('"quijote"', "token de entrada",
                                                  NARANJA_TERRACOTA, 2.8)

        nodo_id = self._factory_nodo_pipeline("ID: 1605", "índice entero",
                                               MARRON_OSCURO, 2.2)

        fl_tok = Arrow(ORIGIN, RIGHT * 0.9, color=MARRON_OSCURO, stroke_width=2.5,
                       max_tip_length_to_length_ratio=0.22)
        fl_tok_lbl = Text("tokenizar", font=FUENTE, font_size=13, color=MARRON_OSCURO,
                          slant=ITALIC).next_to(fl_tok, UP, buff=0.08)

        fl_look = Arrow(ORIGIN, RIGHT * 0.9, color=MARRON_OSCURO, stroke_width=2.5,
                        max_tip_length_to_length_ratio=0.22)
        fl_look_lbl = Text("lookup", font=FUENTE, font_size=13, color=MARRON_OSCURO,
                           slant=ITALIC).next_to(fl_look, UP, buff=0.08)


        grupo_W, filas_W = self._factory_matriz_W(fila_activa=self._FILA_ACTIVA_W)


        fl_res = Arrow(ORIGIN, RIGHT * 0.75, color=VERDE_OLIVA, stroke_width=2.5,
                       max_tip_length_to_length_ratio=0.22)


        grupo_res, celdas_res = self._factory_vector_resultado(self._VALORES_EMBEDDING)


        pipeline = VGroup(
            nodo_token,
            VGroup(fl_tok, fl_tok_lbl),
            nodo_id,
            VGroup(fl_look, fl_look_lbl),
            grupo_W,
            fl_res,
            grupo_res,
        ).arrange(RIGHT, buff=0.28)
        if pipeline.width > 12.8:
            pipeline.scale(12.8 / pipeline.width)
        pipeline.next_to(label, DOWN, buff=0.8).set_x(0)


        for fl_grp, ref in [(VGroup(fl_tok, fl_tok_lbl), nodo_id),
                             (VGroup(fl_look, fl_look_lbl), nodo_id)]:
            fl_grp[0].set_y(ref.get_center()[1])
            fl_grp[1].next_to(fl_grp[0], UP, buff=0.08)
        fl_res.set_y(filas_W.get_center()[1])


        brace_dims     = Brace(celdas_res, direction=RIGHT, color=VERDE_OLIVA)
        brace_dims_lbl = Text("768", font="Monospace", font_size=13,
                              color=VERDE_OLIVA, weight=BOLD
                              ).next_to(brace_dims, RIGHT, buff=0.08)


        self.play(FadeIn(nodo_token, scale=0.92))
        self.play(GrowArrow(fl_tok), FadeIn(fl_tok_lbl, shift=DOWN * 0.1))
        self.play(FadeIn(nodo_id, shift=RIGHT * 0.2))
        self.play(GrowArrow(fl_look), FadeIn(fl_look_lbl, shift=DOWN * 0.1))
        self.play(FadeIn(grupo_W, shift=LEFT * 0.12))

        rect_fila = SurroundingRectangle(filas_W[self._FILA_ACTIVA_W],
                                          color=ORO_VIEJO, buff=0.07,
                                          stroke_width=3, corner_radius=0.08)
        self.play(Create(rect_fila))
        self.play(Indicate(nodo_id, color=ORO_VIEJO, scale_factor=1.08))
        self.play(
            GrowArrow(fl_res),
            ReplacementTransform(rect_fila.copy(), celdas_res),
            FadeIn(grupo_res[0], shift=DOWN * 0.1),
            run_time=1.2,
        )
        self.play(FadeOut(rect_fila))
        self.play(Create(brace_dims), Write(brace_dims_lbl))
        self.play(Flash(celdas_res.get_center(), color=VERDE_OLIVA,
                        line_length=0.45, num_lines=12))


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

        def _mini_vec(valores, color_fill, color_stroke=None):
            cs = color_stroke or color_fill
            celdas = VGroup()
            for val in valores:
                es_p = (val == "·")
                rect = RoundedRectangle(
                    corner_radius=0.05,
                    width=0.66,
                    height=0.5,
                    fill_color=color_fill if not es_p else FONDO_CAJA,
                    fill_opacity=0.85 if not es_p else 0
                )
                rect.set_stroke(cs if not es_p else FONDO_CAJA, 1.5)
                txt = Text(val, font="Monospace", font_size=15, color=TINTA_NEGRA).move_to(rect)
                celdas.add(VGroup(rect, txt))
            return celdas.arrange(RIGHT, buff=0.08)

        pos_data = [
            (0, ["0.1", "-0.3", "0.5", "·"], ["0.8", "-0.5", "0.1", "·"], ["0.9", "-0.8", "0.9", "·"]),
            (1, ["0.4", "0.2", "-0.1", "·"], ["0.3", "0.7", "-0.4", "·"], ["0.7", "0.9", "-0.4", "·"]),
            (2, ["-0.8", "0.6", "0.1", "·"], ["0.1", "-0.2", "0.9", "·"], ["0.4", "0.3", "1.0", "·"]),
        ]
        palabras = ["perro", "muerde", "hombre"]

        cols = VGroup()
        for pos_num, tok_vals, pos_vals, sum_vals in pos_data:
            palabra_lbl = Text(palabras[pos_num], font=FUENTE, font_size=22,
                            weight=BOLD, color=TINTA_NEGRA)
            pos_lbl_txt = Text(f"pos {pos_num}", font=FUENTE, font_size=16, color=ACERO)

            v_tok = _mini_vec(tok_vals, PAPEL_TAN)
            v_pos = _mini_vec(pos_vals, NARANJA_TERRACOTA)
            v_sum = _mini_vec(sum_vals, CAJA_INFERIOR)

            lbl_tok = Text("token", font=FUENTE, font_size=13, color=MARRON_OSCURO)
            lbl_pos = Text("posición", font=FUENTE, font_size=13, color=NARANJA_TERRACOTA)
            lbl_sum = Text("final", font=FUENTE, font_size=13, color=TINTA_NEGRA, weight=BOLD)

            mas = Text("+", font=FUENTE, font_size=28, weight=BOLD, color=TINTA_NEGRA)
            sep = Line(LEFT * v_pos.width / 2, RIGHT * v_pos.width / 2,
                    color=MARRON_OSCURO, stroke_width=1.8)

            col_content = VGroup(
                VGroup(lbl_tok, v_tok).arrange(DOWN, buff=0.06),
                mas,
                VGroup(lbl_pos, v_pos).arrange(DOWN, buff=0.06),
                sep,
                VGroup(lbl_sum, v_sum).arrange(DOWN, buff=0.06),
            ).arrange(DOWN, buff=0.18, aligned_edge=LEFT)

            encabezado = VGroup(palabra_lbl, pos_lbl_txt).arrange(DOWN, buff=0.08)
            encabezado.next_to(col_content, UP, buff=0.25)

            caja = RoundedRectangle(
                corner_radius=0.14,
                width=col_content.width + 0.6,
                height=col_content.height + encabezado.height + 0.7,
                fill_color=FONDO_CAJA,
                fill_opacity=1,
                stroke_color=MARRON_OSCURO,
                stroke_width=1.8,
            )

            col_group = VGroup(caja, encabezado, col_content)
            col_content.move_to(caja.get_center() + DOWN * 0.25)
            encabezado.next_to(col_content, UP, buff=0.2)

            cols.add(col_group)

        cols.arrange(RIGHT, buff=0.5).next_to(linea, DOWN, buff=0.9)

        if cols.width > 13.0:
            cols.scale(13.0 / cols.width)

        self.play(
            LaggedStart(*[FadeIn(c, scale=0.92) for c in cols], lag_ratio=0.35),
            run_time=1.2
        )

        puntos_sum = [cols[i][2][-1] for i in range(3)]
        self.play(
            LaggedStart(
                *[Indicate(p, scale_factor=1.12, color=ORO_VIEJO) for p in puntos_sum],
                lag_ratio=0.3,
            ),
            run_time=1.0
        )

        mensaje = Text(
            "Cada token lleva: ¿qué palabra?  +  ¿en qué posición?",
            font=FUENTE,
            font_size=21,
            color=TINTA_NEGRA
        ).next_to(cols, DOWN, buff=0.4)

        self.play(Write(mensaje))

        self._siguiente()
        self.play(FadeOut(cols, mensaje))

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