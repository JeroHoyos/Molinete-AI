import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import numpy as np
import os
from colores import *
from objetos import *


class SlideEmbeddings:
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

        self._siguiente()
        self.play(FadeOut(pregunta, filas))

    # análogo Rey/Reina/Hombre/Mujer:  offset respecto al centro, color, dir. etiqueta
    _ANALOGIA = {
        "Hombre": (np.array([1.9, -1.05, 0]), NARANJA_TERRACOTA, DR),
        "Rey":    (np.array([1.9, 1.35, 0]),  NARANJA_TERRACOTA, UR),
        "Mujer":  (np.array([-1.9, -1.05, 0]), VERDE_OLIVA,      DL),
        "Reina":  (np.array([-1.9, 1.35, 0]),  VERDE_OLIVA,      UL),
    }

    def _acto_mapa_semantico(self, linea: Mobject) -> None:
        centro = np.array([0.0, -0.3, 0.0])
        grid = self._emb_grid(centro, unidad=0.72, n=3)
        self.play(Create(grid), run_time=0.9)

        # etiquetas de los ejes semánticos (Monospace soporta ← → ↑)
        ejes = VGroup(
            Text("masculino →", font="Monospace", font_size=14, color=AZUL_NOCHE)
            .next_to(grid[1].get_right(), UP, buff=0.12),
            Text("← femenino", font="Monospace", font_size=14, color=AZUL_NOCHE)
            .next_to(grid[1].get_left(), UP, buff=0.12),
            Text("↑ nobleza", font="Monospace", font_size=14, color=AZUL_NOCHE)
            .next_to(grid[2].get_top(), RIGHT, buff=0.12),
        )
        self.play(FadeIn(ejes))

        # puntos del análogo en el espacio
        pos = {n: centro + off for n, (off, _, _) in self._ANALOGIA.items()}
        pts = {n: self._emb_punto(pos[n], c, n, d)
               for n, (_, c, d) in self._ANALOGIA.items()}
        self.play(LaggedStart(Create(pts["Hombre"]), Create(pts["Rey"]),
                              lag_ratio=0.45), run_time=0.9)
        self.play(LaggedStart(Create(pts["Mujer"]), Create(pts["Reina"]),
                              lag_ratio=0.45), run_time=0.9)

        # dirección compartida "nobleza": Hombre→Rey ∥ Mujer→Reina
        nobleza = VGroup(
            self._emb_vec(pos["Hombre"], pos["Rey"], LAVANDA, width=3.4),
            self._emb_vec(pos["Mujer"], pos["Reina"], LAVANDA, width=3.4),
        )
        cap_nob = Text("misma dirección: nobleza", font=FUENTE, font_size=15,
                       color=MARRON_OSCURO).next_to(nobleza, UP, buff=0.05).set_x(0)
        self.play(GrowArrow(nobleza[0]), GrowArrow(nobleza[1]), run_time=0.8)
        self.play(FadeIn(cap_nob, shift=UP * 0.1))
        self._siguiente()

        # aritmética vectorial: Rey - Hombre + Mujer ≈ Reina
        self.play(FadeOut(cap_nob), run_time=0.3)
        op = self._emb_vec(pos["Rey"], pos["Reina"], ORO_VIEJO, width=4.5)
        movpt = Dot(pos["Rey"], radius=0.11, color=ORO_VIEJO)\
            .set_stroke(MARRON_OSCURO, width=1.5)
        self.play(FadeIn(movpt, scale=0.6))
        self.play(GrowArrow(op), movpt.animate.move_to(pos["Reina"]), run_time=1.2)

        formula = MathTex(r"\vec{\text{Rey}} - \vec{\text{Hombre}} + \vec{\text{Mujer}}",
                          r"\approx", r"\vec{\text{Reina}}", font_size=30)
        formula[0].set_color(TINTA_NEGRA)
        formula[2].set_color(VERDE_OLIVA)
        caja_f = SurroundingRectangle(formula, color=NARANJA_TERRACOTA,
                                      fill_color=PAPEL_CREMA, fill_opacity=0.95,
                                      buff=0.2, corner_radius=0.12, stroke_width=2.2)
        formula_grp = VGroup(caja_f, formula).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(caja_f, scale=0.9), Write(formula), run_time=1.0)
        self.play(Indicate(pts["Reina"], scale_factor=1.35, color=ORO_VIEJO),
                  Flash(pos["Reina"], color=ORO_VIEJO, line_length=0.45, num_lines=14))
        self.play(FadeOut(movpt), run_time=0.2)

        self._siguiente()
        self.play(FadeOut(grid, ejes, *pts.values(), nobleza, op, formula_grp))

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


