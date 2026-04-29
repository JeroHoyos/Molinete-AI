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
        """Representa un token en el espacio 2D de embeddings (dirección + magnitud)."""
        def __init__(self, nombre, angulo_deg, magnitud, color, dir_etiqueta):
            self.nombre      = nombre
            self.angulo_deg  = angulo_deg
            self.magnitud    = magnitud
            self.color       = color
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

    # ──────────────────────────────────────────────────────────────────────────
    # FACTORIES
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod

    def _factory_vector(token, ejes):
        origen  = ejes.c2p(0, 0)
        extremo = ejes.c2p(*token.coordenada[:2])
        flecha = Arrow(origen, extremo, color=token.color, buff=0,
                       stroke_width=3.5, max_tip_length_to_length_ratio=0.10)
        punto  = Dot(extremo, radius=0.11, color=token.color)
        etiq   = Text(token.nombre, font=FUENTE, font_size=20,
                      color=token.color, weight=BOLD)
        etiq.set_background_stroke(color=PAPEL_CREMA, width=5)
        etiq.next_to(extremo, token.dir_etiqueta, buff=0.14)
        return VGroup(flecha, punto, etiq)

    @staticmethod

    def _factory_lineas_delta(tokens, ejes):
        p = {k: ejes.c2p(*v.coordenada[:2]) for k, v in tokens.items()}

        lg1 = DashedLine(p["Rey"],    p["Reina"],  color=LAVANDA, stroke_width=2.8)
        lg2 = DashedLine(p["Hombre"], p["Mujer"],  color=LAVANDA, stroke_width=2.8)
        lineas_genero = VGroup(lg1, lg2)

        brace_g = BraceBetweenPoints(p["Reina"] + UP * 0.05, p["Rey"] + UP * 0.05,
                                     direction=UP, color=LAVANDA)
        delta_g = tokens["Rey"].coordenada[0] - tokens["Reina"].coordenada[0]
        lbl_bg  = Text(f"d. genero = {delta_g:+.1f}", font="Monospace",
                       font_size=13, color=LAVANDA, weight=BOLD
                       ).next_to(brace_g, UP, buff=0.12)
        brace_genero = VGroup(brace_g, lbl_bg)

        ln1 = DashedLine(p["Hombre"], p["Rey"],   color=AZUL_NOCHE, stroke_width=2.8,
                         dash_length=0.12)
        ln2 = DashedLine(p["Mujer"],  p["Reina"], color=AZUL_NOCHE, stroke_width=2.8,
                         dash_length=0.12)
        lineas_nobleza = VGroup(ln1, ln2)

        brace_n = BraceBetweenPoints(p["Hombre"] + RIGHT * 0.05, p["Rey"] + RIGHT * 0.05,
                                     direction=RIGHT, color=AZUL_NOCHE)
        delta_n = tokens["Rey"].coordenada[1] - tokens["Hombre"].coordenada[1]
        lbl_bn  = Text(f"d. nobleza = {delta_n:+.1f}", font="Monospace",
                       font_size=13, color=AZUL_NOCHE, weight=BOLD
                       ).next_to(brace_n, RIGHT, buff=0.12)
        brace_nobleza = VGroup(brace_n, lbl_bn)

        return lineas_genero, brace_genero, lineas_nobleza, brace_nobleza

    @staticmethod

    def _factory_formula_analogia():
        lhs     = MathTex(r"\vec{\text{Rey}} - \vec{\text{Hombre}} + \vec{\text{Mujer}}",
                          font_size=32, color=TINTA_NEGRA)
        rhs     = MathTex(r"\approx \vec{\text{Reina}}", font_size=32, color=VERDE_OLIVA)
        formula = VGroup(lhs, rhs).arrange(RIGHT, buff=0.18)
        fondo   = SurroundingRectangle(formula, color=NARANJA_TERRACOTA,
                                       fill_color=PAPEL_CREMA, fill_opacity=0.95,
                                       buff=0.22, corner_radius=0.14, stroke_width=2.2)
        return VGroup(fondo, formula)

    @staticmethod

    def _factory_nodo_pipeline(etiqueta, subtexto="", color_borde=NARANJA_TERRACOTA,
                                ancho=2.8):
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

    def _factory_flecha_anotada(etiqueta, color, largo=1.2):
        flecha = Arrow(ORIGIN, RIGHT * largo, color=color, stroke_width=2.8,
                       max_tip_length_to_length_ratio=0.22)
        if etiqueta:
            lbl = Text(etiqueta, font=FUENTE, font_size=13,
                       color=color, slant=ITALIC).next_to(flecha, UP, buff=0.10)
            return VGroup(flecha, lbl)
        return VGroup(flecha)

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

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE: Embeddings — De Tokens a Significado
    # ══════════════════════════════════════════════════════════════════════════


    def slide_embeddings(self) -> None:
        # ── Cabecera ──────────────────────────────────────────────────────────
        t1     = Text("Embeddings: Vectores de ", font=FUENTE, font_size=40,
                    weight=BOLD, color=TINTA_NEGRA)
        t2     = Text("Significado", font=FUENTE, font_size=40,
                    weight=BOLD, color=NARANJA_TERRACOTA)
        titulo = VGroup(t1, t2).arrange(RIGHT, buff=0.08).to_edge(UP, buff=0.5)
        linea  = Line(LEFT * 6, RIGHT * 6, color=MARRON_OSCURO, stroke_width=2
                    ).next_to(titulo, DOWN, buff=0.15)

        llanuras = crear_llanuras_manchegas()
        adornos  = self._crear_adornos_esquinas(escala=0.55, buff=0.5)
        adornos[1].add_updater(lambda m, dt: m.rotate(dt * 0.4))
        self._animar_entrada_slide(titulo, linea, fondo=llanuras, adornos=adornos)

        # ── BLOQUE A: Geometría del significado ───────────────────────────────
        geo_label = Text("Geometría del significado", font=FUENTE, font_size=26,
                        weight=BOLD, color=NARANJA_TERRACOTA, slant=ITALIC
                        ).next_to(linea, DOWN, buff=0.6)

        self.play(FadeIn(geo_label, shift=DOWN * 0.15))

        ejes = Axes(
            x_range=[-3.8, 3.8, 1], y_range=[-2.6, 2.6, 1],
            x_length=7.2, y_length=3.4,
            axis_config={"color": MARRON_OSCURO, "stroke_width": 1.5,
                        "include_ticks": False, "stroke_opacity": 0.5},
        ).next_to(geo_label, DOWN, buff=0.18).set_x(0)

        self.play(Create(ejes), run_time=0.9)

        # Secuencia declarativa de actos:
        # (tokens_visibles, mostrar_delta_genero, mostrar_delta_nobleza, mostrar_formula)
        ACTOS_GEO = [
            (["Rey", "Hombre"],                       False, False, False),
            (["Rey", "Hombre", "Reina", "Mujer"],     False, False, False),
            (["Rey", "Hombre", "Reina", "Mujer"],     True,  False, False),
            (["Rey", "Hombre", "Reina", "Mujer"],     True,  True,  False),
            (["Rey", "Hombre", "Reina", "Mujer"],     True,  True,  True),
        ]

        vectores_en_pantalla: dict = {}
        capas_delta: list          = []
        formula_mostrada           = None

        for tokens_vis, show_dg, show_dn, show_formula in ACTOS_GEO:
            # Añadir vectores nuevos
            nuevos = [k for k in tokens_vis if k not in vectores_en_pantalla]
            if nuevos:
                nvg = {k: self._factory_vector(self._TOKENS[k], ejes) for k in nuevos}
                self.play(
                    LaggedStart(*[Create(v) for v in nvg.values()], lag_ratio=0.35),
                    run_time=1.0,
                )
                vectores_en_pantalla.update(nvg)

            # Δ género
            if show_dg and not capas_delta:
                lg, bg, _, _ = self._factory_lineas_delta(self._TOKENS, ejes)
                self.play(
                    LaggedStart(Create(lg),
                                AnimationGroup(Create(bg[0]), Write(bg[1])),
                                lag_ratio=0.45),
                    run_time=1.2,
                )
                capas_delta = [lg, bg]

            # Δ nobleza
            if show_dn and len(capas_delta) < 4:
                _, _, ln, bn = self._factory_lineas_delta(self._TOKENS, ejes)
                self.play(
                    LaggedStart(Create(ln),
                                AnimationGroup(Create(bn[0]), Write(bn[1])),
                                lag_ratio=0.45),
                    run_time=1.2,
                )
                capas_delta += [ln, bn]

            # Fórmula de analogía
            if show_formula and formula_mostrada is None:
                formula = self._factory_formula_analogia()
                # Anclar la fórmula debajo de los ejes en lugar de al borde inferior
                formula.next_to(ejes, DOWN, buff=0.25).set_x(0)
                self.play(FadeIn(formula[0], scale=0.88), Write(formula[1][0]),
                        run_time=0.9)
                self.play(Write(formula[1][1]))
                p_reina = ejes.c2p(*self._TOKENS["Reina"].coordenada[:2])
                self.play(
                    Indicate(vectores_en_pantalla["Reina"],
                            color=ORO_VIEJO, scale_factor=1.18),
                    Flash(p_reina, color=ORO_VIEJO, line_length=0.52, num_lines=16),
                    run_time=1.0,
                )
                formula_mostrada = formula

        # Limpiar bloque A
        salida_a  = [ejes, geo_label]
        salida_a += list(vectores_en_pantalla.values())
        salida_a += capas_delta
        if formula_mostrada:
            salida_a.append(formula_mostrada)
        self._siguiente()  # pausa entre bloque A y B
        self.play(FadeOut(*salida_a))

        # ── BLOQUE B: Lookup Table ────────────────────────────────────────────
        self._animar_pipeline_embeddings(linea)

        adornos[1].clear_updaters()
        self._siguiente()  # pausa final antes de siguiente diapo
        self.limpiar_pantalla()
    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE: Embeddings de Posición
    # ══════════════════════════════════════════════════════════════════════════


    def slide_position_embeddings(self) -> None:
        t1     = Text("Embeddings de ", font=FUENTE, font_size=42,
                      weight=BOLD, color=TINTA_NEGRA)
        t2     = Text("Posición", font=FUENTE, font_size=42,
                      weight=BOLD, color=NARANJA_TERRACOTA)
        titulo = VGroup(t1, t2).arrange(RIGHT, buff=0.08).to_edge(UP, buff=0.5)
        linea  = Line(LEFT * 6, RIGHT * 6, color=MARRON_OSCURO, stroke_width=2
                      ).next_to(titulo, DOWN, buff=0.15)

        llanuras = crear_llanuras_manchegas()
        adornos  = self._crear_adornos_esquinas(escala=0.55, buff=0.5)
        adornos[1].add_updater(lambda m, dt: m.rotate(dt * 0.4))
        self._animar_entrada_slide(titulo, linea, fondo=llanuras, adornos=adornos)

        self._acto_problema_orden(linea)
        self._acto_atencion_sin_orden(linea)
        self._acto_solucion_posicion(linea)
        self._acto_tabla_posiciones(linea)

        adornos[1].clear_updaters()
        self._siguiente()
        self.limpiar_pantalla()

    # ══════════════════════════════════════════════════════════════════════════
    # HELPERS INTERNOS
    # ══════════════════════════════════════════════════════════════════════════


    def _animar_pipeline_embeddings(self, ancla: Mobject) -> None:
        nodo_palabra  = self._factory_nodo_pipeline('"quijote"', "token de entrada",
                                                    NARANJA_TERRACOTA, 2.9)
        flecha_tok    = self._factory_flecha_anotada("tokenizar", MARRON_OSCURO, 1.1)
        nodo_id       = self._factory_nodo_pipeline("ID: 1605", "índice entero",
                                                    MARRON_OSCURO, 2.3)
        flecha_look   = self._factory_flecha_anotada("lookup", MARRON_OSCURO, 1.1)
        grupo_W, filas_W = self._factory_matriz_W(fila_activa=self._FILA_ACTIVA_W)
        flecha_res    = Arrow(ORIGIN, RIGHT * 0.9, color=VERDE_OLIVA,
                              stroke_width=2.8, max_tip_length_to_length_ratio=0.22)
        grupo_res, celdas_res = self._factory_vector_resultado(self._VALORES_EMBEDDING)

        pipeline = VGroup(nodo_palabra, flecha_tok, nodo_id,
                          flecha_look, grupo_W, flecha_res, grupo_res)
        pipeline.arrange(RIGHT, buff=0.22)
        if pipeline.width > 12.5:
            pipeline.scale(12.5 / pipeline.width)
        pipeline.next_to(ancla, DOWN, buff=1.5).set_x(0)

        for flecha, ref in [(flecha_tok, nodo_id), (flecha_look, nodo_id)]:
            flecha[0].set_y(ref.get_center()[1])
            if len(flecha) > 1:
                flecha[1].next_to(flecha[0], UP, buff=0.08)
        flecha_res.set_y(filas_W.get_center()[1])

        brace_dims     = Brace(celdas_res, direction=RIGHT, color=VERDE_OLIVA)
        brace_dims_lbl = Text("768", font="Monospace", font_size=13,
                              color=VERDE_OLIVA, weight=BOLD
                              ).next_to(brace_dims, RIGHT, buff=0.10)
        rect_fila      = SurroundingRectangle(filas_W[self._FILA_ACTIVA_W],
                                              color=ORO_VIEJO, buff=0.07,
                                              stroke_width=3, corner_radius=0.08)

        self.play(FadeIn(nodo_palabra, scale=0.92))
        self.play(GrowArrow(flecha_tok[0]),
                  FadeIn(flecha_tok[1] if len(flecha_tok) > 1 else VMobject(),
                         shift=DOWN * 0.1))
        self.play(FadeIn(nodo_id, shift=RIGHT * 0.2))
        self.play(GrowArrow(flecha_look[0]),
                  FadeIn(flecha_look[1] if len(flecha_look) > 1 else VMobject(),
                         shift=DOWN * 0.1))
        self.play(FadeIn(grupo_W, shift=LEFT * 0.14))
        self.play(Create(rect_fila))
        self.play(Indicate(nodo_id, color=ORO_VIEJO, scale_factor=1.08))
        self.play(
            GrowArrow(flecha_res),
            ReplacementTransform(rect_fila.copy(), celdas_res),
            FadeIn(grupo_res[0], shift=DOWN * 0.1),
            run_time=1.3,
        )
        self.play(FadeOut(rect_fila))
        self.play(Create(brace_dims), Write(brace_dims_lbl))
        self.play(Flash(celdas_res.get_center(), color=VERDE_OLIVA,
                        line_length=0.45, num_lines=14))



    def _acto_problema_orden(self, linea: Mobject) -> None:
        label  = Text("Sin posición el orden es invisible", font=FUENTE,
                      font_size=26, weight=BOLD, color=NARANJA_TERRACOTA
                      ).move_to(UP * 1.5)
        fila_a = self._fila_tokens(["El", "perro", "muerde", "al", "hombre"])
        fila_b = self._fila_tokens(["El", "hombre", "muerde", "al", "perro"])
        lbl_a  = Text("Frase A:", font=FUENTE, font_size=20, weight=BOLD, color=MARRON_OSCURO)
        lbl_b  = Text("Frase B:", font=FUENTE, font_size=20, weight=BOLD, color=MARRON_OSCURO)
        grupo_a = VGroup(lbl_a, fila_a).arrange(RIGHT, buff=0.3)
        grupo_b = VGroup(lbl_b, fila_b).arrange(RIGHT, buff=0.3)
        VGroup(grupo_a, grupo_b).arrange(DOWN, buff=0.8).move_to(DOWN * 0.5)

        self.play(Write(label))
        self.play(FadeIn(grupo_a, shift=RIGHT * 0.3))
        self.play(FadeIn(grupo_b, shift=RIGHT * 0.3))
        self.play(
            fila_a[1][0].animate.set_fill(ROJO_TOMATE, opacity=0.85),
            fila_a[1][1].animate.set_color(BLANCO),
            fila_b[1][0].animate.set_fill(ROJO_TOMATE, opacity=0.85),
            fila_b[1][1].animate.set_color(BLANCO),
            fila_a[4][0].animate.set_fill(VERDE_OLIVA, opacity=0.85),
            fila_a[4][1].animate.set_color(BLANCO),
            fila_b[4][0].animate.set_fill(VERDE_OLIVA, opacity=0.85),
            fila_b[4][1].animate.set_color(BLANCO),
        )
        nota = Text("Mismo bolsillo de tokens → sin orden", font=FUENTE,
                    font_size=21, color=NARANJA_TERRACOTA
                    ).next_to(VGroup(grupo_a, grupo_b), DOWN, buff=1.0)
        self.play(Write(nota))
        self._siguiente()
        self.play(FadeOut(label, grupo_a, grupo_b, nota))


    def _acto_atencion_sin_orden(self, linea: Mobject) -> None:
        label       = Text("La atención no tiene concepto de orden", font=FUENTE,
                           font_size=26, weight=BOLD, color=NARANJA_TERRACOTA
                           ).move_to(UP * 1.5)
        posiciones  = [LEFT * 2.2 + UP * 0.6, LEFT * 0.8 + DOWN * 0.6,
                       RIGHT * 0.8 + DOWN * 0.6, RIGHT * 2.2 + UP * 0.6]
        palabras    = ["perro", "muerde", "al", "hombre"]
        nodos       = VGroup()
        for pos, palabra in zip(posiciones, palabras):
            circ = Circle(radius=0.4, fill_color=PAPEL_TAN, fill_opacity=0.9,
                          stroke_color=MARRON_OSCURO, stroke_width=2)
            lbl  = Text(palabra, font=FUENTE, font_size=18, color=TINTA_NEGRA
                        ).move_to(circ.get_center())
            nodos.add(VGroup(circ, lbl).move_to(pos))

        lineas_attn = VGroup(*[
            Line(nodos[i].get_center(), nodos[j].get_center(),
                 stroke_width=1.5, stroke_color=MARRON_OSCURO, stroke_opacity=0.4)
            for i in range(len(nodos)) for j in range(i + 1, len(nodos))
        ])
        red  = VGroup(lineas_attn, nodos).move_to(DOWN * 0.5)
        nota = Text("Todos ↔ todos · sin orden", font=FUENTE, font_size=20,
                    color=MARRON_OSCURO).next_to(red, DOWN, buff=0.7)

        self.play(Write(label))
        self.play(FadeIn(red, shift=UP * 0.2))
        self.play(FadeIn(nota, shift=UP * 0.2))
        self._siguiente()
        self.play(FadeOut(label, red, nota))


    def _acto_solucion_posicion(self, linea: Mobject) -> None:
        label = Text("Posición inyectada en el vector", font=FUENTE,
                     font_size=26, weight=BOLD, color=VERDE_OLIVA
                     ).move_to(UP * 1.5)

        def _mini_vector(valores, color):
            celdas = VGroup()
            for val in valores:
                es_puntos = (val == "·")
                rect = RoundedRectangle(corner_radius=0.05, width=0.65, height=0.5,
                                        fill_color=color if not es_puntos else FONDO_CAJA,
                                        fill_opacity=0.85 if not es_puntos else 0)
                rect.set_stroke(MARRON_OSCURO if not es_puntos else FONDO_CAJA, 1.5)
                txt = Text(val, font="Monospace", font_size=15, color=TINTA_NEGRA
                           ).move_to(rect.get_center())
                celdas.add(VGroup(rect, txt))
            return celdas.arrange(RIGHT, buff=0.08)

        vec_tok = _mini_vector(["0.12", "-0.30", "0.55", "·"], PAPEL_TAN)
        vec_pos = _mini_vector(["0.84", "-0.54", "0.14", "·"], NARANJA_TERRACOTA)
        vec_sum = _mini_vector(["0.96", "-0.84", "0.69", "·"], CAJA_INFERIOR)

        token_label = VGroup(
            Text("perro", font=FUENTE, font_size=30, weight=BOLD, color=TINTA_NEGRA),
            Text("(posición 1)", font=FUENTE, font_size=20, color=PAPEL_TAN),
        ).arrange(DOWN, buff=0.15)

        lbl_tok = Text("Token Embedding\n(¿qué palabra?)", font=FUENTE, font_size=17,
                       weight=BOLD, color=MARRON_OSCURO)
        lbl_pos = Text("Position Embedding\n(¿en qué lugar?)", font=FUENTE, font_size=17,
                       weight=BOLD, color=NARANJA_TERRACOTA)
        lbl_sum = Text("Vector final =\nsignificado + posición", font=FUENTE,
                       font_size=18, weight=BOLD, color=TINTA_NEGRA)

        VGroup(vec_tok, vec_pos).arrange(DOWN, buff=0.8).move_to(DOWN * 0.2)
        mas = Text("+", font=FUENTE, font_size=40, weight=BOLD, color=TINTA_NEGRA
                   ).move_to(VGroup(vec_tok, vec_pos).get_center())

        lbl_tok.next_to(vec_tok, RIGHT, buff=0.5)
        lbl_pos.next_to(vec_pos, RIGHT, buff=0.5).align_to(lbl_tok, LEFT)
        token_label.next_to(VGroup(vec_tok, vec_pos), LEFT, buff=1.5)

        flecha_tok_arr = Arrow(token_label.get_right(), vec_tok.get_left(),
                               color=MARRON_OSCURO, buff=0.2,
                               max_tip_length_to_length_ratio=0.15)
        flecha_pos_arr = Arrow(token_label.get_right(), vec_pos.get_left(),
                               color=NARANJA_TERRACOTA, buff=0.2,
                               max_tip_length_to_length_ratio=0.15)

        self.play(Write(label))
        self.play(FadeIn(token_label))
        self.play(GrowArrow(flecha_tok_arr), FadeIn(vec_tok), Write(lbl_tok))
        self.play(GrowArrow(flecha_pos_arr), FadeIn(vec_pos), Write(lbl_pos))
        self.play(Write(mas))
        sep = Line(vec_pos.get_left() + LEFT * 0.2,
                   vec_pos.get_right() + RIGHT * 0.2,
                   color=MARRON_OSCURO, stroke_width=2
                   ).next_to(vec_pos, DOWN, buff=0.3)
        vec_sum.next_to(sep, DOWN, buff=0.3)
        lbl_sum.next_to(vec_sum, RIGHT, buff=0.5).align_to(lbl_tok, LEFT)

        self.play(Create(sep))
        self.play(
            ReplacementTransform(vec_tok.copy(), vec_sum),
            ReplacementTransform(vec_pos.copy(), vec_sum),
            Write(lbl_sum),
        )
        self.play(Indicate(vec_sum, color=NARANJA_TERRACOTA, scale_factor=1.05))
        self._siguiente()
        self.play(FadeOut(
            label, token_label, flecha_tok_arr, flecha_pos_arr,
            vec_tok, lbl_tok, vec_pos, lbl_pos, mas, sep, vec_sum, lbl_sum,
        ))


    def _acto_tabla_posiciones(self, linea: Mobject) -> None:
        titulo = Text(
            "Una fila por posición → la tabla es el Context Window",
            font=FUENTE, font_size=24, weight=BOLD, color=TINTA_NEGRA,
        ).move_to(UP * 1.5)

        def _fila_pos(num, color, opacidad=0.55):
            etiq   = Text(f"Pos {num}:", font=FUENTE, font_size=18, color=TINTA_NEGRA)
            celdas = VGroup(*[
                RoundedRectangle(corner_radius=0.03, width=0.42, height=0.32,
                                 fill_color=color, fill_opacity=opacidad,
                                 ).set_stroke(MARRON_OSCURO, 1)
                for _ in range(8)
            ]).arrange(RIGHT, buff=0.05)
            puntos = Text("···", font_size=14, color=MARRON_OSCURO
                          ).next_to(celdas, RIGHT, buff=0.1)
            return VGroup(etiq, celdas, puntos).arrange(RIGHT, buff=0.3)

        f0 = _fila_pos(0,    PAPEL_TAN,          0.7)
        f1 = _fila_pos(1,    NARANJA_TERRACOTA,   0.5)
        f2 = _fila_pos(2,    PAPEL_TAN,           0.5)
        fv = Text("·  ·  ·", font_size=28, color=MARRON_OSCURO)
        fn = _fila_pos(1023, CAJA_INFERIOR,        0.5)

        tabla     = VGroup(f0, f1, f2, fv, fn).arrange(DOWN, buff=0.28)
        llave     = Brace(tabla, direction=LEFT, color=MARRON_OSCURO)
        lbl_llave = Text("1 024 posiciones\n(block_size)", font=FUENTE,
                         font_size=18, color=MARRON_OSCURO, line_spacing=1.2
                         ).next_to(llave, LEFT, buff=0.2)
        nota_limite = Text("Sin fila 1024 → fuera del contexto", font=FUENTE,
                           font_size=21, weight=BOLD, color=NARANJA_TERRACOTA)

        VGroup(
            VGroup(lbl_llave, llave, tabla),
            nota_limite,
        ).arrange(RIGHT, buff=0.8).next_to(titulo, DOWN, buff=0.6)

        caja_fn = SurroundingRectangle(fn, color=NARANJA_TERRACOTA,
                                       stroke_width=4, buff=0.07, corner_radius=0.06)

        self.play(Write(titulo))
        self.play(FadeIn(tabla, shift=UP * 0.3))
        self.play(GrowFromCenter(llave), Write(lbl_llave))
        self.play(Create(caja_fn))
        self.play(Write(nota_limite))
        self.play(Indicate(caja_fn, color=NARANJA_TERRACOTA, scale_factor=1.05))

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
