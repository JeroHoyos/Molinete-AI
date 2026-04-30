import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
from slides.slides_00_inicio import SlidesInicio
from slides.slides_01_intro import SlidesIntro
from slides.slides_02_tensores import SlidesTensores
from slides.slides_03_tokenizacion import SlidesTokenizacion
from slides.slides_04_embeddings import SlidesEmbeddings
from slides.slides_05_atencion import SlidesAtencion
from slides.slides_06_mlp import SlidesMLP
from slides.slides_07_entrenamiento import SlidesEntrenamiento
from slides.slides_08_final import SlidesFinal


class Presentacion(SlidesInicio, SlidesIntro, SlidesTensores, SlidesTokenizacion,
                   SlidesEmbeddings, SlidesAtencion, SlidesMLP, SlidesEntrenamiento,
                   SlidesFinal, Slide):

    _LANG_POR_ARCHIVO = {
        "python_bindings.rs": "rust",
        "BDPtokenizer.rs":    "rust",
        "pair_counts.rs":     "rust",
    }

    def construct(self):
        self._slide_actual = 0
        self._TOTAL_SLIDES = 50
        self.camera.background_color = WHITE

        self.slide_pronto_iniciamos()
        self.slide_introduction()
        self.slide_credits()
        self.slide_que_es_transformer()
        self.slide_por_que_rust()
        self.slide_molinete_ai()

        self.slide_roadmap()

        self.slide_que_es_un_tensor()
        self.mostrar_snippet("tensor.rs")
        self.slide_strides()

        self.slide_matmul()
        self.mostrar_snippet("matmul_base.rs")

        self.slide_simd()
        self.mostrar_snippet("simd_vectorization.rs")

        self.slide_cache_blocking()
        self.mostrar_snippet("cache_blocking.rs")

        self.slide_parallel()
        self.mostrar_snippet("parallel.rs")

        self.slide_batched_matmul()
        self.mostrar_snippet("batched_matmul.rs")

        self.slide_softmax()
        self.mostrar_snippet("softmax.rs")

        self.slide_problema_strawberry()
        self.slide_tokenizacion()
        self.mostrar_snippet("BDPtokenizer.rs")

        self.slide_byte_pair_encoding()
        self.mostrar_snippet("pair_counts.rs")

        self.slide_embeddings()
        self.slide_position_embeddings()
        self.mostrar_snippet("embedding.rs")

        self.slide_mha_acto1_intuicion()
        self.slide_mha_acto2_qkv()
        self.slide_mha_acto3_formula_y_flujo()
        self.slide_mha_acto4_multihead()
        self.mostrar_snippet("attention.rs")

        self.slide_arquitectura_neurona()
        self.slide_zoom_neurona()
        self.mostrar_snippet("mlp_forward.rs")
        self.slide_activacion()
        self.mostrar_snippet("gelu.rs")

        self.slide_layer_normalization()
        self.mostrar_snippet("normalization.rs")

        self.slide_residual()
        self.slide_capa_transformer()
        self.mostrar_snippet("block_backward.rs")

        self.slide_entrenamiento()
        self.slide_training_metrics()
        self.mostrar_snippet("compute_loss.rs")

        self.slide_descenso_gradiente()
        self.slide_backpropagation()
        self.mostrar_snippet("linear_backward.rs")

        self.slide_adam()
        self.mostrar_snippet("adamw_update.rs")

        self.slide_dropout()
        self.mostrar_snippet("dropout.rs")

        self.slide_temperature()
        self.mostrar_snippet("temperature.rs")

        self.slide_rust_python_bridge()
        self.slide_model_in_action()
        self.slide_final()

    def mostrar_snippet(self, titulo_archivo: str):
        self.diapo_codigo(
            codigo_fuente=RUST_SNIPPETS[titulo_archivo],
            titulo_archivo=titulo_archivo,
        )
        self.limpiar_pantalla()

    def _llanuras(self) -> VGroup:
        return crear_llanuras_manchegas()

    def crear_titulo(self, texto, palabra_clave=None, color_clave=NARANJA_TERRACOTA, font_size=35):
        self._slide_actual += 1
        t2c = {palabra_clave: color_clave} if palabra_clave else {}
        titulo = Text(texto, font=FUENTE, font_size=font_size, color=TINTA_NEGRA, t2c=t2c).to_edge(UP)
        linea = Underline(titulo, color=color_clave, stroke_width=4)
        contador = self._contador_slide(self._slide_actual)
        self.add(contador)
        return titulo, linea

    def crear_bloque(self, texto="", color_fondo=FONDO_CAJA, color_texto=TINTA_NEGRA, ancho=0.8, alto=0.8):
        rect = RoundedRectangle(
            corner_radius=0.15, width=ancho, height=alto,
            fill_color=color_fondo, fill_opacity=1,
            stroke_color=MARRON_OSCURO, stroke_width=2
        )
        lbl = Text(str(texto), font=FUENTE, font_size=24, color=color_texto).move_to(rect.get_center())
        return VGroup(rect, lbl)

    def crear_matriz_bloques(self, filas, columnas, color_fondo=FONDO_CAJA, color_texto=TINTA_NEGRA, valores=None, ancho=0.8, alto=0.8):
        if valores is None:
            valores = [""] * (filas * columnas)
        matriz = VGroup()
        idx = 0
        for i in range(filas):
            fila_bloques = VGroup()
            for j in range(columnas):
                texto = valores[idx] if idx < len(valores) else ""
                bloque = self.crear_bloque(
                    texto=texto,
                    color_fondo=color_fondo,
                    color_texto=color_texto,
                    ancho=ancho,
                    alto=alto
                )
                fila_bloques.add(bloque)
                idx += 1
            fila_bloques.arrange(RIGHT, buff=0.05)
            matriz.add(fila_bloques)
        matriz.arrange(DOWN, buff=0.05)
        return matriz

    def limpiar_pantalla(self, wait_previo: float = 0.0):
        if wait_previo > 0:
            self.wait(wait_previo)
        for mob in self.mobjects:
            mob.clear_updaters()
            for submob in mob.get_family():
                submob.clear_updaters()
        if self.mobjects:
            self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.5)

    def _siguiente(self):
        dot = Dot(radius=0.11, color=NARANJA_TERRACOTA, fill_opacity=1, z_index=100)
        dot.to_corner(DR, buff=0.3)
        ring = Circle(
            radius=0.11, color=NARANJA_TERRACOTA,
            stroke_width=2.2, fill_opacity=0, z_index=99
        ).move_to(dot)
        self.add(dot, ring)
        self.play(
            ring.animate.scale(2.8).set_stroke(opacity=0),
            dot.animate.set_opacity(0.5),
            run_time=0.4, rate_func=linear
        )
        self.play(dot.animate.set_opacity(1.0), run_time=0.2, rate_func=linear)
        self.next_slide()
        self.play(FadeOut(dot, ring), run_time=0.2)

    def diapo_codigo(self, codigo_fuente: str, titulo_archivo: str = "codigo.rs"):
        if titulo_archivo in self._LANG_POR_ARCHIVO:
            lang = self._LANG_POR_ARCHIVO[titulo_archivo]
        elif titulo_archivo.endswith(".py"):
            lang = "python"
        elif titulo_archivo.endswith(".toml"):
            lang = "toml"
        else:
            lang = "rust"

        alto_header  = 0.6
        padding_x    = 0.6
        padding_y    = 0.5
        MARGEN_PANTALLA_X = 0.3
        MARGEN_PANTALLA_Y = 0.3

        MAX_CAJA_ANCHO = config.frame_width  - 2 * MARGEN_PANTALLA_X
        MAX_CAJA_ALTO  = config.frame_height - 2 * MARGEN_PANTALLA_Y
        MAX_CODIGO_ANCHO = MAX_CAJA_ANCHO - padding_x
        MAX_CODIGO_ALTO  = MAX_CAJA_ALTO  - padding_y - alto_header

        bloque_codigo = Code(
            code_string=codigo_fuente,
            language=lang,
            formatter_style=EstiloCervantino,
            background="rectangle"
        ).scale(0.8)

        if bloque_codigo.width > MAX_CODIGO_ANCHO:
            bloque_codigo.scale_to_fit_width(MAX_CODIGO_ANCHO)
        if bloque_codigo.height > MAX_CODIGO_ALTO:
            bloque_codigo.scale_to_fit_height(MAX_CODIGO_ALTO)
        if bloque_codigo.width > MAX_CODIGO_ANCHO:
            bloque_codigo.scale_to_fit_width(MAX_CODIGO_ANCHO)

        if len(bloque_codigo) > 0:
            bloque_codigo[0].set_opacity(0)
        if len(bloque_codigo) > 1:
            bloque_codigo[1].set_color(PAPEL_TAN)

        ancho_caja = min(bloque_codigo.width + padding_x, MAX_CAJA_ANCHO)
        ancho_caja = max(ancho_caja, 6.0)
        alto_caja  = bloque_codigo.height + padding_y + alto_header

        sombra = RoundedRectangle(
            corner_radius=0.1,
            width=ancho_caja, height=alto_caja,
            color=NEGRO_SUAVE, fill_color=NEGRO_SUAVE, fill_opacity=0.25, stroke_width=0
        ).shift(DOWN * 0.15 + RIGHT * 0.15)

        editor_bg = RoundedRectangle(
            corner_radius=0.1,
            width=ancho_caja, height=alto_caja,
            color=MARRON_OSCURO, stroke_width=3,
            fill_color=FONDO_CAJA, fill_opacity=1,
        )

        editor_header = Rectangle(
            width=ancho_caja, height=alto_header,
            color=MARRON_OSCURO, stroke_width=3,
            fill_color=MARRON_OSCURO, fill_opacity=1,
        ).align_to(editor_bg, UP)

        dot_1 = Circle(radius=0.08, color=TINTA_NEGRA, fill_color=ROJO_MAC,    fill_opacity=1, stroke_width=1.5)
        dot_2 = Circle(radius=0.08, color=TINTA_NEGRA, fill_color=AMARILLO_MAC, fill_opacity=1, stroke_width=1.5)
        dot_3 = Circle(radius=0.08, color=TINTA_NEGRA, fill_color=VERDE_MAC,   fill_opacity=1, stroke_width=1.5)

        botones = VGroup(dot_1, dot_2, dot_3).arrange(RIGHT, buff=0.2)
        botones.move_to(editor_header.get_left() + RIGHT * 0.5)

        file_title = Text(
            titulo_archivo, font="Times New Roman", font_size=20,
            color=PAPEL_CREMA, weight=BOLD
        ).move_to(editor_header)

        editor_ui = VGroup(sombra, editor_bg, editor_header, botones, file_title)
        editor_ui.move_to(ORIGIN)
        area_util = editor_bg.get_center() + DOWN * (alto_header / 2)
        bloque_codigo.move_to(area_util)

        self.play(
            FadeIn(sombra, shift=UP * 0.1),
            DrawBorderThenFill(editor_bg),
            run_time=0.8
        )
        self.play(
            FadeIn(editor_header, shift=DOWN * 0.2),
            FadeIn(botones, shift=RIGHT * 0.1),
            Write(file_title),
            run_time=0.6
        )

        lineas_codigo  = list(bloque_codigo[2]) if len(bloque_codigo) > 2 else []
        numeros_codigo = list(bloque_codigo[1]) if len(bloque_codigo) > 1 else []

        animaciones_lineas = [
            FadeIn(VGroup(num, lin), shift=UP * 0.15, scale=0.95)
            for num, lin in zip(numeros_codigo, lineas_codigo)
        ]
        tiempo_animacion_codigo = max(1.5, len(animaciones_lineas) * 0.08)
        self.play(LaggedStart(*animaciones_lineas, lag_ratio=0.1), run_time=tiempo_animacion_codigo)

        self.next_slide()

        self.play(
            FadeOut(VGroup(editor_ui, bloque_codigo), shift=DOWN * 0.3),
            run_time=0.8
        )

    def mostrar_snippet_python_bindings(self):
        self.diapo_codigo(
            codigo_fuente=RUST_SNIPPETS["python_bindings.rs"],
            titulo_archivo="python_bindings.rs",
        )
        self.limpiar_pantalla()

    def _crear_adornos_esquinas(self, escala=0.6, buff=0.8):
        molino    = crear_molino().scale(escala).to_corner(DL, buff=buff)
        sol       = crear_sol_cervantino().scale(escala * 1.55).to_corner(UR, buff=buff)
        pergamino = crear_pergamino().scale(escala).to_corner(UL, buff=buff)
        yelmo     = crear_yelmo_mambrino().scale(escala * 1.2).to_corner(DR, buff=buff)
        return VGroup(molino, sol, pergamino, yelmo)

    def _animar_entrada_slide(self, titulo, linea, adornos=None, fondo=None):
        animaciones = [Write(titulo), GrowFromCenter(linea)]
        if fondo is not None:
            animaciones.append(FadeIn(fondo, shift=UP * 0.3))
        if adornos is not None:
            animaciones.append(
                LaggedStart(
                    *[FadeIn(a, scale=0.5) for a in adornos],
                    lag_ratio=0.2
                )
            )
        self.play(*animaciones)

    def _crear_burbuja_chat(self, texto, color_fondo, color_texto, es_usuario=True, t2c_dict=None):
        txt = Text(texto, font=FUENTE, font_size=24, color=color_texto, t2c=t2c_dict)
        fondo = RoundedRectangle(
            width=txt.width + 0.8,
            height=txt.height + 0.5,
            corner_radius=0.2,
            fill_color=color_fondo, fill_opacity=1,
            stroke_width=0 if es_usuario else 1.5,
            stroke_color=MARRON_OSCURO
        )
        sombra = fondo.copy().set_fill(MARRON_OSCURO, 0.1).set_stroke(width=0).shift(RIGHT * 0.05 + DOWN * 0.05)
        txt.move_to(fondo.get_center())
        burbuja_base = VGroup(sombra, fondo, txt)
        remitente = Text(
            "Tú" if es_usuario else "Molinete AI",
            font=FUENTE, font_size=16,
            color=MARRON_OSCURO, weight=BOLD
        )
        if es_usuario:
            remitente.next_to(burbuja_base, UP, buff=0.2, aligned_edge=RIGHT)
        else:
            remitente.next_to(burbuja_base, UP, buff=0.1, aligned_edge=LEFT)
        return VGroup(remitente, burbuja_base)

    def _crear_fresa(self):
        cuerpo = Polygon(
            [0, 0.3, 0], [-0.25, 0.1, 0], [-0.2, -0.2, 0],
            [0, -0.4, 0], [0.2, -0.2, 0], [0.25, 0.1, 0],
            fill_color=ROJO_TOMATE, fill_opacity=1,
            stroke_width=1.5, stroke_color=MARRON_OSCURO
        )
        hojas = Polygon(
            [0, 0.2, 0], [-0.2, 0.4, 0], [-0.1, 0.25, 0],
            [0, 0.45, 0], [0.1, 0.25, 0], [0.2, 0.4, 0],
            fill_color=VERDE_OLIVA, fill_opacity=1,
            stroke_width=1.5, stroke_color=MARRON_OSCURO
        )
        return VGroup(cuerpo, hojas).scale(0.85)

    def _crear_burbuja_transformer(self, texto, es_usuario=True):
        color_fondo = CAJA_INFERIOR if es_usuario else FONDO_CAJA
        color_borde = MARRON_OSCURO if es_usuario else NARANJA_TERRACOTA
        txt = Text(texto, font=FUENTE, font_size=22, color=TINTA_NEGRA)
        burbuja = RoundedRectangle(
            width=max(txt.width + 0.6, 1.8), height=txt.height + 0.5, corner_radius=0.2,
            fill_color=color_fondo, fill_opacity=1, stroke_color=color_borde, stroke_width=2
        )
        txt.move_to(burbuja.get_center())
        return VGroup(burbuja, txt)

    def _animar_pensamiento_transformer(self, texto_ganador, texto_perdedor, pct_ganador, pct_perdedor, obj_referencia):
        opcion_ganadora  = Text(f"{texto_ganador} ({pct_ganador})",  font=FUENTE, font_size=20, weight=BOLD, color=NARANJA_TERRACOTA)
        opcion_perdedora = Text(f"{texto_perdedor} ({pct_perdedor})", font=FUENTE, font_size=16, color=ACERO)
        textos_menu = VGroup(opcion_ganadora, opcion_perdedora).arrange(DOWN, buff=0.15)
        caja_menu = SurroundingRectangle(
            textos_menu, color=NARANJA_TERRACOTA, stroke_width=2,
            fill_color=FONDO_CAJA, fill_opacity=0.97, corner_radius=0.1, buff=0.25
        )
        menu_final = VGroup(caja_menu, textos_menu).next_to(obj_referencia, RIGHT, buff=0.5)
        self.play(Wiggle(obj_referencia))
        self.play(FadeIn(menu_final, shift=RIGHT * 0.3))
        self.play(Indicate(opcion_ganadora, scale_factor=1.2, color=NARANJA_TERRACOTA))
        self.play(FadeOut(menu_final))

    def _crear_item_utilidad(self, titulo_item, descripcion):
        t_titulo = Text(titulo_item, font=FUENTE, font_size=26, color=NARANJA_TERRACOTA, weight=BOLD)
        t_desc   = Text(descripcion,  font=FUENTE, font_size=22, color=TINTA_NEGRA)
        grupo_texto = VGroup(t_titulo, t_desc).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        marcador = Rectangle(width=0.08, height=grupo_texto.height, fill_color=MARRON_OSCURO, fill_opacity=1, stroke_width=0)
        return VGroup(marcador, grupo_texto).arrange(RIGHT, buff=0.25)

    def _crear_panel_probabilidades(self, palabra_objetivo):
        dummies = ["hidalgo", "espada", "vino", "capa", "oro", "plaza", "mujer", "caballo"]
        random.shuffle(dummies)
        probs = [random.uniform(0.7, 0.9), random.uniform(0.05, 0.1), random.uniform(0.01, 0.03)]
        datos = [
            (palabra_objetivo, probs[0], NARANJA_TERRACOTA),
            (dummies[0],       probs[1], PAPEL_TAN),
            (dummies[1],       probs[2], CAJA_INFERIOR),
        ]
        txts = VGroup(*[
            Text(d[0], font=FUENTE, font_size=16, color=TINTA_NEGRA) for d in datos
        ]).arrange(DOWN, aligned_edge=RIGHT, buff=0.15)
        bars = VGroup(*[
            RoundedRectangle(
                corner_radius=0.05, height=0.15,
                width=max(0.1, d[1] * 2.5),
                fill_color=d[2], fill_opacity=1, stroke_width=0
            ) for d in datos
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        for i in range(len(datos)):
            bars[i].match_y(txts[i])
        col = VGroup(txts, bars).arrange(RIGHT, buff=0.15)
        bg = RoundedRectangle(
            corner_radius=0.1, width=col.width + 0.5,
            height=txts[0].height * 3 + 0.6,
            fill_color=CAJA_INFERIOR, fill_opacity=0.3,
            stroke_color=MARRON_OSCURO, stroke_width=1
        ).move_to(col)
        return VGroup(bg, col).move_to(LEFT * 0.5 + DOWN * 0.4)

    def _crear_fila_vocabulario(self, vocab, total_tokens, compresion, es_final=False):
        color_c = NARANJA_TERRACOTA if es_final else TINTA_NEGRA
        peso_c  = BOLD              if es_final else NORMAL
        return VGroup(
            Text(vocab,        font=FUENTE, font_size=24, color=TINTA_NEGRA),
            Text(total_tokens, font=FUENTE, font_size=24, color=TINTA_NEGRA),
            Text(compresion,   font=FUENTE, font_size=24, color=color_c, weight=peso_c),
        )

    def _hacer_editor(self, codigo: str, lang: str, titulo_arch: str,
                      escala: float = 0.62, ancho_max: float = 5.8) -> VGroup:
        bloque = Code(
            code_string=codigo,
            language=lang,
            formatter_style=EstiloCervantino,
            background="rectangle",
        ).scale(escala)
        if bloque.width > ancho_max:
            bloque.scale_to_fit_width(ancho_max)
        if len(bloque) > 0:
            bloque[0].set_opacity(0)
        if len(bloque) > 1:
            bloque[1].set_color(PAPEL_TAN)
        ancho_caja = max(bloque.width + 0.4, 4.5)
        alto_caja  = bloque.height + 0.75
        bg = RoundedRectangle(
            corner_radius=0.1, width=ancho_caja, height=alto_caja,
            fill_color=FONDO_CAJA, fill_opacity=1,
            stroke_color=MARRON_OSCURO, stroke_width=2,
        )
        header = Rectangle(
            width=ancho_caja, height=0.38,
            fill_color=MARRON_OSCURO, fill_opacity=1, stroke_width=0,
        ).align_to(bg, UP)
        dots = VGroup(
            Circle(radius=0.06, fill_color=ROJO_MAC,     fill_opacity=1, stroke_width=0),
            Circle(radius=0.06, fill_color=AMARILLO_MAC, fill_opacity=1, stroke_width=0),
            Circle(radius=0.06, fill_color=VERDE_MAC,    fill_opacity=1, stroke_width=0),
        ).arrange(RIGHT, buff=0.13).move_to(header.get_left() + RIGHT * 0.4)
        ftitle = Text(titulo_arch, font="Monospace", font_size=13,
                      color=PAPEL_CREMA).move_to(header)
        bloque.move_to(bg.get_center() + DOWN * 0.12)
        return VGroup(bg, header, dots, ftitle, bloque)

    def _contador_slide(self, numero: int) -> VGroup:
        total = self._TOTAL_SLIDES
        txt = Text(f"{numero} / {total}", font=FUENTE, font_size=14, color=MARRON_OSCURO)
        barra_fondo = Rectangle(
            width=1.8, height=0.06,
            fill_color=CAJA_INFERIOR, fill_opacity=1, stroke_width=0
        )
        progreso = min(numero / total, 1.0)
        barra_llena = Rectangle(
            width=max(0.05, 1.8 * progreso), height=0.06,
            fill_color=NARANJA_TERRACOTA, fill_opacity=1, stroke_width=0
        ).align_to(barra_fondo, LEFT)
        barra = VGroup(barra_fondo, barra_llena)
        grupo = VGroup(txt, barra).arrange(DOWN, buff=0.06)
        grupo.to_corner(DL, buff=0.25)
        return grupo

    def _crear_pergamino_decorativo(self, ancho: float, alto: float) -> VGroup:
        sombra = RoundedRectangle(
            corner_radius=0.18, width=ancho + 0.12, height=alto + 0.12,
            fill_color=MADERA_OSCURA, fill_opacity=0.25, stroke_width=0
        ).shift(DR * 0.08)
        exterior = RoundedRectangle(
            corner_radius=0.18, width=ancho, height=alto,
            fill_color=PERGAMINO_CLARO, fill_opacity=0.97,
            stroke_color=MARRON_OSCURO, stroke_width=2.2
        )
        interior = RoundedRectangle(
            corner_radius=0.12, width=ancho - 0.22, height=alto - 0.22,
            fill_opacity=0,
            stroke_color=NARANJA_TERRACOTA, stroke_width=0.8, stroke_opacity=0.6
        )
        def _ornamento(pos):
            petal_v = Ellipse(width=0.12, height=0.22, fill_color=OCRE_CERVANTINO,
                              fill_opacity=0.85, stroke_width=0)
            petal_h = Ellipse(width=0.22, height=0.12, fill_color=OCRE_CERVANTINO,
                              fill_opacity=0.85, stroke_width=0)
            centro = Dot(radius=0.045, color=MARRON_OSCURO)
            return VGroup(petal_v, petal_h, centro).move_to(pos)
        esquinas = VGroup(
            _ornamento(exterior.get_corner(UL) + DR * 0.22),
            _ornamento(exterior.get_corner(UR) + DL * 0.22),
            _ornamento(exterior.get_corner(DL) + UR * 0.22),
            _ornamento(exterior.get_corner(DR) + UL * 0.22),
        )
        separador = Line(
            exterior.get_left() + RIGHT * 0.35,
            exterior.get_right() + LEFT * 0.35,
            stroke_color=NARANJA_TERRACOTA, stroke_width=1.2, stroke_opacity=0.6
        ).align_to(exterior, UP).shift(DOWN * 0.42)
        return VGroup(sombra, exterior, interior, esquinas, separador)

    def _crear_panel_probs_rico(self, palabra_objetivo: str, posicion=None) -> VGroup:
        dummies = ["hidalgo", "espada", "vino", "capa", "oro", "plaza", "mujer", "caballo"]
        random.shuffle(dummies)
        p0 = random.uniform(0.72, 0.92)
        p1 = random.uniform(0.04, 0.12)
        p2 = random.uniform(0.01, 0.04)
        datos = [
            (palabra_objetivo, p0, NARANJA_TERRACOTA),
            (dummies[0],       p1, PAPEL_TAN),
            (dummies[1],       p2, CAJA_INFERIOR),
        ]
        filas = []
        for etiqueta, prob, color in datos:
            txt   = Text(etiqueta, font=FUENTE, font_size=15, color=TINTA_NEGRA)
            barra = RoundedRectangle(
                corner_radius=0.05, height=0.15,
                width=max(0.1, prob * 2.0),
                fill_color=color, fill_opacity=1, stroke_width=0
            )
            pct   = Text(f"{prob*100:.0f}%", font=FUENTE, font_size=13, color=MARRON_OSCURO)
            fila  = VGroup(txt, barra, pct).arrange(RIGHT, buff=0.12, aligned_edge=DOWN)
            filas.append(fila)
        contenido = VGroup(*filas).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        bg = RoundedRectangle(
            corner_radius=0.12,
            width=contenido.width + 0.5, height=contenido.height + 0.4,
            fill_color=FONDO_CAJA, fill_opacity=0.97,
            stroke_color=MARRON_OSCURO, stroke_width=1.4
        ).move_to(contenido)
        panel = VGroup(bg, contenido)
        if posicion is not None:
            panel.move_to(posicion)
        return panel

    def _crear_corazon(self, color, escala=1.0):
        corazon = ParametricFunction(
            lambda t: np.array([
                16 * np.sin(t) ** 3,
                13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t),
                0
            ]),
            t_range=[0, TAU],
            color=color,
            stroke_width=2
        ).scale(0.03 * escala)
        corazon.set_fill(color, opacity=0.7)
        return corazon

    def _sub_slide_lenguaje(self, titulo, palabra_clave, logo_path, pros, contras,
                             adornos, fondo, wiggle_contras=None):
        titulo_obj, linea = self.crear_titulo(
            titulo, palabra_clave=palabra_clave, color_clave=MARRON_OSCURO
        )
        self._animar_entrada_slide(titulo_obj, linea, adornos=adornos, fondo=fondo)
        logo = ImageMobject(os.path.join("assets", logo_path)).set_height(2.6)
        items = []
        for texto, t2c in pros:
            items.append(Text(f"✓ {texto}", font=FUENTE, font_size=26,
                              color=VERDE_OLIVA, t2c=t2c))
        contras_objs = []
        for texto, t2c in contras:
            obj = Text(f"✗ {texto}", font=FUENTE, font_size=26,
                       color=ROJO_CONTRA, t2c=t2c)
            items.append(obj)
            contras_objs.append(obj)
        grupo = VGroup(*items).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        logo.next_to(linea, DOWN, buff=0.7).to_edge(LEFT, buff=1.0)
        grupo.next_to(logo, RIGHT, buff=0.9).set_y(logo.get_center()[1])
        self.play(FadeIn(logo, shift=DOWN * 0.4), run_time=0.8)
        self.play(FadeIn(items[0], shift=RIGHT * 0.3), run_time=0.6)
        self._siguiente()
        self.play(
            LaggedStart(*[FadeIn(c, shift=RIGHT * 0.3) for c in contras_objs], lag_ratio=0.4),
            run_time=1.2,
        )
        if wiggle_contras:
            self.play(*[Wiggle(contras_objs[i]) for i in wiggle_contras], run_time=1.0)
        self._siguiente()
        self.limpiar_pantalla()

    def _sub_slide_rust(self, adornos, fondo):
        titulo_obj, linea = self.crear_titulo(
            "Rust al rescate", palabra_clave="Rust", color_clave=NARANJA_TERRACOTA
        )
        self._animar_entrada_slide(titulo_obj, linea, adornos=adornos, fondo=fondo)
        ferris  = ImageMobject(os.path.join("assets", "logo_rust.png")).set_height(2.0)
        quijote = ImageMobject(os.path.join("assets", "quijote_rust.png")).set_height(2.0)
        sancho  = ImageMobject(os.path.join("assets", "sancho_rust.png")).set_height(2.0)
        cangrejos = Group(quijote, ferris, sancho).arrange(RIGHT, buff=0.5)
        cangrejos.next_to(linea, DOWN, buff=0.5).to_edge(RIGHT, buff=0.6)
        ventajas_data = [
            ("✓ Rendimiento comparable a C++",          {}),
            ("✓ Memoria segura sin GC ni GIL",          {"GC": NARANJA_TERRACOTA, "GIL": NARANJA_TERRACOTA}),
            ("✓ Diferenciador: casi nadie lo usa en AI", {}),
        ]
        ventajas_objs = [
            Text(texto, font=FUENTE, font_size=24, color=TINTA_NEGRA, t2c=t2c)
            for texto, t2c in ventajas_data
        ]
        ventajas = VGroup(*ventajas_objs).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        ventajas.next_to(linea, DOWN, buff=0.5).to_edge(LEFT, buff=0.7)
        ventajas.set_y(cangrejos.get_center()[1])
        self.play(
            LaggedStart(
                FadeIn(quijote, shift=UP * 0.4),
                FadeIn(ferris,  shift=UP * 0.6),
                FadeIn(sancho,  shift=UP * 0.4),
                lag_ratio=0.35,
            ),
            run_time=1.2,
        )
        self._siguiente()
        self.play(
            LaggedStart(*[FadeIn(v, shift=RIGHT * 0.3) for v in ventajas_objs], lag_ratio=0.4),
            run_time=1.4,
        )
        self._siguiente()
        caja = RoundedRectangle(
            corner_radius=0.2,
            width=ventajas.width + 0.8,
            height=ventajas.height + 0.6,
            stroke_color=NARANJA_TERRACOTA,
            stroke_width=4,
            fill_color=FONDO_CAJA,
            fill_opacity=0.3,
        ).move_to(ventajas)
        self.play(
            Create(caja),
            *[v.animate.set_color(NARANJA_TERRACOTA) for v in ventajas_objs],
        )
        self.play(
            Indicate(ferris,  color=NARANJA_TERRACOTA, scale_factor=1.15),
            Indicate(quijote, color=NARANJA_TERRACOTA, scale_factor=1.10),
            Indicate(sancho,  color=NARANJA_TERRACOTA, scale_factor=1.10),
            run_time=1.0,
        )
        self._siguiente()
        self.limpiar_pantalla()
