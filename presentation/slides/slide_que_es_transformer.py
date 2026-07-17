import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import os
from colores import *
from objetos import *


class SlideQueEsTransformer:
    def slide_que_es_transformer(self):

        titulo, linea = self.crear_titulo("¿Qué es un Transformer?", palabra_clave="Transformer", color_clave=NARANJA_TERRACOTA)
        adornos = self._crear_adornos_esquinas(escala=0.8)

        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, adornos=adornos, fondo=llanuras_fondo)

        frase_1 = Text(
            "Un Transformer es una máquina que predice.",
            font=FUENTE, font_size=26, color=TINTA_NEGRA,
            t2c={"predice": NARANJA_TERRACOTA}
        ).move_to(UP * 2.5)

        frase_2 = Text(
            "Dada una secuencia de palabras...",
            font=FUENTE, font_size=24, color=TINTA_NEGRA,
        ).next_to(frase_1, DOWN, buff=0.35)

        frase_3 = Text(
            "...adivina cuál viene a continuación.",
            font=FUENTE, font_size=24, color=TINTA_NEGRA,
            t2c={"cuál viene a continuación": NARANJA_TERRACOTA}
        ).next_to(frase_2, DOWN, buff=0.25)

        self.play(FadeIn(frase_1, shift=UP * 0.2))
        self.play(
            LaggedStart(
                FadeIn(frase_2, shift=UP * 0.15),
                FadeIn(frase_3, shift=UP * 0.15),
                lag_ratio=0.45,
            ),
            run_time=1.2,
        )

        texto_escrito = ["Hoy", "no", "puedo", "quedar,", "estoy", "muy"]
        tokens_burbuja = VGroup(*[
            Text(w, font=FUENTE, font_size=22, color=TINTA_NEGRA)
            for w in texto_escrito
        ]).arrange(RIGHT, buff=0.18)

        cursor = Rectangle(
            width=0.04, height=0.28,
            fill_color=NARANJA_TERRACOTA, fill_opacity=1, stroke_width=0
        ).next_to(tokens_burbuja, RIGHT, buff=0.06)

        contenido_burbuja = VGroup(tokens_burbuja, cursor)

        burbuja = RoundedRectangle(
            corner_radius=0.18, width=contenido_burbuja.width + 1.9, height=0.75,
            fill_color=FONDO_CAJA,
            fill_opacity=1,
            stroke_color=TIERRA_MANCHEGA, stroke_width=1.5
        )
        contenido_burbuja.next_to(burbuja.get_left(), RIGHT, buff=0.35)
        grupo_burbuja = VGroup(burbuja, contenido_burbuja)

        label_movil = Text(
            "Tu móvil:", font=FUENTE, font_size=18, color=TINTA_NEGRA, slant=ITALIC
        )
        bloque_burbuja = VGroup(label_movil, grupo_burbuja)\
            .arrange(RIGHT, buff=0.35).next_to(frase_3, DOWN, buff=0.6)

        datos_sug = [
            ("cansado", NARANJA_TERRACOTA, NARANJA_TERRACOTA, "92%"),
            ("ocupado",  ACERO,            METAL_CLARO,       "6%"),
            ("bien",     ACERO,            METAL_CLARO,       "2%"),
        ]

        cajas_sug = VGroup()
        for palabra, color_txt, color_borde, pct in datos_sug:
            rect = RoundedRectangle(
                corner_radius=0.1, width=2.0, height=0.65,
                fill_color=FONDO_CAJA if color_txt == NARANJA_TERRACOTA else PAPEL_CREMA,
                fill_opacity=1,
                stroke_color=color_borde, stroke_width=1.5
            )
            txt = Text(
                palabra, font=FUENTE, font_size=20, color=color_txt,
                weight=BOLD if color_txt == NARANJA_TERRACOTA else NORMAL
            ).move_to(rect).shift(DOWN * 0.07)
            pct_txt = Text(
                pct, font=FUENTE, font_size=14, color=color_borde
            ).next_to(txt, UP, buff=0.05)
            cajas_sug.add(VGroup(rect, txt, pct_txt))

        cajas_sug.arrange(RIGHT, buff=0.3).next_to(bloque_burbuja, DOWN, buff=0.35)

        caja_todo = SurroundingRectangle(
            VGroup(bloque_burbuja, cajas_sug),
            color=NARANJA_TERRACOTA, corner_radius=0.15,
            buff=0.3, stroke_width=1.5,
            fill_color=MARRON_OSCURO, fill_opacity=0.05
        )

        self.play(FadeIn(caja_todo))
        self.play(
            FadeIn(label_movil), FadeIn(burbuja),
            LaggedStart(*[FadeIn(t, shift=UP*0.1) for t in tokens_burbuja], lag_ratio=0.1),
            run_time=1.2
        )
        for _ in range(2):
            self.play(FadeOut(cursor, run_time=0.3))
            self.play(FadeIn(cursor, run_time=0.3))

        self.play(
            LaggedStart(*[FadeIn(c, shift=UP*0.15) for c in cajas_sug], lag_ratio=0.2),
            run_time=0.9
        )
        self.play(Indicate(cajas_sug[0], color=NARANJA_TERRACOTA, scale_factor=1.1), run_time=0.8)

        palabra_elegida = Text(
            "cansado", font=FUENTE, font_size=22,
            color=NARANJA_TERRACOTA, weight=BOLD
        ).next_to(tokens_burbuja, RIGHT, buff=0.18)\
            .align_to(tokens_burbuja[1], DOWN)
        origen_palabra = cajas_sug[0][1].copy()

        self.play(
            ReplacementTransform(origen_palabra, palabra_elegida),
            cursor.animate.next_to(palabra_elegida, RIGHT, buff=0.06),
            cajas_sug[1].animate.set_opacity(0.35),
            cajas_sug[2].animate.set_opacity(0.35),
            run_time=0.9
        )

        self._siguiente()

        self.play(
            FadeOut(frase_1), FadeOut(frase_2), FadeOut(frase_3),
            FadeOut(caja_todo), FadeOut(label_movil),
            FadeOut(burbuja), FadeOut(tokens_burbuja), FadeOut(cursor),
            FadeOut(cajas_sug), FadeOut(palabra_elegida),
            FadeOut(titulo), FadeOut(linea),
            run_time=0.8
        )

        encabezado_paper = Text(
            '"Attention Is All You Need" (Vaswani et al., 2017)',
            font=FUENTE, font_size=21, slant=ITALIC, color=TINTA_NEGRA,
            t2c={"2017": NARANJA_TERRACOTA}
        ).to_edge(UP, buff=0.35)

        ALTO_BLOQUE, SEP_BLOQUE, BASE_Y = 0.44, 0.22, -2.55
        ENC_X, DEC_X = -2.9, 2.9

        def caja_arq(texto, color_fondo, ancho=2.75, alto=ALTO_BLOQUE, fs=15):
            rect = RoundedRectangle(
                corner_radius=0.08, width=ancho, height=alto,
                fill_color=color_fondo, fill_opacity=1,
                stroke_color=MARRON_OSCURO, stroke_width=1.5
            )
            lbl = Text(texto, font=FUENTE, font_size=fs, color=TINTA_NEGRA)
            if lbl.width > ancho - 0.25:
                lbl.scale_to_fit_width(ancho - 0.25)
            lbl.move_to(rect)
            return VGroup(rect, lbl)

        def flecha_arq(desde, hasta, grosor=3):
            return Arrow(
                desde, hasta, buff=0.04, color=MARRON_OSCURO,
                stroke_width=grosor, max_tip_length_to_length_ratio=0.45
            )

        def torre_arq(definiciones, x_centro):
            bloques = VGroup(*[caja_arq(t, c) for t, c in definiciones])\
                .arrange(UP, buff=SEP_BLOQUE)
            marco = RoundedRectangle(
                corner_radius=0.15,
                width=bloques.width + 0.35, height=bloques.height + 0.28,
                fill_color=CREMA_CALIDA, fill_opacity=0.5,
                stroke_color=MARRON_OSCURO, stroke_width=1.2
            ).move_to(bloques)
            VGroup(marco, bloques).move_to([x_centro, BASE_Y + marco.height / 2, 0])
            flechas = VGroup(*[
                flecha_arq(bloques[i].get_top(), bloques[i + 1].get_bottom(), grosor=2.5)
                for i in range(len(bloques) - 1)
            ])
            emb = caja_arq("Embedding + posición", SALMON_CLARO)\
                .next_to(marco, DOWN, buff=0.3)
            flecha_emb = flecha_arq(emb.get_top(), marco.get_bottom())
            return bloques, flechas, marco, emb, flecha_emb

        enc_bloques, enc_flechas, marco_enc, emb_enc, flecha_enc = torre_arq([
            ("Multi-Head Attention", SALMON_ATENCION),
            ("Add & Norm", AMARILLO_PALIDO),
            ("Feed Forward", CELESTE_PALIDO),
            ("Add & Norm", AMARILLO_PALIDO),
        ], ENC_X)
        nx_enc  = Text("N ×", font=FUENTE, font_size=20, weight=BOLD,
                       color=MARRON_OSCURO).next_to(marco_enc, LEFT, buff=0.3)
        lbl_enc = Text("Encoder", font=FUENTE, font_size=19, weight=BOLD,
                       color=MARRON_OSCURO).next_to(emb_enc, DOWN, buff=0.2)

        dec_bloques, dec_flechas, marco_dec, emb_dec, flecha_dec1 = torre_arq([
            ("Masked Multi-Head Attention", SALMON_ATENCION),
            ("Add & Norm", AMARILLO_PALIDO),
            ("Multi-Head Attention", SALMON_ATENCION),
            ("Add & Norm", AMARILLO_PALIDO),
            ("Feed Forward", CELESTE_PALIDO),
            ("Add & Norm", AMARILLO_PALIDO),
        ], DEC_X)
        nx_dec  = Text("N ×", font=FUENTE, font_size=20, weight=BOLD,
                       color=MARRON_OSCURO).next_to(marco_dec, RIGHT, buff=0.3)
        lbl_dec = Text("Decoder", font=FUENTE, font_size=19, weight=BOLD,
                       color=MARRON_OSCURO).next_to(emb_dec, DOWN, buff=0.2)

        linear = caja_arq("Linear", LAVANDA, ancho=1.7).next_to(marco_dec, UP, buff=0.3)
        flecha_linear = flecha_arq(marco_dec.get_top(), linear.get_bottom())
        softmax = caja_arq("Softmax", MENTA_PALIDA, ancho=1.7).next_to(linear, UP, buff=0.26)
        flecha_softmax = flecha_arq(linear.get_top(), softmax.get_bottom(), grosor=2.5)
        probs = Text("Probabilidades", font=FUENTE, font_size=16,
                     color=MARRON_OSCURO).next_to(softmax, UP, buff=0.15)

        y_cross = dec_bloques[2].get_center()[1]
        flecha_cross = flecha_arq(
            [marco_enc.get_right()[0], y_cross, 0],
            [marco_dec.get_left()[0],  y_cross, 0],
        )

        self.play(FadeIn(encabezado_paper, shift=DOWN * 0.2))
        self.play(
            FadeIn(lbl_enc), FadeIn(emb_enc, shift=UP * 0.15),
            GrowArrow(flecha_enc), FadeIn(marco_enc),
            run_time=0.7
        )
        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.15) for b in enc_bloques], lag_ratio=0.15),
            LaggedStart(*[GrowArrow(f) for f in enc_flechas], lag_ratio=0.15),
            FadeIn(nx_enc),
            run_time=0.9
        )
        self.play(
            FadeIn(lbl_dec), FadeIn(emb_dec, shift=UP * 0.15),
            GrowArrow(flecha_dec1), FadeIn(marco_dec),
            run_time=0.7
        )
        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.15) for b in dec_bloques], lag_ratio=0.12),
            LaggedStart(*[GrowArrow(f) for f in dec_flechas], lag_ratio=0.12),
            FadeIn(nx_dec),
            run_time=1.0
        )
        self.play(
            GrowArrow(flecha_linear), FadeIn(linear, shift=UP * 0.1),
            GrowArrow(flecha_softmax), FadeIn(softmax, shift=UP * 0.1),
            FadeIn(probs),
            run_time=0.7
        )
        self.play(GrowArrow(flecha_cross), run_time=0.6)

        self._siguiente()

        encabezado_gpt = Text(
            "GPT-2 (OpenAI, 2019)", font=FUENTE, font_size=23, weight=BOLD,
            color=TINTA_NEGRA, t2c={"2019": NARANJA_TERRACOTA}
        ).move_to(encabezado_paper)

        torre_encoder = VGroup(lbl_enc, emb_enc, flecha_enc, marco_enc,
                               enc_bloques, enc_flechas, nx_enc)
        torre_decoder = VGroup(lbl_dec, emb_dec, flecha_dec1, marco_dec, dec_bloques,
                               dec_flechas, nx_dec, flecha_linear, linear,
                               flecha_softmax, softmax, probs)

        self.play(
            torre_encoder.animate.set_color(ROJO_CONTRA),
            flecha_cross.animate.set_color(ROJO_CONTRA),
            run_time=0.6
        )
        self.play(
            FadeOut(torre_encoder, shift=LEFT * 0.6),
            FadeOut(flecha_cross),
            ReplacementTransform(encabezado_paper, encabezado_gpt),
            torre_decoder.animate.shift(LEFT * DEC_X),
            run_time=1.1
        )

        par_cross = VGroup(dec_bloques[2], dec_bloques[3],
                           dec_flechas[1], dec_flechas[2])
        self.play(par_cross.animate.set_color(ROJO_CONTRA), run_time=0.5)

        delta        = dec_bloques[4].get_center()[1] - dec_bloques[2].get_center()[1]
        alto_nuevo   = marco_dec.height - delta
        centro_nuevo = marco_dec.get_bottom()[1] + alto_nuevo / 2

        nx_48 = Text("48 ×", font=FUENTE, font_size=20, weight=BOLD,
                     color=NARANJA_TERRACOTA).move_to(nx_dec).shift(DOWN * delta / 2)

        self.play(
            FadeOut(par_cross, shift=RIGHT * 0.5),
            VGroup(dec_bloques[4], dec_bloques[5],
                   dec_flechas[3], dec_flechas[4]).animate.shift(DOWN * delta),
            marco_dec.animate.stretch_to_fit_height(alto_nuevo).move_to([0, centro_nuevo, 0]),
            VGroup(flecha_linear, linear, flecha_softmax, softmax, probs).animate.shift(DOWN * delta),
            ReplacementTransform(nx_dec, nx_48),
            run_time=1.1
        )

        self.play(
            torre_decoder.animate.shift(UP * delta / 2),
            nx_48.animate.shift(UP * delta / 2),
            run_time=0.5
        )

        self._siguiente()

        encabezado_hoy = Text(
            "Y hoy: toda una familia de arquitecturas",
            font=FUENTE, font_size=23, weight=BOLD, color=TINTA_NEGRA,
            t2c={"familia": NARANJA_TERRACOTA},
        ).move_to(encabezado_gpt)

        imagen_arqs = ImageMobject(
            os.path.join("assets", "arquitecturas.png")
        ).scale_to_fit_width(10.2).move_to(DOWN * 0.25)
        marco_arqs = SurroundingRectangle(
            imagen_arqs, buff=0.08,
            color=MARRON_OSCURO, stroke_width=2, corner_radius=0.05,
        )

        self.play(
            ReplacementTransform(encabezado_gpt, encabezado_hoy),
            FadeOut(torre_decoder), FadeOut(nx_48),
            run_time=0.8,
        )
        self.play(FadeIn(imagen_arqs, scale=1.03), FadeIn(marco_arqs), run_time=0.9)

        resalte_gpt2 = RoundedRectangle(
            corner_radius=0.08,
            width=imagen_arqs.width / 4 - 0.1,
            height=imagen_arqs.height / 2 - 0.1,
            stroke_color=NARANJA_TERRACOTA, stroke_width=3.5, fill_opacity=0,
        ).move_to(
            imagen_arqs.get_corner(UL)
            + RIGHT * imagen_arqs.width / 8
            + DOWN * imagen_arqs.height / 4
        )
        self.play(Create(resalte_gpt2), run_time=0.7)

        self._siguiente()
        self.limpiar_pantalla()

