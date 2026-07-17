import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import os
from colores import *
from objetos import *


class SlideBytePairEncoding:
    def slide_byte_pair_encoding(self):

        llanuras_fondo = crear_llanuras_manchegas()
        adornos = self._crear_adornos_esquinas(escala=0.55, buff=0.5)

        molino = adornos[0]

        if len(molino) > 1:
            aspas = molino[-1]
            centro_aspas = aspas.get_center()
            aspas.add_updater(lambda m, dt: m.rotate(-dt * 0.3, about_point=centro_aspas))
        else:
            molino.add_updater(lambda m, dt: m.rotate(-dt * 0.3))

        titulo, linea = self.crear_titulo(
            "Byte Pair Encoding (BPE)",
            palabra_clave="BPE",
            color_clave=NARANJA_TERRACOTA
        )
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo, adornos=adornos)

        subtitulo = Text(
            "Fusiona los pares de caracteres más frecuentes",
            font=FUENTE, font_size=21, color=MARRON_OSCURO, slant=ITALIC
        ).next_to(linea, DOWN, buff=0.25)
        self.play(FadeIn(subtitulo, shift=DOWN * 0.2))

        panel_rect = RoundedRectangle(
            corner_radius=0.18, width=2.4, height=4.6,
            fill_color=FONDO_CAJA, fill_opacity=0.97,
            stroke_color=NARANJA_TERRACOTA, stroke_width=2.5
        ).to_edge(RIGHT, buff=0.35).shift(DOWN * 0.2)

        panel_titulo = Text("Vocabulario", font=FUENTE, font_size=16,
                            color=NARANJA_TERRACOTA, weight=BOLD)
        panel_titulo.next_to(panel_rect.get_top(), DOWN, buff=0.2)

        separador_panel = Line(
            panel_rect.get_left() + RIGHT * 0.2,
            panel_rect.get_right() + LEFT * 0.2,
            color=NARANJA_TERRACOTA, stroke_width=1.5
        ).next_to(panel_titulo, DOWN, buff=0.15)

        self.play(
            FadeIn(panel_rect, shift=LEFT * 0.3),
            Write(panel_titulo),
            Create(separador_panel)
        )

        iter_label_bg = RoundedRectangle(
            corner_radius=0.12, width=2.6, height=0.55,
            fill_color=MARRON_OSCURO, fill_opacity=0.9,
            stroke_width=0
        ).to_edge(LEFT, buff=0.7).shift(UP * 2.3)

        iter_label = Text("Iteración  1 / 4", font=FUENTE, font_size=20,
                          color=PAPEL_CREMA, weight=BOLD)
        iter_label.move_to(iter_label_bg)

        self.play(FadeIn(iter_label_bg), Write(iter_label))

        fusion_bg  = VMobject()
        fusion_txt = VMobject()
        self.add(fusion_bg, fusion_txt)

        def calc_ancho(texto):
            return max(0.38, 0.28 + len(texto) * 0.18)

        token_colors = {
            "_":    PERGAMINO_CLARO,
            "tr":   SALMON_CLARO,
            "es":   MENTA_PALIDA,
            "tri":  LAVANDA,
            "trig": AMARILLO_PALIDO,
        }

        frases_label = ["Frase 1:", "Frase 2:", "Frase 3:"]
        label_colors = [LADRILLO_VIVO, VERDE_OLIVA, OCRE_CERVANTINO]

        def crear_grid(estado_actual):
            filas = VGroup()
            for idx_row, row in enumerate(estado_actual):
                etiqueta = Text(
                    frases_label[idx_row], font=FUENTE, font_size=17,
                    color=label_colors[idx_row], weight=BOLD
                )
                tokens_row = VGroup(*[
                    self.crear_bloque(
                        s,
                        color_fondo=token_colors.get(s, CREMA_CALIDA),
                        ancho=calc_ancho(s),
                        alto=0.52
                    )
                    for s in row
                ]).arrange(RIGHT, buff=0.06)
                fila = VGroup(etiqueta, tokens_row).arrange(RIGHT, buff=0.22)
                filas.add(fila)

            filas.arrange(DOWN, buff=0.32)
            filas.move_to(LEFT * 0.8 + DOWN * 0.4)
            return filas

        def get_new_state_and_indices(estado, char1, char2, new_char):
            new_estado = []
            fusions_indices = []
            for r_idx, row in enumerate(estado):
                new_row = []
                c_idx = 0
                while c_idx < len(row):
                    if (c_idx < len(row) - 1
                            and row[c_idx] == char1
                            and row[c_idx + 1] == char2):
                        fusions_indices.append((r_idx, c_idx, c_idx + 1))
                        new_row.append(new_char)
                        c_idx += 2
                    else:
                        new_row.append(row[c_idx])
                        c_idx += 1
                new_estado.append(new_row)
            return new_estado, fusions_indices

        estado_actual = [
            ["t","r","e","s","_","t","r","i","s","t","e","s"],
            ["t","i","g","r","e","s","_","t","r","a","g","a","n"],
            ["t","r","i","g","o","_","t","r","i","g","a","l"],
        ]

        grid_actual = crear_grid(estado_actual)
        self.play(FadeIn(grid_actual, shift=UP * 0.2), run_time=1.2)
        self.wait(0.3)

        vocab_entries = VGroup()
        vocab_anchor = separador_panel.get_bottom() + DOWN * 0.2

        def agregar_vocab_entry(token, color_tok, color_bg):
            dot = Dot(radius=0.06, color=color_tok)
            tok_txt = Text(token, font=FUENTE, font_size=15,
                           color=TINTA_NEGRA, weight=BOLD)
            tok_rect = RoundedRectangle(
                corner_radius=0.08,
                width=max(0.6, tok_txt.width + 0.25), height=0.32,
                fill_color=color_bg, fill_opacity=1,
                stroke_color=color_tok, stroke_width=1.8
            )
            tok_txt.move_to(tok_rect)
            entry = VGroup(dot, VGroup(tok_rect, tok_txt)).arrange(RIGHT, buff=0.1)

            if len(vocab_entries) == 0:
                entry.next_to(vocab_anchor, DOWN, buff=0.1).align_to(panel_rect, LEFT).shift(RIGHT * 0.25)
            else:
                entry.next_to(vocab_entries[-1], DOWN, buff=0.15).align_to(vocab_entries[-1], LEFT)
            vocab_entries.add(entry)
            return entry

        pasos = [
            ("t",   "r",   "tr",   LADRILLO_VIVO,   SALMON_CLARO,   "1 / 4"),
            ("e",   "s",   "es",   VERDE_OLIVA,     MENTA_PALIDA,   "2 / 4"),
            ("tr",  "i",   "tri",  AZUL_NOCHE,      LAVANDA,        "3 / 4"),
            ("tri", "g",   "trig", OCRE_CERVANTINO, AMARILLO_PALIDO,"4 / 4"),
        ]

        for c1, c2, nuevo, color_resalte, color_bg, iter_str in pasos:

            nueva_iter = Text(f"Iteración  {iter_str}", font=FUENTE, font_size=20,
                              color=PAPEL_CREMA, weight=BOLD).move_to(iter_label_bg)
            self.play(ReplacementTransform(iter_label, nueva_iter), run_time=0.4)
            iter_label = nueva_iter

            # La flecha se dibuja con Manim: la fuente no tiene el glifo "\u2192"
            parte_izq = Text(
                f'Fusionar: "{c1}" + "{c2}"',
                font=FUENTE, font_size=19, color=NARANJA_TERRACOTA, weight=BOLD,
            )
            flecha_fusion = Arrow(
                ORIGIN, RIGHT * 0.55, color=color_resalte, buff=0,
                stroke_width=4, max_tip_length_to_length_ratio=0.4,
            )
            parte_der = Text(
                f'"{nuevo}"', font=FUENTE, font_size=19,
                color=color_resalte, weight=BOLD,
            )
            nueva_fusion = VGroup(parte_izq, flecha_fusion, parte_der)
            nueva_fusion.arrange(RIGHT, buff=0.2)

            nueva_bg = RoundedRectangle(
                corner_radius=0.14,
                width=nueva_fusion.width + 0.5, height=0.62,
                fill_color=MARRON_OSCURO, fill_opacity=0.08,
                stroke_color=color_resalte, stroke_width=2.5,
            ).next_to(iter_label_bg, DOWN, buff=0.25).align_to(iter_label_bg, LEFT)
            nueva_fusion.move_to(nueva_bg)

            self.play(
                ReplacementTransform(fusion_bg, nueva_bg),
                ReplacementTransform(fusion_txt, nueva_fusion),
                run_time=0.5
            )
            fusion_bg, fusion_txt = nueva_bg, nueva_fusion

            new_estado, fusions = get_new_state_and_indices(estado_actual, c1, c2, nuevo)
            new_grid = crear_grid(new_estado)

            pulsos = VGroup()
            indicates = []
            for r, ci1, ci2 in fusions:
                b1 = grid_actual[r][1][ci1]
                b2 = grid_actual[r][1][ci2]
                pulso = SurroundingRectangle(
                    VGroup(b1, b2), color=color_resalte,
                    buff=0.06, stroke_width=3, corner_radius=0.07
                )
                pulsos.add(pulso)
                indicates += [
                    Indicate(b1, color=color_resalte, scale_factor=1.08),
                    Indicate(b2, color=color_resalte, scale_factor=1.08),
                ]

            self.play(
                FadeIn(pulsos),
                LaggedStart(*indicates, lag_ratio=0.04),
                run_time=1.0
            )
            self.play(
                ReplacementTransform(grid_actual, new_grid),
                FadeOut(pulsos),
                run_time=1.1
            )

            entry = agregar_vocab_entry(nuevo, color_resalte, color_bg)
            self.play(FadeIn(entry, scale=0.8), run_time=0.5)

            estado_actual = new_estado
            grid_actual = new_grid
            self.wait(0.25)

        conclusion = Text(
            "Vocabulario aprendido",
            font=FUENTE, font_size=24, color=ORO_VIEJO, weight=BOLD
        ).to_edge(DOWN, buff=0.55)

        self.play(Write(conclusion), run_time=0.8)

        for fila in grid_actual:
            for bloque in fila[1]:
                if bloque[1].text == "trig":
                    self.play(
                        Flash(bloque, color=ORO_VIEJO, line_length=0.35, num_lines=8),
                        Indicate(bloque, color=ORO_VIEJO, scale_factor=1.18),
                        run_time=0.6
                    )

        self.wait(0.8)
        self._siguiente()

        self.limpiar_pantalla()


