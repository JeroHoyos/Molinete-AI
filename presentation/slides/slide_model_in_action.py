import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import os
from colores import *
from objetos import *


class SlideModelInAction:
    def slide_model_in_action(self):

        titulo, linea = self.crear_titulo(
            "Demostración en Vivo",
            palabra_clave="Vivo",
            color_clave=NARANJA_TERRACOTA,
        )
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)

        WIN_W, WIN_H = 12.0, 5.8
        HDR_H, SB_W  = 0.52, 3.1
        SB_H = WIN_H - HDR_H

        app_win = RoundedRectangle(
            corner_radius=0.22, width=WIN_W, height=WIN_H,
            fill_color=FONDO_CAJA, fill_opacity=1,
            stroke_color=MARRON_OSCURO, stroke_width=2)
        app_hdr = Rectangle(
            width=WIN_W, height=HDR_H,
            fill_color=MARRON_OSCURO, fill_opacity=1, stroke_width=0,
        ).align_to(app_win, UP)

        dot_r = Circle(radius=0.08, fill_color=ROJO_MAC,     fill_opacity=1, stroke_width=0)
        dot_a = Circle(radius=0.08, fill_color=AMARILLO_MAC, fill_opacity=1, stroke_width=0)
        dot_v = Circle(radius=0.08, fill_color=VERDE_MAC,    fill_opacity=1, stroke_width=0)
        mac_dots = VGroup(dot_r, dot_a, dot_v).arrange(RIGHT, buff=0.13)
        mac_dots.next_to(app_hdr.get_left(), RIGHT, buff=0.28)
        hdr_lbl = Text("Molinete Chat", font_size=14, color=FONDO_CAJA, weight=BOLD).move_to(app_hdr)

        sb_bg = Rectangle(
            width=SB_W, height=SB_H,
            fill_color=CAJA_INFERIOR, fill_opacity=1, stroke_width=0)
        sb_bg.move_to(app_win.get_corner(UL) + RIGHT*(SB_W/2) + DOWN*(HDR_H + SB_H/2))

        sb_divider = Line(sb_bg.get_corner(UR), sb_bg.get_corner(DR),
                          color=PAPEL_TAN, stroke_width=1.5)

        sb_title = Text("Modelos disponibles", font=FUENTE, font_size=13,
                        color=MARRON_OSCURO, weight=BOLD)
        sb_title.next_to(sb_bg.get_top(), DOWN, buff=0.28)

        MODELOS = [
            ("ChatGPT",     "#10A37F", False),
            ("Claude",      "#CC785C", False),
            ("Gemini",      "#4285F4", False),
            ("Molinete AI", NARANJA_TERRACOTA, True),
        ]
        model_cards = VGroup()
        for name, col, active in MODELOS:
            card_bg = RoundedRectangle(
                corner_radius=0.12, width=SB_W - 0.28, height=0.56,
                fill_color=NARANJA_TERRACOTA if active else FONDO_CAJA,
                fill_opacity=0.18 if active else 0.55,
                stroke_color=col,
                stroke_width=2.5 if active else 0.8)
            dot_m = Circle(radius=0.09, fill_color=col, fill_opacity=1, stroke_width=0)
            lbl_m = Text(name, font=FUENTE, font_size=14,
                         color=MARRON_OSCURO if active else TINTA_NEGRA,
                         weight=BOLD if active else "NORMAL")
            VGroup(dot_m, lbl_m).arrange(RIGHT, buff=0.15).move_to(card_bg)
            model_cards.add(VGroup(card_bg, dot_m, lbl_m))

        model_cards.arrange(DOWN, buff=0.13)
        model_cards.next_to(sb_title, DOWN, buff=0.2)
        model_cards.align_to(sb_bg, LEFT).shift(RIGHT*0.15)

        chat_sub_sep = Line(
            sb_bg.get_corner(UR),
            app_win.get_corner(UR) + DOWN*HDR_H,
            color=PAPEL_TAN, stroke_width=1.5)

        chat_lbl = Text("Chat con Molinete AI", font=FUENTE, font_size=15,
                        color=PAPEL_TAN, weight=BOLD)
        chat_lbl.next_to(chat_sub_sep, DOWN, buff=0.22)
        chat_lbl.set_x((sb_bg.get_right()[0] + app_win.get_right()[0]) / 2)

        INPUT_W = WIN_W - SB_W - 1.0
        input_rect = RoundedRectangle(
            corner_radius=0.18, width=INPUT_W, height=0.54,
            fill_color=BLANCO, fill_opacity=0.92,
            stroke_color=PAPEL_TAN, stroke_width=1.5)
        input_rect.move_to(app_win.get_corner(DR) + LEFT*(INPUT_W/2 + 0.55) + UP*0.46)
        hint_txt = Text("Escribe un mensaje…", font=FUENTE, font_size=12, color=PAPEL_TAN)
        hint_txt.move_to(input_rect).shift(LEFT*1.5)
        send_circle = Circle(radius=0.20, fill_color=NARANJA_TERRACOTA,
                             fill_opacity=1, stroke_width=0)
        send_circle.next_to(input_rect, RIGHT, buff=0.1)
        send_arrow = Text("→", font_size=14, color=BLANCO).move_to(send_circle)

        app_ui = VGroup(
            app_win, app_hdr, mac_dots, hdr_lbl,
            sb_bg, sb_divider, sb_title, model_cards,
            chat_sub_sep, chat_lbl,
            input_rect, hint_txt, send_circle, send_arrow,
        )
        app_ui.next_to(linea, DOWN, buff=0.28)

        self.play(FadeIn(VGroup(app_win, app_hdr, mac_dots, hdr_lbl), shift=UP*0.3), run_time=0.8)
        self.play(FadeIn(VGroup(sb_bg, sb_divider, sb_title)), run_time=0.4)
        for card in model_cards:
            self.play(FadeIn(card, shift=RIGHT*0.1), run_time=0.2)
        self._siguiente()

        self.play(Indicate(model_cards[-1][0], scale_factor=1.09, color=NARANJA_TERRACOTA), run_time=0.7)
        self.play(
            FadeIn(VGroup(chat_sub_sep, chat_lbl), shift=DOWN*0.1),
            FadeIn(VGroup(input_rect, hint_txt, send_circle, send_arrow), shift=UP*0.1),
            run_time=0.6,
        )
        self._siguiente()

        sb_right_x  = sb_bg.get_right()[0]
        win_right_x = app_win.get_right()[0]
        sep_y       = chat_sub_sep.get_start()[1]

        burbuja_u = self._crear_burbuja_transformer(
            "En un lugar de la Mancha,", es_usuario=True)
        burbuja_ia = self._crear_burbuja_transformer(
            "de cuyo nombre no quiero\nacordarme, no ha mucho...", es_usuario=False)
        burbuja_u.scale(0.80)
        burbuja_ia.scale(0.80)

        burbuja_u.move_to([
            win_right_x - burbuja_u.width/2 - 0.30,
            sep_y - burbuja_u.height/2 - 0.55,
            0,
        ])
        burbuja_ia.move_to([
            sb_right_x + burbuja_ia.width/2 + 0.30,
            burbuja_u.get_bottom()[1] - burbuja_ia.height/2 - 0.35,
            0,
        ])

        self.play(FadeIn(burbuja_u, shift=UP*0.15), run_time=0.5)
        self.wait(0.4)
        self.play(FadeIn(burbuja_ia, shift=UP*0.15), run_time=0.5)
        self.wait(0.8)
        self._siguiente()

        transicion_lbl = Text("Cambiando al Demo", font_size=24,
                              color=NARANJA_TERRACOTA, weight=BOLD)
        caja_transicion = SurroundingRectangle(
            transicion_lbl, color=MARRON_OSCURO, fill_color=FONDO_CAJA,
            fill_opacity=1, buff=0.4)
        grupo_transicion = VGroup(caja_transicion, transicion_lbl).move_to(app_win.get_center())

        fade_group = VGroup(
            sb_bg, sb_divider, sb_title, model_cards,
            chat_sub_sep, chat_lbl,
            input_rect, hint_txt, send_circle, send_arrow,
            burbuja_u, burbuja_ia,
        )
        self.play(fade_group.animate.set_opacity(0.1), FadeIn(grupo_transicion, scale=0.8))
        self._siguiente()
        self.limpiar_pantalla()


