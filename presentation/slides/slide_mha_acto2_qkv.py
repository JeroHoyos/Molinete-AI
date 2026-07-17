import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import numpy as np
import os
from colores import *
from objetos import *


COLOR_HIDALGO = NARANJA_TERRACOTA
COLOR_GIGANTE = VERDE_OLIVA
COLOR_ESTE    = AZUL_NOCHE


def _celdas_mat(filas, cols, color, ops=None, w=0.34, h=0.3, buff=0.06):
    if ops is None:
        ops = [0.82] * (filas * cols)
    grid = VGroup()
    idx = 0
    for r in range(filas):
        fila = VGroup()
        for c in range(cols):
            fila.add(RoundedRectangle(
                corner_radius=0.04, width=w, height=h,
                fill_color=color, fill_opacity=ops[idx],
                stroke_color=MARRON_OSCURO, stroke_width=1.1))
            idx += 1
        fila.arrange(RIGHT, buff=buff)
        grid.add(fila)
    grid.arrange(DOWN, buff=buff)
    return grid


def _corchetes(obj, color=MARRON_OSCURO, ext=0.13, buff=0.1):
    izq = obj.get_left()[0] - buff
    der = obj.get_right()[0] + buff
    top = obj.get_top()[1] + 0.07
    bot = obj.get_bottom()[1] - 0.07

    def br(x, dirx):
        m = VMobject(stroke_color=color, stroke_width=3)
        m.set_points_as_corners([
            [x + dirx * ext, top, 0], [x, top, 0],
            [x, bot, 0], [x + dirx * ext, bot, 0]])
        return m
    return VGroup(br(izq, 1), br(der, -1))


_UNIDAD = 0.72
_UMAX = 6
_VMAX = 4


def _espacio(origen, unidad=_UNIDAD, u_max=_UMAX, v_max=_VMAX):
    """Malla completa cuyos ejes son parte de la propia malla (se deforman juntos)."""
    lineas = VGroup()
    for u in range(1, u_max + 1):
        lineas.add(Line(origen + RIGHT * u * unidad,
                        origen + RIGHT * u * unidad + UP * v_max * unidad,
                        stroke_color=PAPEL_TAN, stroke_width=1.2, stroke_opacity=0.5))
    for v in range(1, v_max + 1):
        lineas.add(Line(origen + UP * v * unidad,
                        origen + UP * v * unidad + RIGHT * u_max * unidad,
                        stroke_color=PAPEL_TAN, stroke_width=1.2, stroke_opacity=0.5))
    ax = Arrow(origen + LEFT * 0.15, origen + RIGHT * (u_max * unidad + 0.35), buff=0,
               color=MARRON_OSCURO, stroke_width=3, max_tip_length_to_length_ratio=0.028)
    ay = Arrow(origen + DOWN * 0.15, origen + UP * (v_max * unidad + 0.35), buff=0,
               color=MARRON_OSCURO, stroke_width=3, max_tip_length_to_length_ratio=0.04)
    return VGroup(lineas, ax, ay)


def _P(origen, u, v, unidad=_UNIDAD):
    return origen + RIGHT * u * unidad + UP * v * unidad


def _PM(origen, u, v, M, unidad=_UNIDAD):
    """Posicion (u,v) sobre la malla ya transformada por M: cae en la interseccion."""
    off = np.array([M[0, 0] * u + M[0, 1] * v, M[1, 0] * u + M[1, 1] * v, 0]) * unidad
    return origen + off


def _punto(pos, color, nombre):
    halo = Dot(pos, radius=0.17, color=color).set_opacity(0.22)
    dot = Dot(pos, radius=0.1, color=color).set_stroke(MARRON_OSCURO, width=1.5)
    lbl = Text(nombre, font=FUENTE, font_size=16, color=color, weight=BOLD)\
        .next_to(dot, UP, buff=0.13)
    return VGroup(halo, dot, lbl)


def _chip_pal(word, color, fs=15, w=0.82, h=0.38):
    caja = RoundedRectangle(corner_radius=0.08, width=w, height=h,
                            fill_color=FONDO_CAJA, fill_opacity=1,
                            stroke_color=color, stroke_width=2.2)
    lbl = Text(word, font=FUENTE, font_size=fs, color=color, weight=BOLD).move_to(caja)
    return VGroup(caja, lbl)


class SlideMhaActo2Qkv:
    def slide_mha_acto2_qkv(self):
        titulo, linea = self.crear_titulo(
            "Query, Key y Value", palabra_clave="Query",
            color_clave=NARANJA_TERRACOTA)
        llanuras_fondo = crear_llanuras_manchegas()
        adornos = self._crear_adornos_esquinas()
        adornos[1].add_updater(lambda m, dt: m.rotate(dt * 0.15))
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo, adornos=adornos)

        # ── Acto A: la frase se genera y se corta antes de "molino" ─────
        palabras = ["El", "hidalgo", "vio", "al", "gigante,",
                    "pero", "este", "era", "un", "..."]
        IDX_H, IDX_G, IDX_E = 1, 4, 6
        col_de = {IDX_H: COLOR_HIDALGO, IDX_G: COLOR_GIGANTE, IDX_E: COLOR_ESTE}

        frase = VGroup(*[
            Text(p, font=FUENTE, font_size=28, color=TINTA_NEGRA) for p in palabras
        ]).arrange(RIGHT, buff=0.24).move_to(UP * 1.6)

        for w in frase[:-1]:
            self.play(FadeIn(w, shift=UP * 0.12), run_time=0.22)
        self.play(FadeIn(frase[-1], shift=UP * 0.12), run_time=0.3)
        self.play(Indicate(frase[-1], color=NARANJA_TERRACOTA, scale_factor=1.5))
        self._siguiente()

        # ── Acto B: las palabras se transforman en la matriz X ──────────
        n_col = len(palabras) - 1
        columnas = VGroup(*[
            _celdas_mat(3, 1, col_de.get(i, CREMA_CALIDA),
                        ops=[0.75, 0.4, 0.58], w=0.36, h=0.3)
            for i in range(n_col)
        ]).arrange(RIGHT, buff=0.08).move_to(DOWN * 0.4)

        corch = _corchetes(columnas)
        lbl_X = MathTex("X", font_size=54, color=MARRON_OSCURO)\
            .next_to(corch, LEFT, buff=0.28)

        self.play(
            LaggedStart(*[
                ReplacementTransform(frase[i], columnas[i]) for i in range(n_col)
            ], lag_ratio=0.07),
            FadeOut(frase[-1], shift=DOWN * 0.2),
            run_time=1.6,
        )
        self.play(Create(corch), Write(lbl_X), run_time=0.6)
        self._siguiente()

        # ── Acto C: espacio de embeddings (centrado, sin marco) ─────────
        ORIGEN_C = np.array([-2.2, -1.9, 0])
        EMB = {"hidalgo": (1.8, 2.6), "gigante": (3.2, 2.4), "este": (2.5, 2.05)}
        COL = {"hidalgo": COLOR_HIDALGO, "gigante": COLOR_GIGANTE, "este": COLOR_ESTE}

        plano_c = _espacio(ORIGEN_C)
        pts = {n: _punto(_P(ORIGEN_C, *EMB[n]), COL[n], n) for n in EMB}

        self.play(
            FadeOut(VGroup(*[columnas[i] for i in range(n_col) if i not in col_de])),
            FadeOut(corch), FadeOut(lbl_X),
            LaggedStart(GrowArrow(plano_c[1]), GrowArrow(plano_c[2]),
                        Create(plano_c[0]), lag_ratio=0.15),
            run_time=1.0,
        )
        self.play(
            ReplacementTransform(columnas[IDX_H], pts["hidalgo"]),
            ReplacementTransform(columnas[IDX_G], pts["gigante"]),
            ReplacementTransform(columnas[IDX_E], pts["este"]),
            run_time=1.2,
        )
        lineas_duda = VGroup(*[
            DashedLine(pts["este"][1].get_center(), pts[n][1].get_center(),
                       dash_length=0.08, stroke_color=COLOR_ESTE,
                       stroke_width=2, stroke_opacity=0.7)
            for n in ("hidalgo", "gigante")
        ])
        signo = Text("?", font=FUENTE, font_size=30, color=COLOR_ESTE, weight=BOLD)\
            .next_to(pts["este"][1], DOWN, buff=0.18)
        self.play(Create(lineas_duda), FadeIn(signo, scale=0.5),
                  Indicate(pts["este"][1], color=COLOR_ESTE), run_time=0.8)
        self._siguiente()

        # ── Acto D: WQ·X = Q, WK·X = K, y el espacio se transforma ──────
        self.play(
            VGroup(plano_c, *pts.values()).animate.shift(LEFT * 2.6),
            FadeOut(lineas_duda), FadeOut(signo),
            run_time=0.8,
        )
        ORIGEN_D = ORIGEN_C + LEFT * 2.6

        def fila_mult(y, cW, cRes, tex_w, tex_res):
            wq = _celdas_mat(2, 2, cW, w=0.3, h=0.3)
            xm = _celdas_mat(2, 3, CREMA_CALIDA, w=0.3, h=0.3)
            res = _celdas_mat(2, 3, cRes, w=0.3, h=0.3)
            sig_x = MathTex(r"\times", font_size=30, color=MARRON_OSCURO)
            sig_e = MathTex("=", font_size=30, color=MARRON_OSCURO)
            fila = VGroup(wq, sig_x, xm, sig_e, res).arrange(RIGHT, buff=0.2)
            fila.move_to(RIGHT * 3.4 + UP * y)
            lw = MathTex(tex_w, font_size=26, color=cW).next_to(wq, LEFT, buff=0.18)
            lr = MathTex(tex_res, font_size=26, color=cRes).next_to(res, DOWN, buff=0.12)
            lx = MathTex("X", font_size=24, color=MARRON_OSCURO).next_to(xm, DOWN, buff=0.12)
            return fila, wq, xm, res, sig_x, sig_e, lw, lr, lx

        f1, wq1, xm1, q1, sx1, se1, lw1, lr1, lx1 = fila_mult(
            1.15, NARANJA_TERRACOTA, NARANJA_TERRACOTA, r"W_Q", "Q")
        f2, wk2, xm2, k2, sx2, se2, lw2, lr2, lx2 = fila_mult(
            -0.9, VERDE_OLIVA, VERDE_OLIVA, r"W_K", "K")

        # WQ * X = Q
        self.play(FadeIn(wq1, shift=RIGHT * 0.1), Write(lw1),
                  FadeIn(xm1), Write(lx1), Write(sx1), Write(se1), run_time=0.7)
        self.play(
            LaggedStart(*[c.animate.set_fill(NARANJA_TERRACOTA, 0.82)
                          for f in q1 for c in f], lag_ratio=0.08),
            Write(lr1),
            run_time=0.8,
        )
        self.play(Flash(q1.get_center(), color=NARANJA_TERRACOTA,
                        line_length=0.25, num_lines=9), run_time=0.4)
        # WK * X = K
        self.play(FadeIn(wk2, shift=RIGHT * 0.1), Write(lw2),
                  FadeIn(xm2), Write(lx2), Write(sx2), Write(se2), run_time=0.7)
        self.play(
            LaggedStart(*[c.animate.set_fill(VERDE_OLIVA, 0.82)
                          for f in k2 for c in f], lag_ratio=0.08),
            Write(lr2),
            run_time=0.8,
        )
        self.play(Flash(k2.get_center(), color=VERDE_OLIVA,
                        line_length=0.25, num_lines=9), run_time=0.4)
        self._siguiente()

        # ── Q · Kᵀ: Q y K se multiplican entre sí y nacen los scores ────
        self.play(FadeOut(VGroup(f1, lw1, lr1, lx1, f2, lw2, lr2, lx2)), run_time=0.5)

        Qb = _celdas_mat(2, 4, NARANJA_TERRACOTA, w=0.34, h=0.34)
        Kb = _celdas_mat(4, 2, VERDE_OLIVA, w=0.34, h=0.34)
        # scores 4x4: una casilla por par de palabras (similitud query·key)
        s_ops = [
            [0.90, 0.20, 0.25, 0.28],
            [0.24, 0.85, 0.30, 0.32],
            [0.28, 0.26, 0.90, 0.36],
            [0.30, 0.26, 0.82, 0.86],
        ]
        Sb = _celdas_mat(4, 4, MARRON_OSCURO,
                         ops=[s_ops[i][j] for i in range(4) for j in range(4)],
                         w=0.36, h=0.36)
        sxb = MathTex(r"\times", font_size=34, color=MARRON_OSCURO)
        sib = MathTex("=", font_size=34, color=MARRON_OSCURO)
        prod = VGroup(Qb, sxb, Kb, sib, Sb).arrange(RIGHT, buff=0.3)\
            .move_to(RIGHT * 2.5 + DOWN * 0.15)
        lQ = MathTex("Q", font_size=28, color=NARANJA_TERRACOTA).next_to(Qb, DOWN, buff=0.16)
        lK = MathTex(r"K^{\top}", font_size=28, color=VERDE_OLIVA).next_to(Kb, DOWN, buff=0.16)
        lS = MathTex(r"Q\,K^{\top}", font_size=26, color=MARRON_OSCURO).next_to(Sb, DOWN, buff=0.16)

        self.play(FadeIn(Qb, shift=RIGHT * 0.1), Write(lQ),
                  FadeIn(Kb, shift=LEFT * 0.1), Write(lK),
                  Write(sxb), Write(sib), run_time=0.8)
        # los scores aparecen casilla a casilla (producto punto de cada Q con cada K)
        self.play(LaggedStart(*[GrowFromCenter(c) for f in Sb for c in f],
                              lag_ratio=0.03), Write(lS), run_time=1.2)
        self.play(Flash(Sb.get_center(), color=NARANJA_TERRACOTA,
                        line_length=0.3, num_lines=12), run_time=0.4)
        self._siguiente()

        # ── los scores alteran y transforman el espacio vectorial ───────
        M_QK = np.array([[1.12, -0.42], [0.3, 1.02]])
        QK = {"hidalgo": (1.4, 3.2), "gigante": (4.7, 1.5), "este": (3.05, 2.35)}
        ORIGEN_E = np.array([-2.16, -2.0, 0])
        shift_e = ORIGEN_E - ORIGEN_D

        # centramos el espacio vectorial; los scores quedan como operador al lado
        scores_grp = VGroup(Sb, lS)
        self.play(
            FadeOut(VGroup(Qb, Kb, sxb, sib, lQ, lK)),
            VGroup(plano_c, *pts.values()).animate.shift(shift_e),
            scores_grp.animate.scale(0.75).move_to(RIGHT * 4.7 + UP * 0.7),
            run_time=0.9,
        )

        fantasma = plano_c.copy().set_stroke(opacity=0.15)
        self.add(fantasma)
        # los scores "caen" sobre el espacio y lo deforman
        s_fly = Sb.copy()
        self.play(
            s_fly.animate.scale(0.25).move_to(plano_c.get_center()).set_opacity(0.0),
            ApplyMatrix(M_QK, plano_c, about_point=ORIGEN_E),
            *[pts[n].animate.move_to(_PM(ORIGEN_E, *QK[n], M_QK)) for n in pts],
            Indicate(Sb, color=NARANJA_TERRACOTA, scale_factor=1.08),
            run_time=1.7,
        )
        self.remove(s_fly)
        self._siguiente()

        # ── Acto E: mapa de calor de afinidad (la diagonal es 100%) ─────
        self.play(
            FadeOut(VGroup(plano_c, fantasma, Sb, lS, *pts.values())),
            run_time=0.6,
        )

        PAL = ["hidalgo", "vio", "gigante", "este"]
        COL_PAL = {"hidalgo": COLOR_HIDALGO, "vio": MARRON_OSCURO,
                   "gigante": COLOR_GIGANTE, "este": COLOR_ESTE}
        # afinidad causal: la diagonal (cada palabra consigo misma) siempre 100%
        W = [
            [1.00, None, None, None],
            [0.30, 1.00, None, None],
            [0.22, 0.18, 1.00, None],
            [0.12, 0.10, 0.83, 1.00],
        ]

        CELDA = 1.02
        gx, gy = 0.55, -0.35
        X0 = gx - 1.5 * CELDA
        Y0 = gy + 1.5 * CELDA
        LADO = 4 * CELDA

        top_words = VGroup(*[
            Text(PAL[j], font=FUENTE, font_size=21, color=COL_PAL[PAL[j]], weight=BOLD)
            .move_to([X0 + j * CELDA, Y0 + CELDA / 2 + 0.34, 0]) for j in range(4)
        ])
        left_words = VGroup(*[
            Text(PAL[i], font=FUENTE, font_size=21, color=COL_PAL[PAL[i]], weight=BOLD)
            .next_to([gx - LADO / 2, Y0 - i * CELDA, 0], LEFT, buff=0.28) for i in range(4)
        ])
        frame = RoundedRectangle(corner_radius=0.14, width=LADO + 0.12, height=LADO + 0.12,
                                 fill_opacity=0, stroke_color=MARRON_OSCURO,
                                 stroke_width=2.2).move_to([gx, gy, 0])

        # heatmap: intensidad = afinidad; la diagonal (oro) siempre al 100%
        celdas = []
        masks = VGroup()
        for i in range(4):
            fila = []
            for j in range(4):
                centro = np.array([X0 + j * CELDA, Y0 - i * CELDA, 0])
                if j > i:
                    masks.add(RoundedRectangle(
                        corner_radius=0.1, width=CELDA * 0.5, height=CELDA * 0.5,
                        fill_color=ACERO, fill_opacity=0.08, stroke_width=0).move_to(centro))
                    fila.append(None)
                    continue
                val = W[i][j]
                diag = (i == j)
                op = 0.14 + val * 0.86
                sq = RoundedRectangle(
                    corner_radius=0.1, width=CELDA * 0.84, height=CELDA * 0.84,
                    fill_color=NARANJA_TERRACOTA, fill_opacity=op,
                    stroke_color=ORO_VIEJO if diag else MARRON_OSCURO,
                    stroke_width=3.2 if diag else 1.2).move_to(centro)
                num = Text(f"{int(round(val * 100))}%", font=FUENTE,
                           font_size=20 if diag else 17,
                           color=BLANCO if op > 0.5 else MARRON_OSCURO,
                           weight=BOLD if diag else NORMAL).move_to(centro + UP * 0.1)
                track = RoundedRectangle(corner_radius=0.03, width=CELDA * 0.52, height=0.09,
                                         fill_color=PAPEL_CREMA, fill_opacity=0.55,
                                         stroke_width=0).move_to(centro + DOWN * 0.2)
                fill = RoundedRectangle(corner_radius=0.03, width=max(0.04, CELDA * 0.52 * val),
                                        height=0.09, fill_color=MARRON_OSCURO,
                                        fill_opacity=0.85, stroke_width=0)
                fill.align_to(track, LEFT).match_y(track)
                fila.append(VGroup(sq, num, track, fill))
            celdas.append(fila)

        marco_este = RoundedRectangle(
            corner_radius=0.14, width=LADO + 0.2, height=CELDA + 0.08,
            fill_color=COLOR_ESTE, fill_opacity=0.07,
            stroke_color=COLOR_ESTE, stroke_width=3)\
            .move_to([gx, Y0 - 3 * CELDA, 0])

        tabla_e = VGroup(top_words, left_words, frame, masks, marco_este,
                         *[celdas[i][j] for i in range(4) for j in range(i + 1)])

        self.play(
            LaggedStart(*[FadeIn(w, shift=DOWN * 0.1) for w in top_words], lag_ratio=0.08),
            LaggedStart(*[FadeIn(w, shift=RIGHT * 0.1) for w in left_words], lag_ratio=0.08),
            Create(frame), FadeIn(masks),
            run_time=1.1,
        )
        # la diagonal aparece primero: cada palabra consigo misma = 100%
        self.play(*[GrowFromCenter(celdas[i][i]) for i in range(4)], run_time=0.7)
        # luego el resto del triángulo: afinidad con las palabras anteriores
        for i in range(1, 4):
            self.play(*[FadeIn(celdas[i][j], scale=0.8) for j in range(i)], run_time=0.4)

        self.play(Create(marco_este),
                  Indicate(celdas[3][2], color=COLOR_GIGANTE, scale_factor=1.2))
        self._siguiente()

        # ── Acto F: matriz de Value (palabra → E → W_V → v) ─────────────
        self.play(FadeOut(tabla_e), run_time=0.6)

        CW, RH = 1.18, 0.82
        J_HL = 3  # columna resaltada = query "este"

        def fila_val(i):
            c = COL_PAL[PAL[i]]
            chip = _chip_pal(PAL[i], c, fs=15, w=0.8, h=0.36)
            a2 = Arrow([-0.09, 0, 0], [0.09, 0, 0], buff=0, color=NARANJA_TERRACOTA,
                       stroke_width=2, max_tip_length_to_length_ratio=0.5)
            v = MathTex(rf"\vec v_{{{i + 1}}}", font_size=26, color=NARANJA_TERRACOTA)
            row = VGroup(chip, a2, v).arrange(RIGHT, buff=0.12)
            wv = MathTex(r"W_V", font_size=13, color=NARANJA_TERRACOTA).next_to(a2, UP, buff=0.03)
            grp = VGroup(row, wv)
            grp.next_to(np.array([-CW / 2, -i * RH, 0]), LEFT, buff=0.15)
            return grp

        def col_query(j):
            c = COL_PAL[PAL[j]]
            chip = _chip_pal(PAL[j], c, fs=14, w=0.76, h=0.34)
            a = Arrow([0, 0.09, 0], [0, -0.09, 0], buff=0, color=MARRON_OSCURO,
                      stroke_width=2, max_tip_length_to_length_ratio=0.5)
            stack = VGroup(chip, a).arrange(DOWN, buff=0.08)
            stack.next_to(np.array([j * CW, RH / 2, 0]), UP, buff=0.14)
            return stack

        filas_v = VGroup(*[fila_val(i) for i in range(4)])
        cols_q = VGroup(*[col_query(j) for j in range(4)])

        gx_v = 1.5 * CW
        gy_v = -1.5 * RH
        Wg = 4 * CW
        Hg = 4 * RH
        frame_v = RoundedRectangle(corner_radius=0.12, width=Wg, height=Hg,
                                   fill_opacity=0, stroke_color=MARRON_OSCURO,
                                   stroke_width=1.8).move_to([gx_v, gy_v, 0])
        lineas_v = VGroup(*[
            Line([(k - 0.5) * CW, gy_v + Hg / 2, 0], [(k - 0.5) * CW, gy_v - Hg / 2, 0],
                 stroke_color=PAPEL_TAN, stroke_width=0.9) for k in range(1, 4)
        ] + [
            Line([gx_v - Wg / 2, -(k - 0.5) * RH, 0], [gx_v + Wg / 2, -(k - 0.5) * RH, 0],
                 stroke_color=PAPEL_TAN, stroke_width=0.9) for k in range(1, 4)
        ])
        rejilla_v = VGroup(frame_v, lineas_v)

        # columna resaltada (query "este")
        col_hl = RoundedRectangle(corner_radius=0.1, width=CW - 0.05, height=Hg + 0.06,
                                  fill_color=COLOR_ESTE, fill_opacity=0.08,
                                  stroke_color=COLOR_ESTE, stroke_width=2.6)\
            .move_to([J_HL * CW, gy_v, 0])

        # pesos de atención de "este" sobre cada v_i (gigante domina)
        pesos_e = [0.05, 0.04, 0.83, 0.08]
        celdas_hl = VGroup()
        for i in range(4):
            dom = (i == int(np.argmax(pesos_e)))
            num = Text(f"{pesos_e[i]:.2f}", font=FUENTE,
                       font_size=20 if dom else 16,
                       color=COLOR_GIGANTE if dom else ACERO,
                       weight=BOLD if dom else NORMAL)
            vlab = MathTex(rf"\vec v_{{{i + 1}}}",
                           font_size=24 if dom else 19,
                           color=NARANJA_TERRACOTA if dom else ACERO)
            cel = VGroup(num, vlab).arrange(RIGHT, buff=0.1).move_to([J_HL * CW, -i * RH, 0])
            celdas_hl.add(cel)

        vmat = VGroup(filas_v, cols_q, rejilla_v, col_hl, celdas_hl)
        vmat.move_to(DOWN * 0.2).scale(0.9)

        # aparece la matriz de Value: cada palabra -> E -> W_V -> v
        self.play(LaggedStart(*[FadeIn(f, shift=RIGHT * 0.15) for f in filas_v],
                              lag_ratio=0.12), run_time=1.2)
        self.play(Create(rejilla_v),
                  LaggedStart(*[FadeIn(c, shift=DOWN * 0.1) for c in cols_q], lag_ratio=0.1),
                  run_time=1.0)
        # se elige una consulta (este) y se leen sus pesos sobre los Value
        self.play(cols_q[J_HL].animate.set_opacity(1.0),
                  *[cols_q[j].animate.set_opacity(0.4) for j in range(4) if j != J_HL],
                  Create(col_hl), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(c, scale=0.8) for c in celdas_hl], lag_ratio=0.15),
                  run_time=1.0)
        self.play(Indicate(celdas_hl[2], color=COLOR_GIGANTE, scale_factor=1.2),
                  Indicate(filas_v[2], color=COLOR_GIGANTE, scale_factor=1.08),
                  run_time=0.8)
        self._siguiente()

        # ── Acto G: el espacio transformado (centrado); este se acerca ──
        self.play(FadeOut(vmat), run_time=0.5)

        ORIGEN_G = np.array([-2.16, -2.0, 0])
        plano_g = _espacio(ORIGEN_G)
        fantasma_g = plano_g.copy().set_stroke(opacity=0.13)
        plano_g.apply_matrix(M_QK, about_point=ORIGEN_G)
        pts_g = {n: _punto(_PM(ORIGEN_G, *QK[n], M_QK), COL[n], n) for n in QK}

        self.play(
            FadeIn(fantasma_g),
            LaggedStart(GrowArrow(plano_g[1]), GrowArrow(plano_g[2]),
                        Create(plano_g[0]), lag_ratio=0.12),
            LaggedStart(*[FadeIn(pts_g[n], scale=0.6) for n in pts_g], lag_ratio=0.2),
            run_time=1.1,
        )
        # este se desplaza hacia gigante; el punto y su etiqueta se mueven juntos
        destino = _PM(ORIGEN_G, *QK["gigante"], M_QK) + UP * 0.6 + LEFT * 0.55
        shift_este = destino - pts_g["este"][1].get_center()
        flecha = Arrow(pts_g["este"][1].get_center(), destino, buff=0.16,
                       color=NARANJA_TERRACOTA, stroke_width=4,
                       max_tip_length_to_length_ratio=0.22)
        self.play(GrowArrow(flecha), run_time=0.6)
        self.play(pts_g["este"].animate.shift(shift_este), run_time=1.0)
        self.play(
            Flash(destino, color=NARANJA_TERRACOTA, line_length=0.22, num_lines=10),
            Indicate(pts_g["gigante"], color=COLOR_GIGANTE, scale_factor=1.12),
            run_time=0.7,
        )
        self._siguiente()
        self.limpiar_pantalla()
