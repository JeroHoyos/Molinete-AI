from manim import *
import numpy as np

# Configuración para formato Reel / TikTok (Vertical 9:16)
config.pixel_width = 1080
config.pixel_height = 1920
config.frame_width = 8.0
config.frame_height = 14.22

# Paleta Maquiavélica 
BG_START = "#0F0000"
BG_END = "#000000"
RED_BRIGHT = "#FF0000"     
RED_BLOOD = "#FF3333"      
RED_GLOW = "#FF6666"       
RED_DARK = "#660000"       

class DarkGauss(Scene):
    def construct(self):
        self.camera.background_color = BG_START

        # ==========================================
        # FASE 1: Introducción ominosa
        # ==========================================
        
        title = Text("GAUSS INTEGRAL", font_size=55, color=RED_BRIGHT, weight=BOLD)
        title.to_edge(UP, buff=1.5)
        self.play(FadeIn(title, shift=DOWN, scale=1.2), run_time=2.5, rate_func=rate_functions.ease_out_cubic)

        grid = NumberPlane(
            x_range=[-5, 5, 1], y_range=[-2, 3, 1],
            background_line_style={"stroke_color": RED_DARK, "stroke_width": 2, "stroke_opacity": 0.3},
            axis_config={"stroke_color": RED_DARK, "stroke_opacity": 0.5}
        ).shift(DOWN * 1)
        self.play(Create(grid, run_time=1.5, lag_ratio=0.2))

        axes = Axes(x_range=[-4, 4], y_range=[-0.5, 1.5], x_length=7, y_length=4).shift(UP*1)
        curve = axes.plot(lambda x: np.exp(-x**2), color=RED_BLOOD, stroke_width=8)
        area = axes.get_area(curve, color=RED_DARK, opacity=0.6)

        eq_I = MathTex(r"I = \int_{-\infty}^{\infty} e^{-x^2} dx", font_size=70, color=RED_BRIGHT)
        eq_I.move_to(DOWN * 2)

        self.play(
            Create(curve, run_time=2.5),
            FadeIn(area, run_time=2.5),
            Write(eq_I, run_time=2)
        )
        self.wait(1.5)

        bg_darkness = FullScreenRectangle(color=BG_END, fill_opacity=1).set_z_index(-10)

        # Apagón visual - Prepara la transición
        self.play(
            FadeIn(bg_darkness, run_time=1.5),
            FadeOut(grid, curve, area, axes, run_time=1),
            eq_I.animate(run_time=1.5).move_to(UP * 2).set_color(RED_BRIGHT)
        )
        self.wait(0.5)

        # ==========================================
        # FASE 2: Preparando el terreno
        # ==========================================
        
        eq_I_y = MathTex(r"I = \int_{-\infty}^{\infty} e^{-y^2} dy", font_size=60, color=RED_GLOW).next_to(eq_I, DOWN, buff=0.5)
        self.play(FadeIn(eq_I_y, shift=UP), run_time=0.8)

        eq_step = MathTex(r"I^2 = I \cdot I", font_size=75, color=RED_BRIGHT).move_to(DOWN * 1.5)
        self.play(Write(eq_step, run_time=0.6))
        self.wait(0.5)

        # ==========================================
        # MOMENTO DE TENSIÓN (El "Silencio")
        # ==========================================
        
        # Limpiamos todo el lienzo excepto el paso actual. Cero superposiciones.
        self.play(
            FadeOut(title), 
            FadeOut(eq_I), 
            FadeOut(eq_I_y),
            eq_step.animate.move_to(ORIGIN).scale(1.2), # Lo llevamos al centro exacto
            run_time=1
        )
        
        # Efecto de latido (Heartbeat)
        self.wait(0.5)
        self.play(Indicate(eq_step, scale_factor=1.1, color=RED_BRIGHT), run_time=0.5)
        self.wait(0.3)
        self.play(Indicate(eq_step, scale_factor=1.15, color=WHITE), run_time=0.3)
        self.wait(1.2) # <- SILENCIO ABSOLUTO ANTES DE EXPLOTAR

        # ==========================================
        # FASE 3: FRENESÍ ANALÍTICO (-25% Velocidad)
        # ==========================================
        
        # Tamaños ajustados para que nada choque con los bordes del video vertical
        fast_equations = [
            MathTex(r"I^2 = \left( \int e^{-x^2} dx \right) \left( \int e^{-y^2} dy \right)", font_size=45, color=RED_BLOOD),
            MathTex(r"I^2 = \iint_{\mathbb{R}^2} e^{-(x^2+y^2)} dx dy", font_size=60, color=RED_BRIGHT),
            MathTex(r"x = r \cos\theta, \quad y = r \sin\theta", font_size=50, color=RED_GLOW),
            MathTex(r"dx dy = r \, dr \, d\theta", font_size=55, color=RED_GLOW),
            MathTex(r"I^2 = \int_{0}^{2\pi} \int_{0}^{\infty} e^{-r^2} r \, dr \, d\theta", font_size=55, color=RED_BLOOD),
            MathTex(r"I^2 = 2\pi \int_{0}^{\infty} e^{-r^2} r \, dr", font_size=60, color=RED_BLOOD),
            MathTex(r"u = r^2 \implies r \, dr = \frac{du}{2}", font_size=50, color=RED_GLOW),
            MathTex(r"I^2 = \pi \int_{0}^{\infty} e^{-u} du", font_size=65, color=RED_BRIGHT),
            MathTex(r"I^2 = \pi \lim_{b \to \infty} \left[ -e^{-u} \right]_{0}^{b}", font_size=60, color=RED_BRIGHT),
            MathTex(r"I^2 = \pi (0 + 1)", font_size=75, color=RED_BRIGHT),
            MathTex(r"I^2 = \pi", font_size=110, color=RED_BRIGHT)
        ]

        # Usamos ReplacementTransform para que las cosas NUNCA se encimen
        current_eq = eq_step
        for tex in fast_equations:
            tex.move_to(ORIGIN) # Todo ocurre exactamente en el centro
            # run_time de 0.2s es ~25% más lento que antes, más digerible
            self.play(ReplacementTransform(current_eq, tex), run_time=0.2)
            self.wait(0.1)
            current_eq = tex
            
        self.wait(0.4) # Micro-pausa antes del drop final
        
        # ==========================================
        # FASE 4: CLÍMAX INFERNAL
        # ==========================================
        
        final_ans = MathTex(r"I = \sqrt{\pi}", font_size=180, color=RED_BRIGHT).move_to(ORIGIN)
        
        # Flash blanco para el "Beat Drop"
        flash = FullScreenRectangle(color=WHITE, fill_opacity=1).set_z_index(10)
        self.play(FadeIn(flash, run_time=0.05)) # Entra en 1 frame
        
        self.play(
            ReplacementTransform(current_eq, final_ans),
            FadeOut(flash, run_time=0.4) 
        )
        
        # Aura explosiva
        glow_1 = SurroundingRectangle(final_ans, color=RED_BRIGHT, buff=0.4, stroke_width=6, stroke_opacity=0.8)
        glow_2 = SurroundingRectangle(final_ans, color=RED_BLOOD, buff=0.6, stroke_width=12, stroke_opacity=0.4)
        glow_3 = SurroundingRectangle(final_ans, color=RED_DARK, buff=0.9, stroke_width=20, stroke_opacity=0.2)
        
        self.play(Create(glow_1), Create(glow_2), Create(glow_3), run_time=0.2)
        self.play(Indicate(final_ans, scale_factor=1.15, color=WHITE), run_time=0.5)
        
        self.wait(3)