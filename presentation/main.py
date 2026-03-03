from manim import *
from manim_slides import Slide
import random

class Presentacion(Slide):
    def construct(self):
        # Fondo limpio
        self.camera.background_color = WHITE

        # =========================================================
        # PALETA DE COLORES GLOBAL
        # =========================================================
        RUST_COLOR = "#CE412B"
        HIGHLIGHT_COLOR = "#00599C" 
        ALERT_COLOR = "#D32F2F"
        ALERT_COLOR = "#D32F2F"
        TOKEN_COLOR_1 = "#8E44AD" 
        TOKEN_COLOR_2 = "#E67E22" 
        TOKEN_COLOR_FINAL = "#2980B9" 
        # =========================================================
        # SLIDE 1 — Animación del Transformer
        # =========================================================
        
        # 1. Título y subtítulo alineados arriba (UP)
        titulo_principal = Text("Construyendo un Transformer con Rust", font_size=40, weight=BOLD, color=BLACK, t2c={"Rust": RUST_COLOR})
        linea_principal = Underline(titulo_principal, color=RUST_COLOR, stroke_width=4)
        subtitulo = Text("Jerónimo Hoyos Botero", font_size=24, color=DARK_GRAY)
        
        grupo_nombres = VGroup(titulo_principal, linea_principal, subtitulo)
        grupo_nombres.arrange(DOWN, buff=0.2)
        grupo_nombres.to_edge(UP, buff=0.5) # Cambiado de DOWN a UP

        # 2. Cajas del Transformer (ligeramente más abajo para centrar la animación)
        cajas = VGroup(*[
            Rectangle(
                width=2.6, height=1.8, 
                fill_color="#F2F2F2", fill_opacity=1, 
                stroke_color=DARK_GRAY, stroke_width=1
            ).shift(UP * i * 0.05 + RIGHT * i * 0.05)
            for i in range(6)
        ])
        
        # Desplazamos las cajas hacia abajo (DOWN * 0.5) para dar espacio al título
        cajas.move_to(LEFT * 5.5 + DOWN * 0.5) 
        
        texto_transformer = Text("Molinete AI", font_size=20, weight=BOLD, color=BLACK).move_to(cajas.get_center())
        grupo_transformer = VGroup(cajas, texto_transformer)

        flecha = Arrow(start=LEFT, end=RIGHT, color=BLACK, max_tip_length_to_length_ratio=0.15).scale(0.5)
        flecha.next_to(cajas, RIGHT, buff=0.2)

        punto_alineacion_izq = flecha.get_right() + RIGHT * 0.3
        
        # La altura del texto también se adapta automáticamente porque depende de 'cajas'
        pos_text_start = punto_alineacion_izq + RIGHT * 3.8 + UP * 1.6 

        def create_probs(target_word):
            dummies = ["hidalgo", "espada", "vino", "capa", "oro", "plaza", "mujer", "caballo", "sed", "honor", "cielo", "muerte"]
            random.shuffle(dummies)
            
            p1 = random.uniform(0.75, 0.95)
            p2 = random.uniform(0.05, 0.15)
            p3 = random.uniform(0.01, 0.04)
            p4 = random.uniform(0.01, 0.02)
            
            datos = [
                (target_word, p1, "#73A98B", True),
                (dummies[0], p2, "#B4C6D1", False),
                (dummies[1], p3, "#DDE6ED", False),
                (dummies[2], p4, "#DDE6ED", False),
            ]
            
            textos = VGroup(*[Text(d[0], font_size=16, color=BLACK) for d in datos])
            barras = VGroup(*[Rectangle(height=0.15, width=max(d[1] * 2.0, 0.05), fill_color=d[2], fill_opacity=1, stroke_width=0) for d in datos])
            porcentajes = VGroup(*[Text(f"{int(d[1]*100)}%", font_size=14, color=BLACK) for d in datos])
            
            textos.arrange(DOWN, aligned_edge=RIGHT, buff=0.15)
            barras.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
            porcentajes.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
            
            for i in range(len(datos)):
                barras[i].match_y(textos[i])
                porcentajes[i].match_y(textos[i])
                
            columnas = VGroup(textos, barras, porcentajes).arrange(RIGHT, buff=0.1)
            row0 = VGroup(textos[0], barras[0], porcentajes[0])
            
            fondo = Rectangle(
                width=columnas.width + 0.4, height=row0.height + 0.2, 
                fill_color="#EBE3BA", fill_opacity=1, stroke_width=0.5, stroke_color=DARK_GRAY
            )
            fondo.move_to(row0)
            fondo.match_x(columnas)
            
            puntos = Text("⋮", font_size=18, color=BLACK, weight=BOLD)
            puntos.next_to(columnas, DOWN, buff=0.1)
            
            todo = VGroup(fondo, columnas, puntos)
            todo.move_to(punto_alineacion_izq, aligned_edge=LEFT) 
            return todo

        # Animación de entrada del título unificada
        self.play(Write(titulo_principal), Create(linea_principal), FadeIn(subtitulo, shift=DOWN), run_time=1.5)
        self.next_slide()

        self.play(FadeIn(cajas, shift=RIGHT), Write(texto_transformer), run_time=1)
        self.play(GrowArrow(flecha), run_time=0.4)

        lineas = [
            ["retorciendo", "el", "mostacho", "soldadesco,"],
            ["por", "ver", "que", "ya", "su", "bolsa", "le", "repica,"],
            ["a", "un", "corrillo", "llegó", "de", "gente", "rica"],
            ["y", "en", "el", "nombre", "de", "Dios", "pidió", "refresco."],
            ["Den", "voacedes,", "por", "Dios,", "a", "mi", "pobreza,"],
            ["les", "dice;", "donde", "no,", "por", "ocho", "santos"],
            ["que", "haré", "lo", "que", "hacer", "suelo", "sin", "tardanza."]
        ]
        
        current_y = pos_text_start[1]
        current_probs = create_probs("...")
        current_probs.set_opacity(0)
        self.add(current_probs)

        for linea in lineas:
            current_x = pos_text_start[0]
            
            for word in linea:
                new_probs = create_probs(word)
                word_mob = Text(word, font_size=20, color=BLACK)
                word_mob.move_to(np.array([current_x, current_y, 0]), aligned_edge=LEFT)
                
                self.play(
                    ReplacementTransform(current_probs, new_probs),
                    FadeIn(word_mob, shift=LEFT*0.05),
                    run_time=0.6 
                )
                self.wait(0.2) 
                
                current_probs = new_probs
                current_x += word_mob.width + 0.1
                
            current_y -= 0.45

        self.next_slide()
        
        # Limpiar pantalla para la siguiente slide
        elementos_a_borrar = [mob for mob in self.mobjects]
        self.play(*[FadeOut(mob) for mob in elementos_a_borrar])
        # =========================================================
        # SLIDE 2 — Créditos
        # =========================================================
        titulo_creditos = Text("Esta presentación se basa en:", color=BLACK, font_size=48, t2c={"basa en:": HIGHLIGHT_COLOR}).to_edge(UP)
        linea_creditos = Underline(titulo_creditos, color=HIGHLIGHT_COLOR, stroke_width=4)

        # Nota: Cambié el \ por / en la ruta de la imagen para evitar errores de escape en Python
        imagen_creditos = ImageMobject("assets/creditos_guia_original.png")
        imagen_creditos.scale(1.5)
        imagen_creditos.next_to(linea_creditos, DOWN, buff=0.8)

        self.play(Write(titulo_creditos), Create(linea_creditos))
        self.play(FadeIn(imagen_creditos, shift=UP))
        
        self.next_slide()
        self.play(FadeOut(Group(titulo_creditos, linea_creditos, imagen_creditos)))

                # =========================================================
        # SLIDE 8 — ¿Qué es un Transformer?
        # =========================================================
        titulo6 = Text(
            "La Arquitectura Transformer", 
            font_size=50, 
            color=BLACK, 
            weight=BOLD,
            t2c={"Transformer": HIGHLIGHT_COLOR}
        ).to_edge(UP)
        linea6 = Underline(titulo6, color=HIGHLIGHT_COLOR, stroke_width=4)

        punto_t1 = Text("• Es el 'motor' de los LLM modernos\n  (GPT, Llama, Molinete).", font_size=28, color=BLACK)
        punto_t2 = Text("• Mecanismo de Atención:\n  Descubre qué palabras están conectadas\n  entre sí, sin importar la distancia.", font_size=28, color=DARK_GRAY)
        punto_t3 = Text("• Procesamiento en Paralelo:\n  A diferencia de modelos antiguos, lee\n  todo el texto de golpe, no palabra por palabra.", font_size=28, color=BLACK)

        textos_transformer = VGroup(punto_t1, punto_t2, punto_t3).arrange(DOWN, aligned_edge=LEFT, buff=0.6)
        imagen_transformer = ImageMobject("assets/arquitectura_transforme.png").scale(0.5)

        # CORRECCIÓN AQUÍ: Cambiado VGroup por Group
        contenido_transformer = Group(textos_transformer, imagen_transformer).arrange(RIGHT, buff=1.0)
        contenido_transformer.next_to(linea6, DOWN, buff=0.8)

        self.play(Write(titulo6), Create(linea6))
        
        self.next_slide()
        self.play(FadeIn(imagen_transformer, shift=UP*0.3))
        
        self.next_slide()
        self.play(LaggedStart(
            FadeIn(punto_t1, shift=RIGHT*0.2),
            FadeIn(punto_t2, shift=RIGHT*0.2),
            FadeIn(punto_t3, shift=RIGHT*0.2),
            lag_ratio=0.5
        ))

        self.next_slide()
        self.play(FadeOut(Group(titulo6, linea6, contenido_transformer)))

        # =========================================================
        # SLIDE 7 — Molinete AI y su origen
        # =========================================================
        titulo5 = Text("Molinete AI", font_size=55, color=HIGHLIGHT_COLOR, weight=BOLD).to_edge(UP)
        linea5 = Underline(titulo5, color=HIGHLIGHT_COLOR, stroke_width=4)

        descripcion = Text("Un modelo de lenguaje entrenado con El Quijote.", font_size=32, color=DARK_GRAY)
        subtitulo = Text("¿Por qué “Molinete”?", font_size=40, color=BLACK, weight=BOLD)
        
        punto1 = Text("• Hace referencia al universo cervantino.", font_size=28, color=BLACK)
        punto2 = Text("• Si Feste toma su identidad del bufón\n  ingenioso en Twelfth Night...", font_size=28, color=DARK_GRAY, slant=ITALIC)
        punto3 = Text("• ...Molinete alude a los fieros oponentes\n  (molinos) del Hidalgo.", font_size=28, color=BLACK)

        grupo_puntos = VGroup(punto1, punto2, punto3).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        textos_molinete = VGroup(descripcion, subtitulo, grupo_puntos).arrange(DOWN, aligned_edge=LEFT, buff=0.8)
        
        imagen_molino = ImageMobject("assets/quijote_vs_molinos.png").scale(1) 
        
        contenido_completo = Group(textos_molinete, imagen_molino).arrange(RIGHT, buff=0.5)
        contenido_completo.next_to(linea5, DOWN, buff=0.8)

        self.play(Write(titulo5), Create(linea5))
        self.play(FadeIn(descripcion, shift=UP*0.3))
        
        self.next_slide()
        
        self.play(
            FadeIn(subtitulo, shift=RIGHT*0.3),
            FadeIn(imagen_molino, shift=LEFT*0.3)
        )
        
        self.play(LaggedStart(
            FadeIn(punto1, shift=RIGHT*0.2),
            FadeIn(punto2, shift=RIGHT*0.2),
            FadeIn(punto3, shift=RIGHT*0.2),
            lag_ratio=0.5
        ))

        self.next_slide()
        self.play(FadeOut(Group(titulo5, linea5, contenido_completo)))


        # =========================================================
        # SLIDE 3 — El problema con Python
        # =========================================================
        titulo1 = Text("¿Por qué no Python?", font_size=60, color=BLACK, t2c={"Python?": HIGHLIGHT_COLOR}).to_edge(UP)
        linea1 = Underline(titulo1, color=HIGHLIGHT_COLOR, stroke_width=4)

        puntos1 = VGroup(
            Text("• Excelente para prototipar, pero...", font_size=36, color=BLACK),
            Text("• El Garbage Collector te quita el control", font_size=36, color=BLACK, t2c={"Garbage Collector": ALERT_COLOR}),
            Text("• No hay control fino sobre la memoria", font_size=36, color=BLACK, t2c={"control fino": ALERT_COLOR}),
            Text("• En un LLM, cada byte y ciclo de CPU cuenta", font_size=36, color=BLACK, t2c={"cada byte": HIGHLIGHT_COLOR}),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5)

        puntos1.next_to(linea1, DOWN, buff=0.8).shift(LEFT * 2)
        imagen1 = ImageMobject("assets/logo_python.png").scale(0.8).to_edge(RIGHT, buff=1.5)

        self.play(Write(titulo1), Create(linea1))
        self.play(FadeIn(imagen1, shift=DOWN), imagen1.animate.scale(1.1).scale(1/1.1))
        self.play(LaggedStart(*[FadeIn(p, shift=RIGHT*0.5) for p in puntos1], lag_ratio=0.2))

        self.next_slide()

        self.play(
            FadeOut(puntos1, shift=LEFT),
            FadeOut(imagen1, shift=RIGHT),
            Uncreate(linea1),
            FadeOut(titulo1, shift=UP)
        )

        # =========================================================
        # SLIDE 4 — El status quo de C++
        # =========================================================
        titulo2 = Text("¿Por qué no C++?", font_size=60, color=BLACK, t2c={"C++?": HIGHLIGHT_COLOR}).to_edge(UP)
        linea2 = Underline(titulo2, color=HIGHLIGHT_COLOR, stroke_width=4)

        puntos2 = VGroup(
            Text("• Es el rey actual del rendimiento (ej. llama.cpp)", font_size=36, color=BLACK),
            Text("• Permite control total del hardware", font_size=36, color=BLACK),
            Text("• PERO... ya todo está hecho en C++", font_size=36, color=BLACK, t2c={"ya todo está hecho": ALERT_COLOR}),
            Text("• No ofrece el mismo factor de innovación y reto", font_size=36, color=BLACK),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)

        puntos2.next_to(linea2, DOWN, buff=0.8).shift(LEFT * 2)
        imagen2 = ImageMobject("assets/logo_cpp.png").scale(0.25).to_edge(RIGHT, buff=2)

        self.play(Write(titulo2), Create(linea2))
        self.play(GrowFromCenter(imagen2))
        self.play(LaggedStart(*[FadeIn(p, shift=UP*0.3) for p in puntos2], lag_ratio=0.2))

        self.next_slide()

        self.play(
            FadeOut(puntos2, shift=DOWN),
            FadeOut(imagen2, scale=0.5),
            Uncreate(linea2),
            FadeOut(titulo2, shift=UP)
        )

        # =========================================================
        # SLIDE 5 — La magia de Rust
        # =========================================================
        titulo3 = Text("Rust: El lenguaje en crecimiento", font_size=60, color=BLACK, t2c={"Rust:": RUST_COLOR}).to_edge(UP)
        linea3 = Underline(titulo3, color=RUST_COLOR, stroke_width=4)

        puntos3 = VGroup(
            Text("• ¡Es un lenguaje increíblemente chévere!", font_size=36, color=BLACK, t2c={"chévere!": HIGHLIGHT_COLOR}),
            Text("• Control de memoria nivel C++, pero SEGURO", font_size=36, color=BLACK, t2c={"SEGURO": HIGHLIGHT_COLOR}),
            Text("• Sistema de Ownership = Cero data races", font_size=36, color=BLACK, t2c={"Ownership": RUST_COLOR}),
            Text("• Un terreno fértil e interesante para construir LLMs", font_size=36, color=BLACK, t2c={"interesante": HIGHLIGHT_COLOR}),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5)

        puntos3.next_to(linea3, DOWN, buff=0.8).shift(LEFT * 2)
        imagen3 = ImageMobject("assets/logo_rust.png").scale(0.25).to_edge(RIGHT, buff=2)

        self.play(Write(titulo3), Create(linea3))
        
        imagen3.rotate(-PI/2)
        self.play(FadeIn(imagen3), imagen3.animate.rotate(PI/2).scale(1.1), run_time=1.5)
        self.play(imagen3.animate.scale(1/1.1), run_time=0.5) 

        self.play(LaggedStart(*[FadeIn(p, shift=RIGHT*0.5) for p in puntos3], lag_ratio=0.2))

        self.next_slide()
        
        self.play(FadeOut(Group(titulo3, linea3, puntos3, imagen3)))


        # =========================================================
        # SLIDE 6 — El problema de "Strawberry"
        # =========================================================
        titulo1 = Text("¿Por qué los LLM no saben 'leer'?", font_size=50, color=BLACK, t2c={"'leer'?": ALERT_COLOR}).to_edge(UP)
        linea1 = Underline(titulo1, color=ALERT_COLOR, stroke_width=4)

        pregunta = Text("Usuario: ¿Cuántas letras 'r' hay en 'strawberry'?", font_size=32, color=DARK_GRAY)
        respuesta = Text("LLM: Hay 2 letras 'r' en 'strawberry'.", font_size=32, color=BLACK, t2c={"2": ALERT_COLOR})
        
        grupo_chat = VGroup(pregunta, respuesta).arrange(DOWN, aligned_edge=LEFT, buff=0.8).shift(UP*1)

        # Representación visual de cómo lo "ve" el LLM (Tokens ficticios)
        texto_explicacion = Text("Para el modelo, la palabra no son letras, son 'pedazos':", font_size=28, color=DARK_GRAY)
        
        # Bloques simulando tokens
        tokens_straw = VGroup(
            self.crear_bloque_token("str", LIGHT_GREY),
            self.crear_bloque_token("aw", LIGHT_GREY),
            self.crear_bloque_token("berry", LIGHT_GREY)
        ).arrange(RIGHT, buff=0.2)
        
        grupo_visual = VGroup(texto_explicacion, tokens_straw).arrange(DOWN, buff=0.5).next_to(grupo_chat, DOWN, buff=1)

        self.play(Write(titulo1), Create(linea1))
        self.play(FadeIn(pregunta, shift=RIGHT*0.5))
        self.wait(0.5)
        self.play(FadeIn(respuesta, shift=RIGHT*0.5))
        
        self.next_slide()

        self.play(FadeIn(texto_explicacion))
        self.play(LaggedStart(*[GrowFromCenter(t) for t in tokens_straw], lag_ratio=0.2))
        
        # Resaltamos que la 'r' está escondida dentro de los bloques
        cruz = Cross(tokens_straw, stroke_color=ALERT_COLOR, stroke_width=6)
        self.play(Create(cruz))

        self.next_slide()

        self.play(
            FadeOut(grupo_chat), FadeOut(grupo_visual), FadeOut(cruz),
            FadeOut(titulo1), Uncreate(linea1)
        )

        # =========================================================
        # SLIDE 7 — ¿Cómo tokenizamos entonces?
        # =========================================================
        titulo2 = Text("La Tokenización", font_size=50, color=BLACK, t2c={"Tokenización": HIGHLIGHT_COLOR}).to_edge(UP)
        linea2 = Underline(titulo2, color=HIGHLIGHT_COLOR, stroke_width=4)

        puntos2 = VGroup(
            Text("1. Por palabra: Vocabulario infinito, ineficiente.", font_size=32, color=BLACK),
            Text("2. Por carácter: Secuencias muy largas, pierde contexto.", font_size=32, color=BLACK),
            Text("3. Sub-palabras (BPE): Una opción más equilibrada.", font_size=32, color=BLACK, weight=BOLD, t2c={"Sub-palabras (BPE):": HIGHLIGHT_COLOR}),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.6).shift(LEFT * 1)

        self.play(Write(titulo2), Create(linea2))
        self.play(LaggedStart(*[FadeIn(p, shift=RIGHT*0.3) for p in puntos2], lag_ratio=0.3))

        self.next_slide()

        self.play(FadeOut(puntos2), FadeOut(titulo2), Uncreate(linea2))

        # =========================================================
        # SLIDE 8 — Animación de Byte Pair Encoding (BPE)
        # =========================================================
        titulo3 = Text("Byte Pair Encoding (BPE)", font_size=50, color=BLACK, t2c={"Byte Pair Encoding (BPE)": TOKEN_COLOR_FINAL}).to_edge(UP)
        linea3 = Underline(titulo3, color=TOKEN_COLOR_FINAL, stroke_width=4)

        explicacion_bpe = Text("Fusión iterativa de los pares más frecuentes:", font_size=28, color=DARK_GRAY).next_to(linea3, DOWN, buff=0.5)

        self.play(Write(titulo3), Create(linea3), FadeIn(explicacion_bpe))

        # Secuencia inicial: t a c o t a c o
        letras = ["t", "a", "c", "o", "t", "a", "c", "o"]
        fila_actual = VGroup(*[self.crear_bloque_token(letra, LIGHT_GREY) for letra in letras]).arrange(RIGHT, buff=0.2).center()
        
        self.play(LaggedStart(*[FadeIn(b, shift=DOWN*0.5) for b in fila_actual], lag_ratio=0.1))
        self.next_slide()

        # PASO 1: Fusionar 't' y 'a'
        texto_paso1 = Text("Paso 1: 't' y 'a' son el par más común", font_size=24, color=BLACK).next_to(fila_actual, DOWN, buff=1)
        self.play(FadeIn(texto_paso1))
        
        # Animamos la fusión de los índices (0,1) y (4,5)
        fila_paso1 = VGroup(
            self.crear_bloque_token("ta", TOKEN_COLOR_1),
            self.crear_bloque_token("c", LIGHT_GREY),
            self.crear_bloque_token("o", LIGHT_GREY),
            self.crear_bloque_token("ta", TOKEN_COLOR_1),
            self.crear_bloque_token("c", LIGHT_GREY),
            self.crear_bloque_token("o", LIGHT_GREY)
        ).arrange(RIGHT, buff=0.2).center()

        self.play(
            ReplacementTransform(VGroup(fila_actual[0], fila_actual[1]), fila_paso1[0]),
            ReplacementTransform(fila_actual[2], fila_paso1[1]),
            ReplacementTransform(fila_actual[3], fila_paso1[2]),
            ReplacementTransform(VGroup(fila_actual[4], fila_actual[5]), fila_paso1[3]),
            ReplacementTransform(fila_actual[6], fila_paso1[4]),
            ReplacementTransform(fila_actual[7], fila_paso1[5]),
        )
        fila_actual = fila_paso1
        self.next_slide()

        # PASO 2: Fusionar 'c' y 'o'
        texto_paso2 = Text("Paso 2: 'c' y 'o' son el par más común", font_size=24, color=BLACK).next_to(fila_actual, DOWN, buff=1)
        self.play(ReplacementTransform(texto_paso1, texto_paso2))

        fila_paso2 = VGroup(
            self.crear_bloque_token("ta", TOKEN_COLOR_1),
            self.crear_bloque_token("co", TOKEN_COLOR_2),
            self.crear_bloque_token("ta", TOKEN_COLOR_1),
            self.crear_bloque_token("co", TOKEN_COLOR_2)
        ).arrange(RIGHT, buff=0.2).center()

        self.play(
            ReplacementTransform(fila_actual[0], fila_paso2[0]),
            ReplacementTransform(VGroup(fila_actual[1], fila_actual[2]), fila_paso2[1]),
            ReplacementTransform(fila_actual[3], fila_paso2[2]),
            ReplacementTransform(VGroup(fila_actual[4], fila_actual[5]), fila_paso2[3]),
        )
        fila_actual = fila_paso2
        self.next_slide()

        # PASO 3: Fusionar 'ta' y 'co'
        texto_paso3 = Text("Paso 3: 'ta' y 'co' forman un nuevo token", font_size=24, color=BLACK).next_to(fila_actual, DOWN, buff=1)
        self.play(ReplacementTransform(texto_paso2, texto_paso3))

        fila_paso3 = VGroup(
            self.crear_bloque_token("taco", TOKEN_COLOR_FINAL),
            self.crear_bloque_token("taco", TOKEN_COLOR_FINAL)
        ).arrange(RIGHT, buff=0.2).center()

        self.play(
            ReplacementTransform(VGroup(fila_actual[0], fila_actual[1]), fila_paso3[0]),
            ReplacementTransform(VGroup(fila_actual[2], fila_actual[3]), fila_paso3[1]),
        )
        
        self.next_slide()
        self.play(FadeOut(VGroup(titulo3, linea3, explicacion_bpe, fila_paso3, texto_paso3)))

        # =========================================================
        # SLIDE 9 — El dilema del tamaño del Vocabulario
        # =========================================================
        titulo4 = Text("El Tamaño del Vocabulario", font_size=55, color=BLACK, t2c={"Vocabulario": HIGHLIGHT_COLOR}).to_edge(UP)
        linea4 = Underline(titulo4, color=HIGHLIGHT_COLOR, stroke_width=4)

        # 1. Crear una tabla visual limpia
        encabezados = VGroup(
            Text("Vocabulario", font_size=30, color=DARK_GRAY, weight=BOLD),
            Text("Total Tokens", font_size=30, color=DARK_GRAY, weight=BOLD),
            Text("Compresión", font_size=30, color=DARK_GRAY, weight=BOLD)
        ).arrange(RIGHT, buff=1.2)

        def crear_fila(v, t, c, color_destacado=BLACK):
            return VGroup(
                Text(v, font_size=28, color=BLACK),
                Text(t, font_size=28, color=BLACK),
                Text(c, font_size=28, color=color_destacado, weight=BOLD)
            ).arrange(RIGHT, buff=1.2)

        fila1 = crear_fila("256", "2,168,312", "1.00x")
        fila2 = crear_fila("1,024", "724,453", "2.99x")
        fila3 = crear_fila("20,534", "460,900", "4.70x", HIGHLIGHT_COLOR)

        # Alinear las columnas manualmente
        for fila in [fila1, fila2, fila3]:
            fila[0].match_x(encabezados[0])
            fila[1].match_x(encabezados[1])
            fila[2].match_x(encabezados[2])

        tabla = VGroup(encabezados, fila1, fila2, fila3).arrange(DOWN, buff=0.4).shift(UP * 0.8)

       # 2. Textos de Trade-offs
        tradeoff_titulo = Text("El Trade-off (Compensación):", font_size=32, color=BLACK, weight=BOLD)
        
        pros = Text("✅ Más vocabulario = Textos más cortos = Inferencia rápida", font_size=26, color=HIGHLIGHT_COLOR)
        cons = Text("❌ Más vocabulario = Matriz de Embeddings gigante = Más VRAM", font_size=26, color=ALERT_COLOR)
        
        textos_inferiores = VGroup(tradeoff_titulo, pros, cons).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        textos_inferiores.next_to(tabla, DOWN, buff=0.8)

        # Animaciones de la Slide 6
        self.play(Write(titulo4), Create(linea4))
        self.play(FadeIn(encabezados))
        self.play(LaggedStart(
            FadeIn(fila1, shift=UP*0.2), 
            FadeIn(fila2, shift=UP*0.2), 
            FadeIn(fila3, shift=UP*0.2), 
            lag_ratio=0.3
        ))
        
        self.next_slide()
        
        # Mostrar los trade-offs uno por uno
        self.play(FadeIn(tradeoff_titulo))
        self.play(FadeIn(pros, shift=RIGHT*0.2))
        self.play(FadeIn(cons, shift=RIGHT*0.2))
        
        self.next_slide()
        self.play(FadeOut(VGroup(titulo4, linea4, tabla, textos_inferiores)))
        
        self.next_slide()
        
        # Mostrar los trade-offs uno por uno
        self.play(FadeIn(tradeoff_titulo))
        self.play(FadeIn(pros, shift=RIGHT*0.2))
        self.play(FadeIn(cons, shift=RIGHT*0.2))

        self.next_slide()
        self.play(FadeOut(VGroup(titulo4, linea4, tabla, textos_inferiores)))

    # =========================================================
    # FUNCIONES AUXILIARES 
    # =========================================================
    def crear_bloque_token(self, texto, color_fondo):
        """
        Función auxiliar movida FUERA de construct() para evitar errores de indentación.
        """
        texto_mob = Text(texto, font_size=36, color=BLACK)
        ancho = max(0.8, texto_mob.width + 0.4)
        fondo = RoundedRectangle(corner_radius=0.1, width=ancho, height=0.8, color=color_fondo, fill_opacity=0.4, stroke_color=DARK_GRAY)
        texto_mob.move_to(fondo.get_center())
        return VGroup(fondo, texto_mob)