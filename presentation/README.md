

# Ejecutar una presentación con Manim Slides

Estas instrucciones explican cómo preparar el entorno, instalar dependencias, compilar las diapositivas y ejecutar la presentación.

---

# 1. Activar el entorno virtual

Primero activa el entorno virtual del proyecto.

### Linux / Mac
```bash
source .venv/bin/activate
````

### Windows

```bash
.venv\Scripts\activate
```

Si no existe un entorno virtual, créalo:

```bash
python -m venv .venv
```

---

# 2. Instalar dependencias

Instalar las dependencias del proyecto usando el archivo `requirements.txt`.

```bash
pip install -r requirements.txt
```

---

# 3. Compilar las diapositivas

Para generar el video de las diapositivas se usa `manim`.

Estructura básica:

```bash
manim_slides render main.py Presentacion
```

---

# 4. Presentar las diapositivas

Una vez renderizadas, se puede iniciar la presentación:

```bash
manim_slides present Presentacion
```

---

# 5. Controles durante la presentación

Controles básicos:

* `→` siguiente diapositiva
* `←` diapositiva anterior
* `f` pantalla completa
* `q` salir

---

# 6. Flujo típico de trabajo

El flujo normal es:

```bash
source venv/bin/activate
pip install -r requirements.txt
manim_slides render presentacion.py Presentacion
manim_slides present Presentacion
```

---

# Notas

* Cada diapositiva se define usando `self.next_slide()`.
* Cada clase de presentación debe heredar de `Slide`.
* Los videos renderizados se guardan en la carpeta `media/`.

# Para cambiar el path entre windows y linux
sed -i 's/\\\\/\//g' slides/Presentacion.json
