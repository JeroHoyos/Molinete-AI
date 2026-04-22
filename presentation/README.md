
# Ejecutar una presentación con Manim Slides

Estas instrucciones explican cómo preparar el entorno, instalar dependencias, compilar las diapositivas y ejecutar la presentación.

---

# 1. Instalar dependencias

Se recomienda usar [`uv`](https://docs.astral.sh/uv/) para gestionar el entorno virtual e instalar dependencias.

### Instalar uv (si no lo tienes)

```bash
pip install uv
```

### Crear entorno e instalar dependencias

```bash
uv venv .env
uv pip install -r requirements.txt
```

Esto crea el entorno en `.env/` e instala todo en un solo paso.

---

# 2. Activar el entorno virtual

### Windows
```bash
.env\Scripts\activate
```

### Linux / Mac
```bash
source .env/bin/activate
```

---

# 3. Compilar las diapositivas

Para generar el video de las diapositivas:

```bash
py -m manim_slides render main.py Presentacion
```

---

# 4. Presentar las diapositivas

Una vez renderizadas, inicia la presentación:

```bash
py -m manim_slides present Presentacion
```

---

# 5. Controles durante la presentación

* `→` siguiente diapositiva
* `←` diapositiva anterior
* `f` pantalla completa
* `q` salir

---

# 6. Flujo típico de trabajo

```bash
uv venv .env
uv pip install -r requirements.txt
.env\Scripts\activate                        # Windows
# source .env/bin/activate                   # Linux / Mac
py -m manim_slides render main.py Presentacion
py -m manim_slides present Presentacion
```

---

# Notas

* Cada diapositiva se define usando `self.next_slide()`.
* Cada clase de presentación debe heredar de `Slide`.
* Los videos renderizados se guardan en la carpeta `media/`.

# Para cambiar el path entre windows y linux
```bash
sed -i 's/\\\\/\//g' slides/Presentacion.json
```
