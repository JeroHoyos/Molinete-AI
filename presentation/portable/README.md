# Presentacion Molinete AI (renderizada)

Carpeta autocontenida con la presentacion ya renderizada. Sirve para presentar en cualquier PC sin necesidad del resto del repositorio, sin compilar nada y sin instalar manim: solo hace falta Python y manim-slides (o ni eso, ver el plan B al final).

## Contenido

- `slides/Presentacion.json` - la definicion de la presentacion: el orden de las diapositivas y que video corresponde a cada una.
- `slides/files/Presentacion/` - los videos de cada diapositiva (incluidas las versiones invertidas para poder retroceder). Las rutas del JSON apuntan aqui de forma relativa, por eso no hay que cambiar la estructura interna ni los nombres.
- `Presentacion.mp4` - la presentacion completa en un solo video, como respaldo.
- `normalizar_rutas.py` - convierte las rutas del JSON al formato de Linux/Mac (ver abajo).
- Este `README.md`.

## Como se genera esta carpeta

No se edita a mano. Desde la carpeta `presentation/` del repositorio:

```bash
uv run python compilar.py
```

Eso renderiza la presentacion y deja aqui la version nueva (reemplaza `slides/` y `Presentacion.mp4`, y normaliza las rutas del JSON automaticamente).

## Llevarla a otro PC

Dos opciones:

1. Copiar esta carpeta `portable/` entera a un USB, tal cual.
2. En zip: desde `presentation/` ejecutar `uv run python comprimir.py`, que crea `presentacion_portable.zip`. Copiar el zip al otro PC junto con `descomprimir.py` y alli ejecutar `python3 descomprimir.py` (o descomprimirlo con cualquier programa de archivos).

## Presentar

1. Instalar el presentador (una sola vez):

   ```bash
   pip install "manim-slides[pyside6]"
   ```

2. Desde ESTA carpeta (la raiz, donde esta este README):

   ```bash
   python -m manim_slides present Presentacion
   ```

   Se usa `python -m manim_slides` en vez de `manim-slides` a secas porque Smart App Control de Windows puede bloquear el lanzador `.exe`.

### Controles durante la presentacion

- Flecha derecha: siguiente diapositiva
- Flecha izquierda: diapositiva anterior
- Espacio: pausar / reanudar
- R: repetir la diapositiva actual
- V: reproducir la diapositiva al reves
- F: pantalla completa
- H: ocultar / mostrar el cursor
- Q: salir

## Si el otro dispositivo es Linux o Mac

El JSON se compila en Windows y puede traer rutas con barra invertida (`slides\files\...`) que Linux no entiende. Antes de presentar, ejecutar una vez desde esta carpeta:

```bash
python3 normalizar_rutas.py
```

Es idempotente e inofensivo en Windows: solo cambia `\` por `/`, y las rutas con `/` funcionan en todos los sistemas. Las compilaciones hechas con `compilar.py` ya salen normalizadas, asi que normalmente no hara ningun cambio; esta aqui por si acaso.

## Plan B

Si en el otro PC no se puede instalar nada, reproducir `Presentacion.mp4` con cualquier reproductor de video y pausar a mano entre diapositivas. Se pierden las pausas interactivas, pero el contenido es el mismo.
