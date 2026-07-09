# Corpus de entrenamiento

Molinete AI se entrena con las obras de Miguel de Cervantes, descargadas de [Project Gutenberg](https://www.gutenberg.org/) (dominio público). Todo se concatena en un único archivo `cervantes.txt` de ~5-7 MB.

| Obra | URL en Gutenberg |
|:---|:---|
| Don Quijote de la Mancha | [pg2000.txt](https://www.gutenberg.org/cache/epub/2000/pg2000.txt) |
| Novelas Ejemplares | [pg61202.txt](https://www.gutenberg.org/cache/epub/61202/pg61202.txt) |
| La Galatea | [pg1445.txt](https://www.gutenberg.org/cache/epub/1445/pg1445.txt) |
| Novelas y Teatro | [pg15115.txt](https://www.gutenberg.org/cache/epub/15115/pg15115.txt) |
| Entremeses | [pg57955.txt](https://www.gutenberg.org/cache/epub/57955/pg57955.txt) |

## Obtener los datos

Inicia la interfaz web y usa la tarjeta **Descargar Corpus de Cervantes**:

```bash
uv run web/server.py
# Abrir http://localhost:7860
```

Esto descarga las obras, inserta cabeceras separadoras entre ellas y genera `cervantes.txt`. No hay más preprocesamiento: el tokenizador BPE aprende directamente del texto crudo, incluyendo los bloques legales de Gutenberg.


## Usar otros textos

Reemplaza `cervantes.txt` por cualquier archivo de texto plano en UTF-8, sin preprocesamiento adicional. Por ejemplo, [Shakespeare completo](https://www.gutenberg.org/files/100/100-0.txt) (el corpus original de Feste, también descargable desde la interfaz web) o solo el Quijote para entrenamientos más rápidos. Si usas otro corpus, respeta los términos de licencia del texto.
