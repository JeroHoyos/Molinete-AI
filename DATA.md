# DATA.md — Corpus de Entrenamiento

> Guía completa sobre los datos de entrenamiento de Molinete AI.

---

## Obras de Cervantes

| Campo | Detalle |
|:---|:---|
| **Autor** | Miguel de Cervantes Saavedra |
| **Fuente** | [Project Gutenberg](https://www.gutenberg.org/) |
| **Licencia** | Dominio Público |
| **Tamaño** | ~5–7 MB |
| **Archivo generado** | `cervantes.txt` |

### Contenido del corpus

| Obra | URL en Gutenberg |
|:---|:---|
| Don Quijote de la Mancha | [pg2000.txt](https://www.gutenberg.org/cache/epub/2000/pg2000.txt) |
| Novelas Ejemplares | [pg61202.txt](https://www.gutenberg.org/cache/epub/61202/pg61202.txt) |
| La Galatea | [pg1445.txt](https://www.gutenberg.org/cache/epub/1445/pg1445.txt) |
| Novelas y Teatro | [pg15115.txt](https://www.gutenberg.org/cache/epub/15115/pg15115.txt) |
| Entremeses | [pg57955.txt](https://www.gutenberg.org/cache/epub/57955/pg57955.txt) |

---

## 1. Obtener los datos

La forma más sencilla es usar el script de descarga incluido en el proyecto:

```bash
python download_data.py
```

Este script descargará todas las obras automáticamente desde Project Gutenberg y las concatenará en un único archivo `cervantes.txt`, con cabeceras separadoras entre cada obra.

También puedes descargarlo desde el menú interactivo:

```bash
python molineteai.py
# → Opción 11: Descargar corpus
```

### Descarga manual

Si prefieres hacerlo manualmente con `curl`:

```bash
curl -o quijote.txt    https://www.gutenberg.org/cache/epub/2000/pg2000.txt
curl -o novelas.txt    https://www.gutenberg.org/cache/epub/61202/pg61202.txt
curl -o galatea.txt    https://www.gutenberg.org/cache/epub/1445/pg1445.txt
curl -o teatro.txt     https://www.gutenberg.org/cache/epub/15115/pg15115.txt
curl -o entremeses.txt https://www.gutenberg.org/cache/epub/57955/pg57955.txt
cat quijote.txt novelas.txt galatea.txt teatro.txt entremeses.txt > cervantes.txt
```

---

## 2. Preprocesamiento

El procesamiento es mínimo y directo. No se aplican transformaciones complejas; el tokenizador BPE aprende directamente del texto crudo.

El script `download_data.py` se encarga de:

1. Descargar cada obra por separado.
2. Insertar cabeceras separadoras entre obras para facilitar la lectura del archivo resultante.
3. Unificar todo en `cervantes.txt`.

**Paso opcional:** Los archivos de Project Gutenberg incluyen un bloque legal al inicio y al final de cada obra (*"The Project Gutenberg License"*). Puedes eliminarlos manualmente si deseas un corpus más limpio, aunque en la práctica el tokenizador los maneja sin problemas dado su tamaño relativo.

---

## 3. ¿Por qué Cervantes?

La elección del corpus no es arbitraria. Cervantes ofrece varias ventajas concretas para este tipo de experimento:

**Libre de derechos.** Al ser obras del Siglo de Oro español, están 100% en dominio público y disponibles legalmente en Project Gutenberg.

**Tamaño ideal.** Con ~5–7 MB, el corpus es suficientemente grande para que el modelo aprenda patrones lingüísticos reales, pero lo bastante pequeño para entrenar en una computadora personal en horas razonables.

**Riqueza léxica.** El corpus combina narrativa (*Don Quijote*, *La Galatea*), cuentos cortos (*Novelas Ejemplares*) y teatro (*Entremeses*). Esta variedad obliga al modelo a aprender estructuras de lenguaje diversas: diálogo, descripción, monólogo, verso.

**Estilo inconfundible.** El castellano del Siglo de Oro es fonética y morfológicamente distintivo. Es muy fácil — y divertido — evaluar cualitativamente si el modelo está aprendiendo algo real, observando si empieza a producir construcciones como *"En un lugar de la Mancha"*, arcaísmos o el ritmo característico de Cervantes.

**Contrapunto cultural.** Feste usa Shakespeare (inglés isabelino). Molinete AI usa Cervantes (español del Siglo de Oro). Ambos son contemporáneos y representan cumbres literarias paralelas. La contraposición es tanto lingüística como histórica.

---

## 4. Usar otros textos

Si prefieres experimentar con un corpus diferente, simplemente reemplaza `cervantes.txt` por cualquier archivo de texto plano en UTF-8. Algunos candidatos compatibles:

- **Shakespeare completo** (el corpus original de Feste): disponible en Gutenberg como [pg100.txt](https://www.gutenberg.org/files/100/100-0.txt)
- **El Quijote solo** (versión reducida para entrenamientos más rápidos)
- **Cualquier obra en dominio público** de tu elección

El menú interactivo (opción 11) también incluye la descarga de Shakespeare para facilitar la comparación.

---

## 5. Ética y reproducibilidad

Este proyecto está diseñado con fines **educativos y reproducibles**, por lo que:

- Usamos exclusivamente textos en **dominio público**.
- Evitamos corpus gigantescos que requieran supercomputadoras o acceso privado a datos.
- No usamos datos extraídos sin consentimiento de internet.
- Todo el corpus es descargable con un solo comando y verificable por cualquiera.

Si decides adaptar el proyecto a otro corpus, asegúrate de respetar los términos de licencia del texto que uses.

---

## 6. Estructura del archivo generado

`cervantes.txt` tiene la siguiente organización:

```
==================================================
--- DON QUIJOTE DE LA MANCHA ---
==================================================

[Texto completo de la obra...]


==================================================
--- NOVELAS EJEMPLARES ---
==================================================

[Texto completo de la obra...]

... (y así con cada obra)
```

El tokenizador BPE entrena directamente sobre este archivo sin ningún paso adicional de limpieza.
