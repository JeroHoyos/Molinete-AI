# Datos de Entrenamiento

## Don Quijote de la Mancha

**Autor:** Miguel de Cervantes  
**Fuente:** Project Gutenberg  
**URL:** https://www.gutenberg.org/cache/epub/2000/pg2000.txt  
**Licencia:** Dominio Público  
**Tamaño:** ~2–3 MB de texto plano  
**Contenido:** Texto completo de *Don Quijote de la Mancha*

---

# Obtener los Datos

Puedes descargar el texto directamente desde Project Gutenberg:

```bash
curl -o quijote.txt https://www.gutenberg.org/cache/epub/2000/pg2000.txt
```

Alternativamente, el archivo puede incluirse directamente en este repositorio, ya que está en **dominio público**.

---

# Estado Legal

*Don Quijote de la Mancha* fue publicado entre **1605 y 1615**, por lo que se encuentra completamente en **dominio público en todo el mundo**.

Los textos distribuidos por Project Gutenberg pueden utilizarse libremente para cualquier propósito.  
Durante el preprocesamiento eliminamos los **encabezados y pies de página legales** añadidos por Gutenberg.

---

# Preprocesamiento de los Datos

El preprocesamiento es mínimo:

1. Eliminar encabezado y pie de página de Project Gutenberg  
2. Mantener el texto lo más intacto posible  
3. Preservar puntuación y formato  
4. Entrenar el tokenizer directamente sobre el texto

Pipeline de preparación:

```
texto_crudo
   ↓
eliminar_header_footer_gutenberg
   ↓
entrenar_tokenizador
   ↓
dataset_entrenamiento
```

---

# ¿Por Qué Don Quijote?

*Don Quijote* es un excelente corpus para entrenar un modelo de lenguaje educativo.

### 1. Dominio Público

No existen problemas de licencias.  
El texto puede distribuirse libremente dentro del repositorio.

### 2. Tamaño Adecuado

El corpus es lo suficientemente pequeño para:

- entrenar en CPU
- realizar experimentos en un portátil
- entrenar tokenizadores rápidamente

pero lo bastante grande para generar patrones interesantes.

### 3. Lenguaje Rico

La obra contiene:

- narración
- diálogos
- descripciones detalladas
- vocabulario variado

Esto ayuda al modelo a aprender **estructuras complejas del lenguaje**.

### 4. Estilo Reconocible

El estilo narrativo de Cervantes es muy característico, lo que facilita observar cuándo el modelo comienza a **imitar la voz del autor**.

### 5. Valor Cultural

Es una de las obras más importantes de la literatura universal, lo que convierte al dataset en algo interesante también desde el punto de vista educativo.

---

# Usar Otros Corpus

El sistema de entrenamiento funciona con **cualquier archivo de texto plano**.

Ejemplo:

```rust
let text = std::fs::read_to_string("tu_corpus.txt")?;
let tokenizer = Tokenizer::train(&text, vocab_size);

// continuar con el entrenamiento
```

Alternativas posibles:

- otras **obras literarias en dominio público**
- **Wikipedia**
- **documentación técnica**
- **repositorios de código**
- **texto propio**

---

# Datos No Incluidos

Este proyecto evita deliberadamente:

- libros modernos con copyright
- datos personales o privados
- scraping que viole términos de servicio
- corpus gigantes que requieran infraestructura compleja

El objetivo del proyecto es ser **educativo, transparente y reproducible**.