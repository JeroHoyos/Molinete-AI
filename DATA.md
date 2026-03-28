# Datos de Entrenamiento: Obras de Cervantes

**Autor:** Miguel de Cervantes Saavedra | **Fuente:** Project Gutenberg  
**Licencia:** Dominio Público | **Tamaño:** ~5–7 MB  
**Contenido:** *Don Quijote*, *Novelas Ejemplares*, *La Galatea*, *Novelas y Teatro*, *Entremeses*.

---

## 1. Obtener los Datos

Para facilitar la descarga, usa el script `download_data.py` incluido en el proyecto. Al ejecutarlo, descargará automáticamente las obras desde Project Gutenberg y las unirá en un único archivo de texto llamado `cervantes.txt`.

## 2. Preprocesamiento

El procesamiento es mínimo y directo:
1. El script agrupa los textos separándolos visualmente.
2. (Opcional) Puedes borrar a mano los textos legales de Gutenberg al inicio y final.
3. El tokenizador se entrena directamente sobre este archivo final.

## 3. ¿Por qué Cervantes?

* **Libre de derechos:** Al ser obras del Siglo de Oro, están 100% en dominio público.
* **Tamaño ideal:** Perfecto para experimentar y entrenar modelos en computadoras personales.
* **Riqueza léxica:** Combina narrativa, poesía y teatro, enseñando al modelo estructuras de lenguaje variadas y complejas.
* **Estilo inconfundible:** Es muy fácil y divertido evaluar el modelo observando cómo empieza a imitar el castellano antiguo.

## 4. Usar otros textos y Ética

Si prefieres otro corpus, simplemente reemplaza `cervantes.txt` por cualquier otro archivo de texto plano. 

Este proyecto está diseñado con fines **educativos y reproducibles**, por lo que evitamos deliberadamente el uso de libros con copyright, datos privados extraídos de internet o corpus gigantescos que requieran supercomputadoras.