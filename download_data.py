import urllib.request
import time

# Lista de las obras principales de Cervantes en español en Gutenberg
obras_cervantes = [
    {"titulo": "Don Quijote de la Mancha", "url": "https://www.gutenberg.org/cache/epub/2000/pg2000.txt"},
    {"titulo": "Novelas Ejemplares", "url": "https://www.gutenberg.org/cache/epub/61202/pg61202.txt"},
    {"titulo": "La Galatea", "url": "https://www.gutenberg.org/cache/epub/1445/pg1445.txt"},
    {"titulo": "Novelas y Teatro", "url": "https://www.gutenberg.org/cache/epub/15115/pg15115.txt"},
    {"titulo": "Entremeses", "url": "https://www.gutenberg.org/cache/epub/57955/pg57955.txt"}
]

nombre_archivo_final = "cervantes.txt"

print("Iniciando la construcción de las Obras Completas de Cervantes...\n")

# Abrimos el archivo final en modo "w" (escritura) una sola vez
with open(nombre_archivo_final, "w", encoding="utf-8") as archivo_final:
    
    # Recorremos la lista de libros uno por uno
    for obra in obras_cervantes:
        titulo = obra["titulo"]
        url = obra["url"]
        
        print(f"Descargando: {titulo}...")
        
        try:
            # Descargamos el contenido del libro actual
            with urllib.request.urlopen(url) as respuesta:
                contenido = respuesta.read().decode('utf-8')
            
            # Escribimos un encabezado bonito para separar los libros en el archivo
            archivo_final.write(f"\n\n{'='*50}\n")
            archivo_final.write(f"--- {titulo.upper()} ---\n")
            archivo_final.write(f"{'='*50}\n\n")
            
            # Escribimos todo el texto del libro
            archivo_final.write(contenido)
            print(f"  ✓ {titulo} añadido exitosamente.")
            
            # Pausa de 2 segundos para no saturar los servidores de Gutenberg (buena práctica)
            time.sleep(2)
            
        except Exception as e:
            print(f"  X Ups, hubo un error al descargar {titulo}: {e}")

print(f"\n¡Proceso terminado! Todo se ha guardado en '{nombre_archivo_final}'.")