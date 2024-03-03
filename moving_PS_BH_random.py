import os
import shutil
import random

# Ruta de la carpeta destino
dest_folder = "/export/data/giuliom/jordit/data2"

# Ruta del archivo que contiene las rutas de los archivos
file_paths_file = "/export/data/giuliom/jordit/output_PS_BH.txt"

# Leer las rutas desde el archivo
with open(file_paths_file, "r") as file:
    file_paths = file.readlines()

# Eliminar los saltos de línea al final de cada ruta
file_paths = [path.strip() for path in file_paths]

# Obtener el nombre de la fuente (viene después de la carpeta BH)
source_name = file_paths[0].split("/BH/")[1].split("/")[0]

# Solicitar al usuario la cantidad de archivos que desea copiar para cada fuente
while True:
    try:
        count_per_source = int(input(f"\nCantidad de archivos a copiar para cada fuente: "))
        if count_per_source >= 0:
            break
        else:
            print("Ingrese un número no negativo.")
    except ValueError:
        print("Ingrese un número válido.")

# Crear un diccionario para almacenar la cantidad de archivos copiados para cada fuente
files_copied_per_source = {source: 0 for source in set(file_path.split("/BH/")[1].split("/")[0] for file_path in file_paths)}

# Iterar sobre las rutas de los archivos
for source_name in set(file_path.split("/BH/")[1].split("/")[0] for file_path in file_paths):
    # Obtener los archivos disponibles para esta fuente
    available_files = [file_path for file_path in file_paths if source_name in file_path]

    # Calcular la cantidad real a copiar para esta fuente
    count_for_source = min(count_per_source, len(available_files))

    # Seleccionar aleatoriamente los archivos para esta fuente
    files_for_source = random.sample(available_files, count_for_source)

    # Iterar sobre los archivos seleccionados y copiarlos
    for file_path in files_for_source:
        # Construir la ruta completa de destino manteniendo la estructura de carpetas
        dest_path = os.path.join(dest_folder, 'BH', source_name, file_path.split("/BH/")[1].split("/")[1], 'pca', os.path.basename(file_path))

        # Verificar si la estructura de carpetas existe y copiar el archivo
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)  # Crear directorios intermedios si no existen
        shutil.copy2(file_path, dest_path)
        print(f"Archivo {os.path.basename(file_path)} copiado exitosamente a {dest_path}.")

        # Actualizar el contador de archivos copiados para esta fuente
        files_copied_per_source[source_name] += 1


# Informar que el proceso ha sido completado
print("\nProceso completado.\n")
