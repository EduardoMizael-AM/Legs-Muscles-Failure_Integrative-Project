import os
import glob
import pandas as pd

data_dir = "data"
csv_files = glob.glob(os.path.join(data_dir, "*", "*.csv"))

print(f"Archivos CSV encontrados en total: {len(csv_files)}\n")

for path in csv_files[:5]: # Muestra los primeros 5 archivos
    file_name = os.path.basename(path)
    folder = os.path.basename(os.path.dirname(path))
    
    try:
        df = pd.read_csv(path, nrows=2) # Lee solo las 2 primeras filas
        columns = list(df.columns)
    except Exception as e:
        columns = f"Error al leer: {e}"
        
    print(f"Carpeta: {folder}")
    print(f"Archivo: {file_name}")
    print(f"Columnas encontradas: {columns}")
    print("-" * 50)