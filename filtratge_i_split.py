import random
import os
from collections import defaultdict

# --- CONFIGURACIÓ ---
arxiu_forms = "iam_dataset/ascii/forms.txt"
arxiu_words = "iam_dataset/ascii/words.txt"
output_dir = "iam_dataset"

# Assegurem que la carpeta de sortida existeix
os.makedirs(output_dir, exist_ok=True)

print("1. Llegint el diccionari d'escriptors (forms.txt)...")
form_to_writer = {}
with open(arxiu_forms, "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("#") or not line.strip(): continue
        parts = line.split()
        form_to_writer[parts[0]] = parts[1]

print("2. Llegint les etiquetes i reconstruint rutes rutes (words.txt)...")
writer_to_data = defaultdict(list)
paraules_ok = 0
paraules_er = 0

with open(arxiu_words, "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("#") or not line.strip(): continue
        
        parts = line.split()
        word_id = parts[0]          # Ex: a01-000u-00-00
        segmentation = parts[1]     # Ex: ok o err
        
        if segmentation != "ok":
            paraules_er += 1
            continue
            
        label = " ".join(parts[8:])
        paraules_ok += 1
        
        # --- LÒGICA DE RECONSTRUCCIÓ DE RUTA ---
        # word_id: a01-000u-00-00
        id_parts = word_id.split('-')
        folder_1 = id_parts[0]                  # a01
        folder_2 = f"{id_parts[0]}-{id_parts[1]}" # a01-000u
        
        # Ruta final: words/a01/a01-000u/a01-000u-00-00.png
        ruta_oficial = f"words/{folder_1}/{folder_2}/{word_id}.png"
        
        # --- VINCULACIÓ AMB L'AUTOR ---
        form_id = folder_2
        if form_id in form_to_writer:
            writer_id = form_to_writer[form_id]
            linea_final = f"{ruta_oficial} {label}"
            writer_to_data[writer_id].append(linea_final)

print(f"  -> Paraules perfectes (ok) guardades: {paraules_ok}")
print(f"  -> Paraules defectuoses (err) descartades: {paraules_er}")

print("\n3. Repartint els autors (Writer-Independent Split)...")
llista_escriptors = list(writer_to_data.keys())
random.seed(42)
random.shuffle(llista_escriptors)

total_writers = len(llista_escriptors)
train_cut = int(total_writers * 0.8)
val_cut = int(total_writers * 0.9)

train_writers = llista_escriptors[:train_cut]
val_writers = llista_escriptors[train_cut:val_cut]
test_writers = llista_escriptors[val_cut:]

def guardar_fitxer(writers, nom_fitxer):
    total_linies = 0
    with open(nom_fitxer, "w", encoding="utf-8") as f:
        for writer in writers:
            for line in writer_to_data[writer]:
                f.write(line + "\n")
                total_linies += 1
    print(f"  - {nom_fitxer}: {total_linies} imatges.")

print("\n4. Generant els fitxers finals...")
guardar_fitxer(train_writers, os.path.join(output_dir, "official_train_gt.txt"))
guardar_fitxer(val_writers, os.path.join(output_dir, "official_val_gt.txt"))
guardar_fitxer(test_writers, os.path.join(output_dir, "official_test_gt.txt"))

print("\nFet! Ara els fitxers .txt tenen les rutes completes i correctes. 🚀")