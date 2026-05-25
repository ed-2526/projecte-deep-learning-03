import os
import random
import glob
import re

# ==========================================
# 1. CONFIGURACIÓ
# ==========================================
BASE_DIR = "esposalles_dataset"
PARTS = ["IEHHR_training_part1", "IEHHR_training_part2", "IEHHR_training_part3"]

# Proporcions de la divisió (80% Train, 10% Val, 10% Test)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

def process_dataset():
    records = []

    # 1. Recopilar tots els registres
    for part in PARTS:
        part_path = os.path.join(BASE_DIR, part)
        if not os.path.exists(part_path):
            print(f"⚠️ Avís: La carpeta {part_path} no existeix. Comprova el nom.")
            continue

        for record_name in os.listdir(part_path):
            record_path = os.path.join(part_path, record_name)
            if os.path.isdir(record_path):
                records.append(record_path)

    print(f"📂 Total de registres (matrimonis) trobats: {len(records)}")

    # 2. Barrejar aleatòriament els registres
    random.seed(42)
    random.shuffle(records)

    # 3. Calcular els talls per a cada subconjunt
    n_records = len(records)
    train_cut = int(n_records * TRAIN_RATIO)
    val_cut = int(n_records * (TRAIN_RATIO + VAL_RATIO))

    train_records = records[:train_cut]
    val_records = records[train_cut:val_cut]
    test_records = records[val_cut:]

    print(f"📊 Repartiment de registres -> Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")

    # ==========================================
    # FUNCIONS AUXILIARS
    # ==========================================
    def extract_word_samples(record_path):
        samples = []
        lines_dir = os.path.join(record_path, "lines")
        words_dir = os.path.join(record_path, "words")

        if not os.path.exists(lines_dir) or not os.path.exists(words_dir):
            return samples

        transcription_files = glob.glob(os.path.join(lines_dir, "*transcription*"))
        if not transcription_files:
            return samples

        transcription_file = transcription_files[0]

        with open(transcription_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue

                line_id, text = line.split(":", 1)
                text = text.strip()
                words_in_text = text.split()

                word_images = glob.glob(os.path.join(words_dir, f"{line_id}_Word*"))

                def get_word_num(filepath):
                    match = re.search(r'Word(\d+)', filepath)
                    return int(match.group(1)) if match else -1
                
                word_images.sort(key=get_word_num)

                for img_path, word_text in zip(word_images, words_in_text):
                    # ==========================================
                    # 🌟 EL CANVI ÉS AQUÍ: 
                    # Retallem 'esposalles_dataset' de la ruta final
                    # ==========================================
                    rel_path = os.path.relpath(img_path, BASE_DIR)
                    
                    # Canviem barres de Windows (\) per barres Linux (/)
                    img_path_clean = rel_path.replace("\\", "/")
                    
                    samples.append(f"{img_path_clean}\t{word_text}")

        return samples

    def write_file(records_list, output_filename):
        total_words = 0
        with open(output_filename, 'w', encoding='utf-8') as out_f:
            for record_path in records_list:
                samples = extract_word_samples(record_path)
                for sample in samples:
                    out_f.write(sample + "\n")
                    total_words += 1
        print(f"✅ Fitxer '{output_filename}' creat amb èxit ({total_words} paraules).")

    # ==========================================
    # 4. EXECUCIÓ I GENERACIÓ
    # ==========================================
    print("-" * 50)
    write_file(train_records, "official_train.txt")
    write_file(val_records, "official_validation.txt")
    write_file(test_records, "official_test.txt")
    print("-" * 50)
    print("Procés finalitzat! Ja pots entrenar la CRNN.")

if __name__ == "__main__":
    process_dataset()