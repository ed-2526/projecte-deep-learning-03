import os
import random

# ==========================================
# CONFIGURACIÓ
# ==========================================
DIR_IAM = "iam_dataset"
DIR_ESPOSALLES = "esposalles_dataset"
DIR_OUT = "hybrid_dataset"

SPLITS = ["official_train.txt", "official_validation.txt", "official_test.txt"]

# Creem la carpeta on guardarem els fitxers combinats
if not os.path.exists(DIR_OUT):
    os.makedirs(DIR_OUT)

def processar_linia(linia, prefix_carpeta):
    """
    Separa la ruta de la imatge del text, afegeix la carpeta arrel a la ruta,
    i ho torna a unir amb un tabulador per estandarditzar-ho.
    """
    # split(None, 1) talla només pel primer espai o tabulador que trobi
    parts = linia.strip().split(None, 1) 
    
    if len(parts) == 2:
        ruta_img, text = parts
        # Assegurem que les barres siguin genèriques i afegim la carpeta pare
        nova_ruta = f"{prefix_carpeta}/{ruta_img}".replace("\\", "/")
        return f"{nova_ruta}\t{text}"
    return None

def generar_dataset_hibrid():
    for split in SPLITS:
        path_iam = os.path.join(DIR_IAM, split)
        path_espo = os.path.join(DIR_ESPOSALLES, split)
        path_out = os.path.join(DIR_OUT, split)

        linies_combinades = []

        # 1. Llegir i adaptar les línies de l'IAM
        if os.path.exists(path_iam):
            with open(path_iam, 'r', encoding='utf-8') as f:
                for linia in f:
                    linia_processada = processar_linia(linia, DIR_IAM)
                    if linia_processada:
                        linies_combinades.append(linia_processada)
        else:
            print(f"⚠️ Avís: No s'ha trobat {path_iam}")

        # 2. Llegir i adaptar les línies de l'Esposalles
        if os.path.exists(path_espo):
            with open(path_espo, 'r', encoding='utf-8') as f:
                for linia in f:
                    linia_processada = processar_linia(linia, DIR_ESPOSALLES)
                    if linia_processada:
                        linies_combinades.append(linia_processada)
        else:
            print(f"⚠️ Avís: No s'ha trobat {path_espo}")

        # 3. Barrejar completament el conjunt (VITAL PER L'ENTRENAMENT)
        random.seed(42) # Llavor fixa per poder repetir l'experiment
        random.shuffle(linies_combinades)

        # 4. Guardar el nou fitxer híbrid
        with open(path_out, 'w', encoding='utf-8') as f:
            for linia in linies_combinades:
                f.write(linia + "\n")

        print(f"✅ Fitxer {split} creat! -> {len(linies_combinades)} paraules en total.")

if __name__ == "__main__":
    print("⏳ Generant dataset híbrid...")
    generar_dataset_hibrid()
    print("🎉 Procés finalitzat! Els teus fitxers estan a la carpeta 'hybrid_dataset'.")