import random
import numpy as np
import jiwer
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.models import create_model
from utils.dataset import OCRDataset

def get_transforms(config):
    """
    Construeix les transformacions dinàmicament segons els paràmetres de la terminal.
    """
    if config.architecture == "CRNN_Original":
        # ==========================================
        # TRANSFORMACIONS PER A LA CNN ORIGINAL (1 Canal)
        # ==========================================
        # 1. Transformacions base obligatòriesSSS
        train_trans_list = [transforms.Resize((32, 128))]
        
        # 2. Augment de dades dinàmic
        if not config.no_random_rotation:
            train_trans_list.append(transforms.RandomRotation(3))
            
        if not config.no_affine:
            train_trans_list.append(transforms.RandomAffine(0, shear=10))
            
        if config.activate_color_jitter:
            # S'aplica només a un 20% (p=0.2) de les imatges amb una força del 15%
            train_trans_list.append(
                transforms.RandomApply([transforms.ColorJitter(brightness=0.15, contrast=0.15)], p=0.2)
            )
            
        if config.activate_gaussian_blur:
            # S'aplica només a un 20% (p=0.2) amb un difuminat gairebé imperceptible
            train_trans_list.append(
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5))], p=0.2)
            )
            
        # 3. Finalització obligatòria (Tensor i Normalització)
        train_trans_list.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        train_trans = transforms.Compose(train_trans_list)
        
        test_trans = transforms.Compose([
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
    else:
        # ==========================================
        # TRANSFORMACIONS PER A LES RESNET (3 Canals + ImageNet)
        # ==========================================
        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)
        
        # 1. Transformacions base obligatòries
        train_trans_list = [
            transforms.Resize((32, 128)),
            transforms.Grayscale(num_output_channels=3)
        ]
        
        # 2. Augment de dades dinàmic
        if not config.no_random_rotation:
            train_trans_list.append(transforms.RandomRotation(3))
            
        if not config.no_affine:
            train_trans_list.append(transforms.RandomAffine(0, shear=10))
            
        if config.activate_color_jitter:
            # S'aplica només a un 20% (p=0.2) de les imatges amb una força del 15%
            train_trans_list.append(
                transforms.RandomApply([transforms.ColorJitter(brightness=0.15, contrast=0.15)], p=0.2)
            )
            
        if config.activate_gaussian_blur:
            # S'aplica només a un 20% (p=0.2) amb un difuminat gairebé imperceptible
            train_trans_list.append(
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5))], p=0.2)
            )
            
        # 3. Finalització obligatòria
        train_trans_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ])
        
        train_trans = transforms.Compose(train_trans_list)
        
        test_trans = transforms.Compose([
            transforms.Resize((32, 128)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ])
        
    return train_trans, test_trans

def ocr_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    return torch.stack(images, 0), torch.cat(targets, 0), torch.tensor(target_lengths, dtype=torch.long)

def preparar_datasets_i_loaders(config):
    root_dir = "/dev/shm/edxnG03_dataset" 

    if config.dataset == "iam":
        base_dir = f"{root_dir}/iam_dataset"
        txt_folder = "iam_dataset"
    elif config.dataset == "esposalles":
        base_dir = f"{root_dir}/esposalles_dataset"
        txt_folder = "esposalles_dataset"
    elif config.dataset == "hybrid":
        base_dir = root_dir  
        txt_folder = "hybrid_dataset"
    else:
        raise ValueError(f"Dataset no reconegut: {config.dataset}")

    train_file = f"{txt_folder}/official_train.txt"
    val_file   = f"{txt_folder}/official_validation.txt"
    test_file  = f"{txt_folder}/official_test.txt"

    with open(train_file, "r", encoding="utf-8") as f:
        linies_train = f.readlines()
    with open(val_file, "r", encoding="utf-8") as f:
        linies_val = f.readlines()
    with open(test_file, "r", encoding="utf-8") as f:
        linies_test = f.readlines()
    
    print(f"📂 Carregant Dataset: {config.dataset.upper()}")
    print(f"📊 Distribució Dades: Train={len(linies_train)} | Val={len(linies_val)} | Test={len(linies_test)}")

    train_transform, test_transform = get_transforms(config)

    train_set = OCRDataset(lines=linies_train, img_dir=base_dir, transform=train_transform)
    char_to_idx = train_set.char_to_idx 
    
    val_set = OCRDataset(lines=linies_val, img_dir=base_dir, transform=test_transform, char_to_idx=char_to_idx)
    test_set = OCRDataset(lines=linies_test, img_dir=base_dir, transform=test_transform, char_to_idx=char_to_idx)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=ocr_collate_fn, num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, collate_fn=ocr_collate_fn, num_workers=8, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, collate_fn=ocr_collate_fn, num_workers=8, pin_memory=True)

    return train_loader, val_loader, test_loader, char_to_idx

def make(config, device="cuda"):
    # 1. Cridem la nova funció que ho fa tot
    train_loader, val_loader, test_loader, char_to_idx = preparar_datasets_i_loaders(config)

    # 2. Construïm el model
    model = create_model(config.architecture, num_classes=len(char_to_idx) + 1, freeze=config.freeze).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 3. Retornem TOTS els loaders!
    return model, train_loader, val_loader, test_loader, criterion, optimizer, char_to_idx


class EarlyStopping:
    """Atura l'entrenament si la validation loss no millora després d'una certa paciència."""
    def __init__(self, patience=5, min_delta=0, path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'   ⚠️ EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Guarda el (millor) model quan la validation loss disminueix."""
        print(f'   💾 Validation loss decreased. Saving best model to {self.path}...')
        torch.save(model.state_dict(), self.path)


# ==========================================
# CTC BEAM SEARCH DECODER (sense dependències externes)
# ==========================================
def ctc_beam_search_decode(log_probs_batch, idx_to_char, beam_width=5):
    """
    CTC Beam Search Decoder implementat en Python pur (sense ctcdecode).

    Per què és millor que el Greedy Decoder per reduir el WER?
    ──────────────────────────────────────────────────────────
    El Greedy agafa sempre la lletra amb probabilitat màxima a cada pas temporal.
    Si en un moment la 'l' de 'Hello' cau per sota del blank, el greedy obté "Helo".

    El Beam Search manté les 'beam_width' millors hipòtesis actives i les combina.
    Si globalment "Hello" és més probable que "Helo" (sumant totes les alineacions
    CTC possibles), el Beam Search ho detecta i retorna "Hello".

    Args:
        log_probs_batch : tensor [seq_len, batch, num_classes] (log-probabilitats)
        idx_to_char     : dict {int → str}, el vocabulari invers
        beam_width      : nombre de hipòtesis actives (5 és un bon equilibri)

    Returns:
        list[str] : text decodificat per a cada element del batch
    """
    # Convertim de log-probabilitats a probabilitats normals
    probs_batch = torch.exp(log_probs_batch).cpu()  # [seq_len, batch, num_classes]
    seq_len, batch_size, num_classes = probs_batch.shape

    decoded_batch = []

    for b in range(batch_size):
        probs = probs_batch[:, b, :].numpy()  # [seq_len, num_classes]

        # Cada hipòtesi: (prefix_tuple, prob_blank, prob_non_blank)
        # prefix_tuple: la seqüència de caràcters acumulada (sense blanks ni repetits CTC)
        beams = [((), 1.0, 0.0)]  # (prefix, Pb, Pnb)

        for t in range(seq_len):
            p_t = probs[t]        # [num_classes]
            new_beams = {}

            for prefix, Pb, Pnb in beams:
                P_total = Pb + Pnb

                # Opció 1: afegir el token BLANK (índex 0) → el prefix no canvia
                key = prefix
                if key not in new_beams:
                    new_beams[key] = [0.0, 0.0]
                new_beams[key][0] += P_total * p_t[0]  # actualitzem Pb

                # Opció 2: afegir un caràcter no-blank
                for c in range(1, num_classes):
                    p_c = p_t[c]
                    new_prefix = prefix + (c,)

                    # Si repetim el mateix caràcter consecutiu, cal que vingui d'un blank
                    if len(prefix) > 0 and prefix[-1] == c:
                        new_Pnb = Pb * p_c
                    else:
                        new_Pnb = P_total * p_c

                    if new_prefix not in new_beams:
                        new_beams[new_prefix] = [0.0, 0.0]
                    new_beams[new_prefix][1] += new_Pnb  # actualitzem Pnb

            # Ordenem per probabilitat total i mantenim els beam_width millors
            beams_sorted = sorted(
                new_beams.items(),
                key=lambda x: x[1][0] + x[1][1],
                reverse=True
            )
            beams = [(pfx, pb, pnb) for pfx, (pb, pnb) in beams_sorted[:beam_width]]

        # Millor hipòtesi: la primera de la llista (la de major probabilitat)
        best_prefix, _, _ = beams[0]
        decoded_text = "".join([idx_to_char.get(c, '') for c in best_prefix])
        decoded_batch.append(decoded_text)

    return decoded_batch


def decode_predictions(outputs, targets, target_lengths, char_to_idx, use_beam_search=False, beam_width=5):
    """
    Tradueix els tensors de la xarxa a text real (Strings).

    Args:
        outputs         : [seq_len, batch, num_classes] (log-softmax de la xarxa)
        targets         : tensor concatenat del ground-truth
        target_lengths  : longituds de cada seqüència target
        char_to_idx     : diccionari lletra → índex
        use_beam_search : si True, usa CTC Beam Search en lloc de Greedy (millor WER)
        beam_width      : nombre de hipòtesis actives del Beam Search (default: 5)
    """
    # 1. Creem un diccionari invers per passar de números a lletres
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    
    # 2. TRADUCCIÓ DE PREDICCIONS
    if use_beam_search:
        # ── CTC Beam Search Decoder (millor qualitat, redueix WER) ──────
        pred_strings = ctc_beam_search_decode(outputs, idx_to_char, beam_width=beam_width)
    else:
        # ── CTC Greedy Decoder (ràpid, per a entrenament) ───────────────
        # outputs fa [seq_len, batch, classes]. Ho girem i agafem la lletra amb més %
        _, max_indices = torch.max(outputs.transpose(0, 1), 2) 
        
        pred_strings = []
        for i in range(max_indices.size(0)):
            raw_pred = max_indices[i].tolist()
            decoded_str = []
            prev_char = -1
            for c in raw_pred:
                if c != prev_char: # Elimina duplicats seguits (a_a -> a)
                    if c != 0:     # 0 és el token 'blank', l'ignorem
                        decoded_str.append(idx_to_char.get(c, ''))
                prev_char = c
            pred_strings.append("".join(decoded_str))
        
    # 3. TRADUCCIÓ DEL GROUND TRUTH (Solucions reals)
    split_targets = torch.split(targets, target_lengths.tolist())
    true_strings = []
    for t in split_targets:
        true_strings.append("".join([idx_to_char.get(c.item(), '') for c in t]))
        
    return true_strings, pred_strings

def calculate_metrics(true_strings, pred_strings):
    """Calcula CER i WER amb protecció per a strings buits."""
    # Evitem que 'jiwer' peti si la xarxa prediu un text completament buit
    clean_trues = [t if len(t.strip()) > 0 else " " for t in true_strings]
    clean_preds = [p if len(p.strip()) > 0 else " " for p in pred_strings]
    
    cer = jiwer.cer(clean_trues, clean_preds)
    wer = jiwer.wer(clean_trues, clean_preds)
    return cer, wer
