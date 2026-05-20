import random
import numpy as np
import jiwer
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.models import create_model
from utils.dataset import IAMDataset

def get_transforms(architecture_name):
    """
    Retorna les transformacions adequades segons si és la CNN original
    o una ResNet pre-entrenada amb ImageNet.
    """
    if architecture_name == "CRNN_Original":
        # ==========================================
        # TRANSFORMACIONS PER A LA CNN ORIGINAL (1 Canal)
        # ==========================================
        train_trans = transforms.Compose([
            transforms.Resize((32, 128)),    # Mantenir les imatges de la mateixa mida
            transforms.RandomRotation(3),    # Rotar la imatge [-x,x] graus
            transforms.RandomAffine(0, shear=10),    # Aplica una transformació d'inclinació (shear) d'un màxim de 10 graus del text
            transforms.ColorJitter(brightness=0.5, contrast=0.5), # DATA AUGMENTATION NOU: Canvi de contrast/brillantor
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)), # DATA AUGMENTATION NOU: Difuminat (simula mala qualitat)
            transforms.ToTensor(),           # Ho converteix a PyTorch entre 0 i 1
            transforms.Normalize((0.5,), (0.5,))    # Normalització dels valors dels píxels entre -1 i 1
        ])
        
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
        
        train_trans = transforms.Compose([
            transforms.Resize((32, 128)),
            transforms.Grayscale(num_output_channels=3), # Enganyem a RGB
            transforms.RandomRotation(3),
            transforms.RandomAffine(0, shear=10),
            transforms.ColorJitter(brightness=0.5, contrast=0.5), # DATA AUGMENTATION NOU: Canvi de contrast/brillantor
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)), # DATA AUGMENTATION NOU: Difuminat (simula mala qualitat)
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std) # Normalització d'ImageNet
        ])
        
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
    base_dir = "/dev/shm/edxnG03_dataset/iam_dataset" # Assegura't que el directori sigui el correcte on estan les imatges

    # Les rutes als 3 arxius que has generat amb l'script
    train_file = "iam_dataset/official_train_gt.txt"
    val_file   = "iam_dataset/official_val_gt.txt"
    test_file  = "iam_dataset/official_test_gt.txt"

    # 1. Llegim les línies exactament com estan als fitxers
    with open(train_file, "r", encoding="utf-8") as f:
        linies_train = f.readlines()
    with open(val_file, "r", encoding="utf-8") as f:
        linies_val = f.readlines()
    with open(test_file, "r", encoding="utf-8") as f:
        linies_test = f.readlines()
    
    # Ja NO fem random.shuffle() aquí. Ho fa el DataLoader a l'època d'entrenament!
    print(f"Distribució Dades (Sense Data Leakage!): Train={len(linies_train)} | Val={len(linies_val)} | Test={len(linies_test)}")

    # 2. Obtenim les transformacions
    train_transform, test_transform = get_transforms(config.architecture)

    # 3. Creem els Datasets
    # El Dataset de Train construeix el diccionari de caràcters per primer cop
    train_set = IAMDataset(lines=linies_train, img_dir=base_dir, transform=train_transform)
    
    # 🌟 El pas crític: Recuperem el diccionari del Train i el forcem a la resta
    char_to_idx = train_set.char_to_idx 
    
    val_set = IAMDataset(lines=linies_val, img_dir=base_dir, transform=test_transform, char_to_idx=char_to_idx)
    test_set = IAMDataset(lines=linies_test, img_dir=base_dir, transform=test_transform, char_to_idx=char_to_idx)

    # 4. Creem DataLoaders
    # IMPORTANT: pin_memory=True i num_workers > 0 acceleren molt la L40S
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

def decode_predictions(outputs, targets, target_lengths, char_to_idx):
    """Tradueix els tensors de la xarxa a text real (Strings)."""
    # 1. Creem un diccionari invers per passar de números a lletres
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    
    # 2. TRADUCCIÓ DE PREDICCIONS (CTC Greedy Decoder)
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
