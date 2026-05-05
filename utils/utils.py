import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.models import CRNN
from utils.dataset import IAMDataset

def ocr_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    return torch.stack(images, 0), torch.cat(targets, 0), torch.tensor(target_lengths, dtype=torch.long)

def preparar_datasets_i_loaders(config):
    base_dir = "iam_dataset" # Ajusta això a la teva carpeta real
    gt_file = f"{base_dir}/linux_gt.txt" # El teu fitxer amb totes les dades

    # 1. Llegim i remenem
    with open(gt_file, "r", encoding="utf-8") as f:
        totes_les_linies = f.readlines()

    random.shuffle(totes_les_linies)

    # 2. Tallem la llista (80% - 10% - 10%)
    total = len(totes_les_linies)
    tall_train = int(0.8 * total)
    tall_val = int(0.9 * total) 

    linies_train = totes_les_linies[:tall_train]
    linies_val = totes_les_linies[tall_train:tall_val]
    linies_test = totes_les_linies[tall_val:]
    
    print(f"Distribució Dades: Train={len(linies_train)} | Val={len(linies_val)} | Test={len(linies_test)}")

    # 3. Transformacions (Train amb Data Augmentation, Val/Test nets)
    train_transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.RandomRotation(3),
        transforms.RandomAffine(0, shear=10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 4. Creem els Datasets
    # ATENCIÓ: Passem la llista de línies en comptes del nom del fitxer
    train_set = IAMDataset(lines=linies_train, img_dir=base_dir, transform=train_transform)
    
    # Agafem el diccionari que ha creat el train_set i l'apliquem a la resta
    char_to_idx = train_set.char_to_idx 
    
    val_set = IAMDataset(lines=linies_val, img_dir=base_dir, transform=test_transform, char_to_idx=char_to_idx)
    test_set = IAMDataset(lines=linies_test, img_dir=base_dir, transform=test_transform, char_to_idx=char_to_idx)

    # 5. Creem DataLoaders
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=ocr_collate_fn, num_workers=12, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, collate_fn=ocr_collate_fn, num_workers=12, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, collate_fn=ocr_collate_fn, num_workers=12, pin_memory=True)

    return train_loader, val_loader, test_loader, char_to_idx

def make(config, device="cuda"):
    # 1. Cridem la nova funció que ho fa tot
    train_loader, val_loader, test_loader, char_to_idx = preparar_datasets_i_loaders(config)

    # 2. Construïm el model
    model = CRNN(num_classes=len(char_to_idx) + 1).to(device)
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
        """Guarda el model quan la validation loss disminueix."""
        print(f'   💾 Validation loss decreased. Saving best model to {self.path}...')
        torch.save(model.state_dict(), self.path)