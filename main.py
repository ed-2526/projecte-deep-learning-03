import os
import random
import wandb

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from train import *
from test import *
from utils.utils import *
from tqdm.auto import tqdm

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def model_pipeline(cfg:dict):
    # tell wandb to get started
    with wandb.init(project="pytorch-demo", config=cfg):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # 1. CANVI: Ara desempaquetem 7 variables, incloent el val_loader!
        model, train_loader, val_loader, test_loader, criterion, optimizer, char_to_idx = make(config, device=device)
    
        # Li preguntem al model on té guardats els seus pesos
        dispositiu_real = next(model.parameters()).device
        print(f"El model s'està executant a: {dispositiu_real}")
        
        # Actualitzem WandB amb el nombre real de classes
        wandb.config.update({"classes": len(char_to_idx) + 1}, allow_val_change=True)

        # 2. CANVI: Passem el val_loader a la funció de train
        train(model, train_loader, val_loader, criterion, optimizer, config, char_to_idx, device=device)

        # 3. El test es queda exactament igual, només s'executa al final de tot
        test(model, test_loader, char_to_idx, device=device)

    return model

if __name__ == "__main__":
    wandb.login()

    config = dict(
        epochs=100,
        batch_size=512,      # Mida recomanada per a OCR
        learning_rate=2e-3,
        dataset="IAM Dataset",
        architecture="ResNet18",
        classes=80,           # Valor aproximat, el make() el sobreescriurà amb el real
        freeze=True
    )

    model = model_pipeline(config)