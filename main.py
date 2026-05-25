import os
import random
import wandb
import argparse

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

        nom_net = f"LR:{config.learning_rate:.4f}_Batch:{config.batch_size}_Freeze:{config.freeze}"
        wandb.run.name = nom_net

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

    parser = argparse.ArgumentParser(description="Entrenament de la xarxa OCR")

    parser.add_argument('--epochs', type=int, default=100, help='Nombre total dèpoques')
    parser.add_argument('--batch_size', type=int, default=64, help='Mida del batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Taxa daprenentatge')
    parser.add_argument('--architecture', type=str, default="CRNN_Original", help='Model a utilitzar')
    parser.add_argument('--patience', type=int, default=5, help='Paciència per a l\'early stopping')
    parser.add_argument('--min_delta', type=float, default=0.015, help='Mínim canvi per a l\'early stopping')

    parser.add_argument('--freeze', action='store_true', help='Congela les capes de la CNN')
    parser.add_argument('--use_beam_search', action='store_true', help='Activa el Beam Search')
    parser.add_argument('--beam_width', type=int, default=5, help='Amplada del Beam Search')

    args = parser.parse_args()

    config = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dataset="IAM Dataset",
        architecture=args.architecture,
        classes=80, 
        freeze=args.freeze,
        use_beam_search=args.use_beam_search,
        beam_width=args.beam_width,
        patience=args.patience,
        min_delta=args.min_delta,
    )

    model = model_pipeline(config)
