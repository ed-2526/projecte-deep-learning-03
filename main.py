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

def model_pipeline(cfg:dict) -> None:
    # tell wandb to get started
    with wandb.init(project="pytorch-demo", config=cfg):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      # make the model, data, and optimization problem
      model, train_loader, test_loader, criterion, optimizer = make(config,device=device)

      # and use them to train the model
      train(model, train_loader, criterion, optimizer, config,device=device)

      # and test its final performance
      test(model, test_loader,device=device)

    return model

if __name__ == "__main__":
    wandb.login()

    # config = dict(
    #     epochs=15,
    #     batch_size=32,      # Mida recomanada per a OCR
    #     learning_rate=1e-3,
    #     dataset="IAM Dataset",
    #     architecture="CRNN",
    #     classes=80          # Valor aproximat, el make() el sobreescriurà amb el real
    # )
    config = dict(
        epochs=1,            # <-- Només 1 època per provar
        batch_size=8,        # <-- Un batch petit de 8 imatges (gasta poca memòria)
        learning_rate=1e-3,
        dataset="IAM Dataset",
        architecture="CRNN"
    )

    model = model_pipeline(config)

