import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.models import CRNN
from utils.dataset import IAMDataset

def ocr_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    return torch.stack(images, 0), torch.cat(targets, 0), torch.tensor(target_lengths, dtype=torch.long)

def get_data(config, train=True, char_to_idx=None):
    base_dir = "iam_dataset"
    gt_file = f"{base_dir}/train_gt.txt" if train else f"{base_dir}/val_gt.txt"
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return IAMDataset(gt_file, base_dir, transform, char_to_idx)

def make(config, device="cuda"):
    train_set = get_data(config, train=True)
    char_to_idx = train_set.char_to_idx
    val_set = get_data(config, train=False, char_to_idx=char_to_idx)
    
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=ocr_collate_fn)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, collate_fn=ocr_collate_fn)

    model = CRNN(num_classes=len(char_to_idx) + 1).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, val_loader, criterion, optimizer, char_to_idx