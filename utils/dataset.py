import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class OCRDataset(Dataset):
    # 1. CANVI: Canviem 'gt_file' per 'lines' (una llista de strings)
    def __init__(self, lines, img_dir, transform=None, char_to_idx=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = []
        
        # 2. CANVI: Eliminem el "with open..." i iterem directament sobre la llista
        for line in lines:
            parts = line.strip().split(maxsplit=1)
            if len(parts) >= 1:
                img_name = parts[0]
                text = parts[1] if len(parts) == 2 else ""
                self.data.append((img_name, text))
        
        # self.data = self.data[:64]

        # Creem o reaprofitem el diccionari de lletres (no canvia res)
        if char_to_idx is None:
            chars = sorted(list(set("".join([d[1] for d in self.data]))))
            self.char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}
        else:
            self.char_to_idx = char_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, text = self.data[idx]
        img_path = os.path.join(self.img_dir, img_name) # img_name ja inclou .png
        
        try:
            image = Image.open(img_path).convert('L')
        except:
            # Fallback intel·ligent per si falta alguna imatge (es queda igual)
            image = Image.new('L', (128, 32), color=255)
            text = ""

        if self.transform: 
            image = self.transform(image)
            
        target = [self.char_to_idx[c] for c in text if c in self.char_to_idx]
        return image, torch.tensor(target, dtype=torch.long), len(target)