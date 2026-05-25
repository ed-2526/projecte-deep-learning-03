import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ==========================================
# 1. EL TEU MODEL ORIGINAL
# ==========================================
class CRNN_Original(nn.Module):
    # NOTA: Hem canviat img_channels=3 per defecte perquè puguis 
    # fer servir el truc de la imatge en color en tots els models sense canviar res més.
    def __init__(self, img_channels=1, num_classes=80, hidden_size=256):
        super(CRNN_Original, self).__init__()
        
        # Extracció de característiques (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)) 
        )
        
        # Processament seqüencial (RNN - BiLSTM)
        # L'alçada es redueix de 32 a 4. 256 canals * 4 = 1024 features d'entrada.
        # Per poder preveure patrons entre lletres. p.ex: 2 consonants després anira una vocal
        self.rnn = nn.LSTM(input_size=1024, hidden_size=hidden_size, 
                           num_layers=2, bidirectional=True, batch_first=True)
        
        # Classificador final
        # Per relacionar-ho amb la loss
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        conv = self.cnn(x) # [batch, 256, 4, seq_len]
        batch, channels, height, width = conv.size()
        conv = conv.view(batch, channels * height, width)
        conv = conv.permute(0, 2, 1) # [batch, seq_len, features]
        
        output, _ = self.rnn(conv)
        output = self.fc(output) # [batch, seq_len, num_classes]
        
        # La CTCLoss espera [seq_len, batch, num_classes]
        return F.log_softmax(output.permute(1, 0, 2), dim=2)


# ==========================================
# 2. RESNET 18 (Ràpida i Molt Intel·ligent)
# Xarxa amb els pesos ja inicialitzats, serveix per obtenir característiques, CNN precalculat (?)
# ==========================================
class ResNet18_CRNN(nn.Module):
    def __init__(self, num_classes=80, hidden_size=256, freeze_resnet=False):
        super(ResNet18_CRNN, self).__init__()
        
        resnet = models.resnet18(weights='DEFAULT')
        
        # Tallem a la layer2 per mantenir seq_len = 32 i alçada = 8.
        # Dimensions: (3x32x128) -> conv1(64x16x64) -> layer1(64x16x64) -> layer2(128x8x32)
        self.cnn = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            # resnet.maxpool, <-- ELIMINAT per no aixafar la longitud!
            resnet.layer1,
            resnet.layer2
        )
        
        # ==========================================
        # 🌟 EL CODI PER CONGELAR LES CAPES
        # ==========================================
        if freeze_resnet:
            for param in self.cnn.parameters():
                param.requires_grad = False
            print("❄️ ResNet18 CONGELADA (Feature Extraction)")
        else:
            print("🔥 ResNet18 OBERTA (Fine-Tuning)")
        
        # ==========================================

        # Matemàtiques: 128 canals * 8 d'alçada = 1024 features
        self.rnn = nn.LSTM(input_size=1024, hidden_size=hidden_size, 
                           num_layers=2, bidirectional=True, batch_first=True)
        
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        batch, channels, height, width = conv.size()
        conv = conv.view(batch, channels * height, width)
        conv = conv.permute(0, 2, 1)
        
        output, _ = self.rnn(conv)
        output = self.fc(output)
        
        return F.log_softmax(output.permute(1, 0, 2), dim=2)


# ==========================================
# 3. RESNET 34 (Més profunda, ideal per Fine-Tuning pesat)
# ==========================================
class ResNet34_CRNN(nn.Module):
    def __init__(self, num_classes=80, hidden_size=256, freeze_resnet=False):
        super(ResNet34_CRNN, self).__init__()
        
        resnet = models.resnet34(weights='DEFAULT')
        
        self.cnn = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            # resnet.maxpool, <-- ELIMINAT
            resnet.layer1,
            resnet.layer2
        )
        
        # ==========================================
        # 🌟 EL CODI PER CONGELAR LES CAPES
        # ==========================================
        if freeze_resnet:
            for param in self.cnn.parameters():
                param.requires_grad = False
            print("❄️ ResNet18 CONGELADA (Feature Extraction)")
        else:
            print("🔥 ResNet18 OBERTA (Fine-Tuning)")
        
        # ==========================================

        self.rnn = nn.LSTM(input_size=1024, hidden_size=hidden_size, 
                           num_layers=2, bidirectional=True, batch_first=True)
        
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        batch, channels, height, width = conv.size()
        conv = conv.view(batch, channels * height, width)
        conv = conv.permute(0, 2, 1)
        
        output, _ = self.rnn(conv)
        output = self.fc(output)
        
        return F.log_softmax(output.permute(1, 0, 2), dim=2)


# ==========================================
# 4. FUNCIÓ SELECTORA (Per cridar des de main.py)
# ==========================================
def create_model(architecture_name, num_classes, freeze=False):
    """
    Funció 'Factory' que retorna el model demanat pel WandB Sweep.
    """
    if architecture_name == "CRNN_Original":
        return CRNN_Original(num_classes=num_classes)
    elif architecture_name == "ResNet18":
        return ResNet18_CRNN(num_classes=num_classes, freeze_resnet=freeze)
    elif architecture_name == "ResNet34":
        return ResNet34_CRNN(num_classes=num_classes, freeze_resnet=freeze)
    else:
        raise ValueError(f"ERROR: L'arquitectura '{architecture_name}' no existeix!")
