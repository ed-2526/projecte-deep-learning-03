import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self, img_channels=1, num_classes=80, hidden_size=256):
        super(CRNN, self).__init__()
        
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
        self.rnn = nn.LSTM(input_size=1024, hidden_size=hidden_size, 
                           num_layers=2, bidirectional=True, batch_first=True)
        
        # Classificador final
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