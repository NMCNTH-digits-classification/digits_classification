import torch
import torch.nn as nn

class DigitsClassifier(nn.Module):
    def __init__(self):
            super(DigitsClassifier, self).__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3),
                nn.ReLU()
            )
            self.fc_layers = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=0.6),
                nn.Linear(64 * 22 * 22, 10)
            )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x