import torch
import torch.nn as nn
import torch.nn.functional as F
#-------------------- Halobeat's architecture-----------------
# ATTENTION: THE COMMENTED ARCHITECHTURE IS NOT MINE (Halobeat)
# It's from another repo: https://github.com/RafayKhattak/Digit-Classification-Pytorch
# It's the same as in referenced repo but with a dropout in the fc layer
# class DigitsClassifier(nn.Module):
#     def __init__(self):
#             super(DigitsClassifier, self).__init__()
#             self.conv_layers = nn.Sequential(
#                 nn.Conv2d(1, 32, kernel_size=3),
#                 nn.ReLU(),
#                 nn.Conv2d(32, 64, kernel_size=3),
#                 nn.ReLU(),
#                 nn.Conv2d(64, 64, kernel_size=3),
#                 nn.ReLU()
#             )
#             self.fc_layers = nn.Sequential(
#                 nn.Flatten(),
#                 nn.Dropout(p=0.6),
#                 nn.Linear(64 * 22 * 22, 10)
#             )

#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = self.fc_layers(x)
#         return x
#-----------------------TranKhangTop's Architecture---------------
class DigitsClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(DigitsClassifier, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLu(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLu(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64*7*7, 128),
            nn.ReLu(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
