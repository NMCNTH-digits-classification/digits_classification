import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitsClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(DigitsClassifier, self).__init__()

        #Layer 1: In(1, 28, 28) -> Out(32, 28, 28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)

        #Layer 2: In(32, 28, 28) -> Out(64, 14, 14)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        #Reduce image size by half (28 -> 14) (14 -> 7)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        

        self.flatten_size = 64 * 7 * 7
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
       
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(-1, self.flatten_size) #convert to vector

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return x

