import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import torch.nn.functional as F

class MnistDataset(Dataset):
    
    def __init__(self, root_dir, train, transform=None):
        self.mnist_data = datasets.MNIST(
            root=root_dir, 
            train=train, 
            download=True
        )
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.mnist_data)

    def __getitem__(self, index):
        image, label = self.mnist_data[index]
        
        if self.transform:
            image = self.transform(image)
            
        label_onehot = F.one_hot(torch.tensor(label), num_classes=10).float()
        #I have no idea how this works, but it works.
        return image, label_onehot


class DataCollator():
    pass
#Check Organize_source_Code.pdf on FIT moodle before write DataCollator.