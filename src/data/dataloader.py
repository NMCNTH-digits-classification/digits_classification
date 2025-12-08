import torch
from torch.utils.data import Dataset,random_split, ConcatDataset
from torchvision import datasets, transforms
import torch.nn.functional as F


class MnistDataset(Dataset):
    
    def __init__(self, root_dir, train, transform):
        self.root_dir = root_dir
        self.train = True
        self.transform = transform

        self.mnist_data = datasets.MNIST(
            root=root_dir, 
            train=train, 
            download=True,
            transform=None
        )

    def __len__(self):
        return len(self.mnist_data)

    def __getitem__(self, index):
        image, label = self.mnist_data[index]
        
        if self.transform:
            image = self.transform(image)
            
        label_onehot = F.one_hot(torch.tensor(label), num_classes=10).float()
 
        return image, label_onehot


class DataCollator():
    def __init__(self):
        pass

    def __call__(self, data):
        images = [item[0] for item in data]
        labels = [item[1] for item in data]
        batched_images = torch.stack(images, dim=0)
        batched_labels = torch.stack(labels, dim=0)
        return batched_images, batched_labels

def getDataSet(root_dir):
    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset= MnistDataset(root_dir, True, transform)
    total_size = len(full_dataset)
    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
    return train_dataset, val_dataset, test_dataset, total_size
def getDataTest(root_dir):
    test_data = MnistDataset(root_dir, False, None)
    return test_data
