import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data.dataloader import getDataSet, DataCollator
from omegaconf import  OmegaConf
from src.models.model import DigitsClassifier
import torch.nn as nn
import torch.optim as optim
transform = transforms.ToTensor()
config = OmegaConf.load("./configs/config.yaml")
collator = DataCollator()
train_dataset, val_dataset, test_dataset,total_size = getDataSet(config.data.root_dir)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=config.data.batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=collator
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=config.data.batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=collator
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=config.data.batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=collator
)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print (f"Using device: {device}")

    model = DigitsClassifier(num_classes=10).to(device)

    learning_rate = config.trainer.learning_rate
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Start trainning...")
    model.train()
    epochs = config.trainer.epochs

    for epoch in range(epochs):
        running_loss = 0.0
        for i,(images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            target_indices = torch.argmax(labels, dim=1)

            outputs = model(images)
            loss = criterion(outputs, target_indices)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {(running_loss / 100):.4f}")
                running_loss = 0.0



def main(): #DataSet size ratio check
    print("Load data completed!!!")
    print(f"Total image: {total_size}")
    print(f"Training: {len(train_dataset)}")        
    print(f"Validation: {len(val_dataset)}")    
    print(f"Testing: {len(test_dataset)}")
    train()

if __name__ == '__main__':
    main()
    try:
        image, labels = next(iter(train_loader))
        print("\n---TEST DATLOADER---") #Data Loader test
        print(f"Size image batch: {image.shape}")
        print(f"Size labels batch: {labels.shape}")
        print(f"labels in batch: {labels[:5]}")
    except Exception as e:
        print(f"Error when get batch: {e}")
  
