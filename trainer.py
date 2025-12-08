import torch
#from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms
from src.data.dataloader import getDataSet, DataCollator, getDataTest
from omegaconf import  OmegaConf
from src.models.model import DigitsClassifier
import torch.nn as nn
import torch.optim as optim
from save_samples import testImage
config = OmegaConf.load("./configs/config.yaml")
collator = DataCollator()
train_dataset, val_dataset, test_dataset,total_size = getDataSet(config.data.root_dir)

loaders = {
    'train': DataLoader(
    dataset=train_dataset,
    batch_size=config.data.batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=collator),

    'val': DataLoader(
    dataset=val_dataset,
    batch_size=config.data.batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=collator),

    'test': DataLoader(
    dataset=test_dataset,
    batch_size=config.data.batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=collator),

}

    

#TODO: Define device to train on, loss function, optimizer.
#TODO: Initialize the model, and load configs(learning rate and epochs) 

epochs = config.trainer.epochs

learning_rate = config.trainer.learning_rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

loss_fn = nn.CrossEntropyLoss()

model = DigitsClassifier().to(device)  

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def checkDataLoader() -> None:
    print("Load data completed!!!")
    print(f"Total image: {total_size}")
    print(f"Training: {len(train_dataset)}")        
    print(f"Validation: {len(val_dataset)}")    
    print(f"Testing: {len(test_dataset)}")

    print("<---Training Starting--->")
    print (f"Using device: {device}")
    

def train() -> None:     
    print("<----Start Training----->")
    writer = SummaryWriter(log_dir="runs/my_experiment") #Tensor Board
    
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for i,(images, labels) in enumerate(loaders['train']):
            images = images.to(device)
            labels = labels.to(device)
            target_indices = torch.argmax(labels, dim=1)

            outputs = model(images)
            loss = loss_fn(outputs, target_indices)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step = epoch * len(loaders['train']) + i
            writer.add_scalar("Loss/train", loss.item(), global_step)
            running_loss += loss.item()
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i+1}/{len(loaders['train'])}], Loss: {(running_loss / 100):.4f}")
                running_loss = 0.0
        #Starting validation progress
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        total_val_steps = len(loaders['val'])
        print(f"--- Starting Validation for Epoch [{epoch+1}/{epochs}] ---")
        with torch.no_grad():
            for i, (images, labels) in enumerate(loaders['val']):
                images = images.to(device)
                labels = labels.to(device)
            
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                _, true_labels = torch.max(labels.data, 1)
            
                total += labels.size(0)
                correct += (predicted == true_labels).sum().item()

           
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Val Step [{i+1}/{total_val_steps}], Val Loss: {loss.item():.4f}")

        avg_val_loss = val_loss / total_val_steps
        accuracy = 100 * correct / total

        
        writer.add_scalar("Loss/validation", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/validation", accuracy, epoch)

        print(f"End of Epoch {epoch+1} -> Avg Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")
        print("--------------------------------------------------")
    torch.save(model.state_dict(), "model.pth")
    writer.close()

def test()-> None:
    print("<---Begin Test Process--->")
    model.load_state_dict(torch.load("model.pth"))
    test_loss = 0.0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loaders['test']:
            images, labels = images.to(device), labels.to(device)

            true_labels = torch.max(labels.data, 1)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            _, true_labels = torch.max(labels.data, 1)

            total += labels.size(0)
            correct += (predicted==true_labels).sum().item()
    avg_test_loss = test_loss / len(loaders['test'])
    test_acc = 100 * correct/total

    print(f"Test Loss: {avg_test_loss}, Accuaracy: {test_acc}")
            
test_data = getDataTest(root_dir=config.data.root_dir)

def run_test() -> None:
    label = testImage()
    with open('model.pth', 'rb') as f: 
        model.load_state_dict(torch.load(f))  
    plt.figure(figsize=(15, 3))
    for i in range(5):
        img = Image.open(f'{i}.jpg')

        plt.subplot(1, 5, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {label[i]}")
        plt.axis('off')

        img = Image.open(f'{i}.jpg')
        img_transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = img_transform(img).unsqueeze(0).to(device)
        output = model(img_tensor)
        pre_label = torch.argmax(output)
        print(f"Predicted label: {pre_label}")
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    checkDataLoader()
    train()
    test()
    #run_test()
    try:
        image, labels = next(iter(loaders['train']))
        print("\n---TEST DATLOADER---") #Data Loader test
        print(f"Size image batch: {image.shape}")
        print(f"Size labels batch: {labels.shape}")
        print(f"labels in batch: {labels[:5]}")
    except Exception as e:
        print(f"Error when get batch: {e}")