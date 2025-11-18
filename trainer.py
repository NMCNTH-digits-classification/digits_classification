import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data.dataloader import getDataSet, DataCollator
from omegaconf import  OmegaConf
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

def main(): #DataSet size ratio check
    print("Load data completed!!!")
    print(f"Total image: {total_size}")
    print(f"Training: {len(train_dataset)}")        
    print(f"Validation: {len(val_dataset)}")    
    print(f"Testing: {len(test_dataset)}")

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
  
