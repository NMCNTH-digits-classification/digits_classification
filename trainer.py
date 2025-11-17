import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import transforms
from src.data.dataloader import MnistDataset, DataCollator
from omegaconf import  OmegaConf
transform = transforms.ToTensor()
config = OmegaConf.load("./configs/config.yaml")
full_train_dataset= MnistDataset(config.data.root_dir, True, transform)
full_test_dataset= MnistDataset(config.data.root_dir, False, transform)
collator = DataCollator()
full_dataset = ConcatDataset([full_train_dataset, full_test_dataset])

total_size = len(full_dataset)
train_size = int(0.6 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

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

def main():
    print("Load data completed!!!")
    print(f"Total image: {total_size}")
    print(f"Training: {len(train_dataset)}")        
    print(f"Validation: {len(val_dataset)}")    
    print(f"Testing: {len(test_dataset)}")
#
if __name__ == '__main__':
    main()
    try:
        image, labels = next(iter(train_loader))
        print("\n---TEST 1 BATCH---")
        print(f"Size image batch: {image.shape}")

        print(f"Size labels batch: {labels.shape}")
        print(f"labels in batch: {labels[:5]}")
    except Exception as e:
        print(f"Error when get batch: {e}")
#This one is for testing to see if the dataloader works correctly or not.  
