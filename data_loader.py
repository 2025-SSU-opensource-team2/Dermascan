from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomRotation, ColorJitter, RandomAffine
from torch.utils.data import Dataset, DataLoader

original_transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

augmentation_transform = Compose([
    Resize((224, 224)),
    RandomHorizontalFlip(p=0.5),
    RandomRotation(20),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    RandomAffine(15, scale=(0.8, 1.2)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class AugmentedDataset(Dataset):
    def __init__(self, base_dataset, transform=None, augmentations=None):
        self.base_dataset = base_dataset
        self.transform = transform
        self.augmentations = augmentations

    def __len__(self):
        return 2 * len(self.base_dataset)

    def __getitem__(self, idx):
        if idx < len(self.base_dataset):
            img, label = self.base_dataset[idx]
            if self.transform:
                img = self.transform(img)
        else:
            img, label = self.base_dataset[idx - len(self.base_dataset)]
            if self.augmentations:
                img = self.augmentations(img)
        return img, label

def get_dataloaders(train_dir, test_dir, batch_size):
    train_dataset = ImageFolder(root=train_dir)
    augmented_train_dataset = AugmentedDataset(
        base_dataset=train_dataset,
        transform=original_transform,
        augmentations=augmentation_transform
    )
    train_loader = DataLoader(augmented_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_transform = original_transform
    test_dataset = ImageFolder(root=test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_dataset, train_loader, test_loader
