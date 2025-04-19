from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomRotation, ColorJitter, RandomAffine
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import os
import numpy as np

import config

light_transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

strong_transform = Compose([
    Resize((224, 224)),
    RandomHorizontalFlip(p=0.5),
    RandomRotation(20),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    RandomAffine(15, scale=(0.8, 1.2)),
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

class ClasswiseAugmentedDataset(Dataset):
    def __init__(self, imagefolder_dataset, strong_transform, light_transform, threshold=30):
        self.samples = imagefolder_dataset.samples
        self.targets = [s[1] for s in self.samples]
        self.classes = imagefolder_dataset.classes
        self.class_to_idx = imagefolder_dataset.class_to_idx
        self.loader = imagefolder_dataset.loader
        self.threshold = threshold

        # 클래스별 transform 매핑 생성
        class_counts = np.bincount(self.targets)
        self.class_transform_map = {
            i: (strong_transform if count < self.threshold else light_transform)
            for i, count in enumerate(class_counts)
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = self.loader(path)
        transform = self.class_transform_map.get(label)
        image = transform(image)
        return image, label

# Sampler for class balancing
def get_balanced_sampler(dataset):
    labels = [label for _, label in dataset.samples]
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

def get_dataloaders(train_dir, test_dir, batch_size):
    # train
    raw_train_dataset = ImageFolder(root=train_dir)
    sampler = get_balanced_sampler(raw_train_dataset)

    train_dataset = ClasswiseAugmentedDataset(
        imagefolder_dataset=raw_train_dataset,
        strong_transform=strong_transform,
        light_transform=light_transform,
        threshold=config.AUG_THRESHOLD
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)

    # test
    test_dataset = ImageFolder(root=test_dir, transform=light_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return raw_train_dataset, train_loader, test_loader