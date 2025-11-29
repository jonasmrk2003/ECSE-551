from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
from PIL import Image
import pandas as pd
import torch
import time

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir 
        self.transform = transform

        # Map string labels to integers
        self.classes = sorted(self.data['label'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.data['label_idx'] = self.data['label'].map(self.class_to_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = f"{self.img_dir}/{self.data.iloc[idx, 0]}"
        image = Image.open(img_name).convert('RGB')
        label = self.data.iloc[idx]['label_idx']

        if self.transform:
            image = self.transform(image)

        return image, label
    
def split_dataset(dataset, train_ratio=0.7, seed=None):
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size], generator=generator)

from torchvision import transforms

def get_cifar10_transforms(mean=None, std=None, augment=True):
    """
    Returns train and test transforms for CIFAR-10.

    Args:
        mean (list or tuple, optional): Mean for normalization. Defaults to CIFAR-10 mean.
        std (list or tuple, optional): Std for normalization. Defaults to CIFAR-10 std.
        augment (bool): Whether to apply training augmentations.

    Returns:
        train_transform, test_transform: torchvision.transforms.Compose objects
    """
    # Default CIFAR-10 mean and std
    if mean is None:
        mean = [0.4914, 0.4822, 0.4465]
    if std is None:
        std = [0.2470, 0.2435, 0.2616]

    # Training transform (with optional augmentation)
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.5)
        ])

    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # resize
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    # Test/validation transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # resize
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, test_transform

def load_data(
    dataset_class,
    csv_file,
    img_dir,
    batch_size,
    train_ratio,
    seed=42,
    augment=True
):
    """
    Load a dataset, split into train/test, apply transforms, and return DataLoaders.

    Args:
        dataset_class: Custom Dataset class
        csv_file (str): Path to CSV labels file
        img_dir (str): Path to images directory
        batch_size (int): Batch size for DataLoaders
        train_ratio (float): Fraction of data for training
        seed (int): Random seed for reproducibility
        augment (bool): Whether to apply training augmentations

    Returns:
        train_loader, test_loader: DataLoaders for training and test sets
    """
    # Load full dataset
    full_dataset = dataset_class(csv_file=csv_file, img_dir=img_dir, transform=None)

    # Split dataset
    train_dataset, test_dataset = split_dataset(full_dataset, train_ratio=train_ratio, seed=seed)

    # Assign transforms
    train_transform, test_transform = get_cifar10_transforms(augment=augment)
    train_dataset.dataset.transform = train_transform
    test_dataset.dataset.transform = test_transform

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs):
    """
    Train a PyTorch model and track train/test accuracy.

    Returns:
        train_accs, test_accs, model_weights: Lists of accuracy per epoch and trained weights
    """
    train_accs, test_accs = [], []
    global_step = 0

    for epoch in range(num_epochs):
        start_time = time.time()  # start timer
        # ---- Training ----
        model.train()
        total_loss, correct, total = 0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            global_step += 1
            total_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        end_time = time.time()  # end timer
        epoch_time = end_time - start_time

        train_loss = total_loss / total
        train_acc = correct / total
        train_accs.append(train_acc)

        # ---- Evaluation ----
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = test_correct / test_total
        test_accs.append(test_acc)

        print(
            f"Epoch {epoch+1}/{num_epochs} "
            f"- Train Loss: {train_loss:.4f} "
            f"- Train Acc: {train_acc:.4f} "
            f"- Test Acc: {test_acc:.4f}"
            f" - Time: {epoch_time:.2f} sec"
        )

    # Return accuracies and final trained weights
    return train_accs, test_accs, model.state_dict()


import numpy as np
import torch

# Mixup helper functions
def mixup_data(x, y, alpha=0.2):
    '''Returns mixed inputs, mixed targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_model_mixup(model, train_loader, test_loader, criterion, optimizer, device, num_epochs, mixup_alpha=0.2):
    train_accs, test_accs = [], []
    global_step = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_loss, correct, total = 0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            # Apply Mixup
            imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, alpha=mixup_alpha)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)

            # For accuracy, use the "hard" labels of the larger weight
            _, predicted = outputs.max(1)
            hard_labels = torch.where(lam >= 0.5, labels_a, labels_b)
            total += hard_labels.size(0)
            correct += predicted.eq(hard_labels).sum().item()

            global_step += 1

        train_loss = total_loss / total
        train_acc = correct / total
        train_accs.append(train_acc)

        # ---- Evaluation ----
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = test_correct / test_total
        test_accs.append(test_acc)

        end_time = time.time()
        epoch_time = end_time - start_time

        print(
            f"Epoch {epoch+1}/{num_epochs} "
            f"- Train Loss: {train_loss:.4f} "
            f"- Train Acc: {train_acc:.4f} "
            f"- Test Acc: {test_acc:.4f} "
            f"- Time: {epoch_time:.2f} sec"
        )

    return train_accs, test_accs, model.state_dict()
