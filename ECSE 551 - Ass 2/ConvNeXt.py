import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from warmup_scheduler import GradualWarmupScheduler
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

# -----------------------
# Dataset
# -----------------------

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        self.classes = sorted(self.data['label'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.data['label_idx'] = self.data['label'].map(self.class_to_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = f"{self.img_dir}/{self.data.iloc[idx]['id']}"
        image = Image.open(img_name).convert('RGB')
        label = self.data.iloc[idx]['label_idx']

        if self.transform:
            image = self.transform(image)

        return image, label


# -----------------------
# Transforms
# -----------------------

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# -----------------------
# Load dataset + split 70:30
# -----------------------

csv_file = "train_labels.csv"
img_dir = "train"

full_dataset = CustomImageDataset(csv_file, img_dir, transform=train_transform)

train_size = int(0.7 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

val_dataset.dataset.transform = test_transform  # override transform for val set

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


# -----------------------
# Model: ConvNeXt
# -----------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

num_classes = len(full_dataset.classes)

weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
model = convnext_tiny(weights=weights)

in_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(in_features, num_classes)

model = model.to(device)


# -----------------------
# Optimizer + Warmup
# -----------------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
scheduler = GradualWarmupScheduler(
    optimizer, multiplier=1.0, total_epoch=5, after_scheduler=scheduler_cosine
)


# -----------------------
# Training loop
# -----------------------

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    print(f"Epoch {epoch+1}/{num_epochs}  Loss: {total_loss/len(train_loader):.4f}")


# -----------------------
# Validation accuracy
# -----------------------

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

acc = correct / total
print("Validation accuracy:", acc)
