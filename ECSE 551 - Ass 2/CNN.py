import torch
import torch.nn as nn
import torch.optim as optim 
from helper_functions import *


class CNN(nn.Module):
    def __init__(self, num_classes, kernel_size=3, pooling_size=2):
        super().__init__()

        self.kernel_size = kernel_size
        self.pooling_size = pooling_size

        self.successive_convolution = nn.Sequential(
            self.conv_block(4),
            self.conv_block(4),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_classes)
        )

    def conv_block(self, out_channels):
        return nn.Sequential(
            nn.LazyConv2d(out_channels, self.kernel_size, padding=1),
            nn.LazyBatchNorm2d(),
            nn.GELU(),
            nn.MaxPool2d(self.pooling_size)
        )

    def forward(self, x):
        x = self.successive_convolution(x)
        x = self.classifier(x)
        return x



device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.2)

# Hyperparameters
batch_size = 32
weight_decay = 0.1
num_epochs = 25

train_loader, test_loader = load_data(
    dataset_class=CustomImageDataset,
    csv_file='train_labels.csv',
    img_dir='train',
    batch_size=batch_size,
    augment=True,
    train_ratio=0.8,
)
# Optimizer
optimizer = optim.AdamW(
    model.parameters(),
    lr=2e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay= weight_decay
)

train_accs, test_accs, weights = train_model(
    model, train_loader, test_loader, criterion, optimizer, device, num_epochs
)