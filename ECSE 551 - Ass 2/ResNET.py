import torch
import torch.nn as nn
import torch.optim as optim 
from helper_functions import *


class ResNET_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        # Main convolutional path
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.CELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels


    def forward(self, x):
        input = x
        if self.downsample:
            input = self.downsample(input)
        out = self.residual(x) + input
        out = self.relu(out)
        return out

class ResNet_10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Layer 1: Initial conv + bn + relu
        self.first_layer = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.CELU(inplace=True)
        )
        

        self.layers = nn.Sequential(
            ResNET_Block(64,128, stride=1),   # layer 2-3: no downsample
            ResNET_Block(128,256, stride=2),  # layer 4-5: downsample
            ResNET_Block(256,512, stride=2),  # layer 6-7: downsample
            ResNET_Block(512,512, stride=2)   # layer 8-9: downsample
        )

        
        
        # Layer 10 Global average pooling + classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.LazyLinear(num_classes)
        )

        
    
    def forward(self, x):
        x = self.first_layer(x)
        x = self.layers(x)
        x = self.classifier(x)
        return x

class ResNet_32(nn.Module):
    def __init__(self, ResNET_block, n_layers, num_classes=10):
        super().__init__()

        self.input_channels = 64
        
        self.first_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
            nn.BatchNorm2d(64),
            nn.CELU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.layers = nn.Sequential(
            self._stack(ResNET_block, output_channels = 64, n_layers = n_layers[0], stride=1),
            self._stack(ResNET_block, output_channels = 128, n_layers = n_layers[1], stride=2),
            self._stack(ResNET_block, output_channels = 256, n_layers = n_layers[2], stride=2),
            self._stack(ResNET_block, output_channels = 512, n_layers = n_layers[3], stride=2)
        )
        # Layer n Global average pooling + classifier
        self.classifier = nn.Sequential(
            nn.AvgPool2d(7, stride=1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def _stack(self, ResNET_block, output_channels, n_layers, stride=1):
        downsample = None
        if stride != 1 or self.input_channels != output_channels:
            
            downsample = nn.Sequential(
                nn.LazyConv2d(output_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(output_channels),
            )
        layers = [ResNET_block(self.input_channels, output_channels, stride, downsample)]
        self.input_channels = output_channels
        for i in range(1, n_layers):
            layers.append(ResNET_block(self.input_channels, output_channels))

        return nn.Sequential(*layers)   
    
    def forward(self, x):
        x = self.first_layer(x)
        x = self.maxpool(x)
        x = self.layers(x)
        x = self.classifier(x)
        return x



# # Hyperparameters
batch_size = 32
weight_decay = 0.001
num_epochs = 25
learning_rate = 0.01
label_smoothing = 0.0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ResNet_32(ResNET_block= ResNET_Block, n_layers = [3, 4, 6, 3], num_classes=10).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)



train_loader, test_loader = load_data(
    dataset_class=CustomImageDataset,
    csv_file='train_labels.csv',
    img_dir='train',
    batch_size=batch_size,
    augment=False,
    train_ratio=0.7,
)
# Optimizer
# optimizer = optim.AdamW(
#     model.parameters(),
#     lr=learning_rate,
#     betas=(0.9, 0.999),
#     eps=1e-8,
#     weight_decay= weight_decay,
#     momentum=0.9
# )
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)  
train_accs, test_accs, weights = train_model(
    model, train_loader, test_loader, criterion, optimizer, device, num_epochs
)