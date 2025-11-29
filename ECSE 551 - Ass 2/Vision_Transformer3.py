# %%
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from warmup_scheduler import GradualWarmupScheduler


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

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, in_chans=3, embed_dim=128):
        super().__init__()

        def conv_block(in_channels, out_channels, pool=False):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            ]
            if pool:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            return nn.Sequential(*layers)

        self.patcher = nn.Sequential(
            conv_block(in_chans, 64, pool=True),    # 32x32 -> 16x16
            conv_block(64, 128, pool=True),         # 16x16 -> 8x8
            conv_block(128, 256, pool=True),        # 8x8 -> 4x4
            nn.Conv2d(256, embed_dim, kernel_size=1),  # project to embed_dim
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

        self.grid_h = img_size // 8
        self.grid_w = img_size // 8
        self.num_patches = self.grid_h * self.grid_w
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        x = self.patcher(x)       # [batch, embed_dim, 4, 4]
        x = x.flatten(2)  # [batch, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [batch, num_patches, embed_dim]
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, num_patches):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim)) # Initialize position embeddings with random weights that will be learned
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim)) # Initialize CLS token

    def forward(self, x):
        # input comes from PatchEmbedding
        batch_size = x.shape[0] # get batch size
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # prepend [CLS] token
        x = x + self.pos_embed # add position embeddings
        return x

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, 
                                         num_heads=num_heads, 
                                         dropout=dropout,
                                         batch_first=True)  # keeps input as [B, N, C]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)  # query, key, value all the same
        x = self.dropout(attn_output)
        return x


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, dropout):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio) # number of hidden units in MLP
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # attention + residual
        x = x + self.mlp(self.norm2(x))   # MLP + residual
        return x


class VisionTransformer(nn.Module): 
    def __init__(self, 
                 img_size, # input image size
                 in_chans, # input image channels, 3 for RGB
                 num_classes, # number of output classes
                 embed_dim, # embedding dimension
                 depth, # depth controls number of "n = loops" to do the transformer block
                 num_heads, # number of attention heads
                 mlp_ratio, # Controls hidden dimension of MLP as a multiple of embed_dim
                 dropout):  # dropout rate
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size=img_size, in_chans=in_chans, embed_dim=embed_dim) # call PatchEmbedding class defined above
        num_patches = self.patch_embed.num_patches # get number of patches from PatchEmbedding

        self.pos_embed = PositionalEncoding(embed_dim, num_patches) # call PositionalEncoding class defined above

        # Use nn.Sequential to stack TransformerBlocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x): # forward pass
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        x = self.blocks(x)        # forward through all Transformer blocks
        x = self.norm(x)
        cls_token = x[:, 0]       # extract [CLS] token
        return self.head(cls_token)



# Transformations for training (with augmentation)
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),    # random crop with padding
    transforms.RandomHorizontalFlip(),       # horizontal flip
    transforms.TrivialAugmentWide(),         # automatic augmentation
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],      # CIFAR-10 mean
        std=[0.2470, 0.2435, 0.2616]        # CIFAR-10 std
    )
])

# Transformations for test/validation (no augmentation)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
])

num_epochs = 50
batch_size = 32
weight_decay = 0.1
dropout_rate = 0.5
attention_heads = 2
layers = 2
hidden_dimension = 32
mlp_ratio = 2

init_lr = 1e-3
final_lr = 1e-5
warmup_epochs = 5

# Load full dataset
full_dataset = CustomImageDataset(
    csv_file='train_labels.csv',
    img_dir='train',
    transform=None  # we will assign transforms later
)

# Split dataset
train_size = int(0.7 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

# Assign transforms to the subsets
train_dataset.dataset.transform = train_transform
test_dataset.dataset.transform = test_transform

# Data loaders
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VisionTransformer(
                            img_size=32, # input image size
                            in_chans=3,  # input image channels, 3 for RGB
                            num_classes=10, # number of output classes
                            embed_dim=hidden_dimension, # embedding dimension
                            depth=layers, # depth controls number of "n = loops" to do the transformer block
                            num_heads=attention_heads, # number of attention heads
                            mlp_ratio=mlp_ratio, # Controls hidden dimension of MLP as a multiple of embed_dim
                            dropout=dropout_rate # dropout rate
).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Optimizer
optimizer = optim.AdamW(
    model.parameters(),
    lr=2e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay= weight_decay
)

# Cosine scheduler (will be wrapped by warmup)
# Note: T_max is total steps after warmup
total_steps = 78125
warmup_steps = 21000
# Cosine scheduler
base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=total_steps - warmup_steps,
    eta_min=final_lr
)
scheduler = GradualWarmupScheduler(
    optimizer,
    multiplier=1.0,
    total_epoch=warmup_steps,  # warmup_steps in steps
    after_scheduler=base_scheduler
)
train_accs = []
test_accs = []
global_step = 0
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        scheduler.step()  # step **per batch**
        global_step += 1

        total_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = total_loss / total
    train_acc = correct / total
    train_accs.append(train_acc)
    model.eval()
    test_correct = 0
    test_total = 0

    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = test_correct / test_total
    test_accs.append(train_acc)
    print(
        f"Epoch {epoch+1}/{num_epochs} "
        f"- Train Loss: {train_loss:.4f} "
        f"- Train Acc: {train_acc:.4f} "
        f"- Test Acc: {test_acc:.4f}"
    )

    
# save metrics
torch.save(
    {
        "train_acc": train_accs,
        "val_acc": test_accs
    },
    "Vision_Transformer_Metrics.pt"
)
# %%

# save the trained model weights
torch.save(model.state_dict(), "VIT3_weights.pt")
print("Saved Vision Transformer 3 weights as VIT3_weights.pt")