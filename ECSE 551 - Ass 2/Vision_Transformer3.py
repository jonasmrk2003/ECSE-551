# %%
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from warmup_scheduler import GradualWarmupScheduler
import copy
from pathlib import Path


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



def build_transforms(augment=True):
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            )
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            )
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    return train_transform, test_transform


def load_dataloaders(
    csv_file,
    img_dir,
    batch_size,
    train_ratio,
    val_ratio,
    augment,
    seed=42,
    split_save_path=None,
    split_load_path=None,
):
    train_t, test_t = build_transforms(augment=augment)
    full_dataset = CustomImageDataset(csv_file=csv_file, img_dir=img_dir, transform=None)
    if split_load_path and Path(split_load_path).exists():
        saved = torch.load(split_load_path, map_location="cpu")
        train_idx = saved.get("train_indices", [])
        val_idx = saved.get("val_indices", [])
        test_idx = saved.get("test_indices", [])
        train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
        test_dataset = torch.utils.data.Subset(full_dataset, test_idx)
    else:
        train_size = int(train_ratio * len(full_dataset))
        val_size = int(val_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )
        if split_save_path:
            torch.save(
                {
                    "train_indices": train_dataset.indices,
                    "val_indices": val_dataset.indices,
                    "test_indices": test_dataset.indices,
                },
                split_save_path,
            )
    train_dataset.dataset.transform = train_t
    val_dataset.dataset.transform = test_t
    test_dataset.dataset.transform = test_t
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def make_optimizer(model, optimizer_name, lr, weight_decay, momentum=0.9):
    name = optimizer_name.lower()
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam" or name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay
        )
    raise ValueError(f"Unknown optimizer {optimizer_name}")


def train_one_vit(config):
    """
    Train one ViT config and save weights/metrics.
    config keys: label, batch_size, weight_decay, dropout, heads, layers,
    hidden_dim, mlp_ratio, lr, optimizer, epochs, train_ratio, augment, csv_file, img_dir
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader = load_dataloaders(
        csv_file=config["csv_file"],
        img_dir=config["img_dir"],
        batch_size=config["batch_size"],
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        augment=config["augment"],
        seed=config.get("seed", 42),
        split_save_path=config.get("split_save_path"),
        split_load_path=config.get("split_load_path"),
    )

    model = VisionTransformer(
        img_size=32,
        in_chans=3,
        num_classes=10,
        embed_dim=config["hidden_dim"],
        depth=config["layers"],
        num_heads=config["heads"],
        mlp_ratio=config["mlp_ratio"],
        dropout=config["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = make_optimizer(
        model,
        optimizer_name=config["optimizer"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        momentum=config.get("momentum", 0.9),
    )

    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = int(0.3 * total_steps)
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_steps - warmup_steps, 1),
        eta_min=config.get("final_lr", 1e-5),
    )
    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=1.0,
        total_epoch=max(warmup_steps, 1),
        after_scheduler=base_scheduler,
    )

    train_accs, val_accs = [], []
    best_state = None
    best_val_acc = -1.0

    for epoch in range(config["epochs"]):
        model.train()
        correct, total = 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_acc = correct / total
        train_accs.append(train_acc)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"[{config['label']}] Epoch {epoch+1}/{config['epochs']} "
            f"- Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}"
        )

    weights_path = f"VIT3_{config['label']}.pt"
    torch.save(best_state if best_state is not None else model.state_dict(), weights_path)
    # Evaluate best on test set
    model.load_state_dict(best_state if best_state is not None else model.state_dict())
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

    metrics_path = f"VIT3_{config['label']}_metrics.pt"
    torch.save({"train_acc": train_accs, "val_acc": val_accs, "test_acc": test_acc}, metrics_path)
    print(
        f"[{config['label']}] Saved weights -> {weights_path}, metrics -> {metrics_path} "
        f"(best val: {best_val_acc:.4f}, test: {test_acc:.4f})"
    )


if __name__ == "__main__":
    # Train a new ViT on a simple 70/30 split (seed 42), same as ResNet now uses
    experiments = [
        {
            "label": "vit3_resnet_split_sgd",
            "batch_size": 32,
            "weight_decay": 0.001,
            "dropout": 0.5,
            "heads": 2,
            "layers": 2,
            "hidden_dim": 128,
            "mlp_ratio": 2,
            "lr": 0.01,
            "optimizer": "sgd",
            "momentum": 0.9,
            "epochs": 20,
            "train_ratio": 0.7,
            "val_ratio": 0.15,  # 70/15/15 split
            "augment": True,
            "csv_file": "train_labels.csv",
            "img_dir": "train",
            "seed": 42,
            # Force a fresh split and save it; do not reuse an old split with no val set
            "split_load_path": None,
            "split_save_path": "vit3_resnet_split_sgd_split.pt",
        },
    ]

    for exp in experiments:
        train_one_vit(exp)
