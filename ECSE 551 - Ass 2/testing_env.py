# %%

from ResNET import *
from helper_functions import *

# Hyperparameters
batch_size = 16
weight_decay = 0.1
num_epochs = 25
learning_rate = 0.001
label_smoothing = 0.1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#model = ResNet_10(ResNET_Block,).to(device)

model = ResNet_32(ResNET_Block, [1, 1, 1, 1], num_classes=10).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)



train_loader, test_loader = load_data(
    dataset_class=CustomImageDataset,
    csv_file='train_labels.csv',
    img_dir='train',
    batch_size=batch_size,
    augment=True,
    train_ratio=0.7,
)
# Optimizer

from sam import SAM
base_opt  = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)
optimizer = SAM(model.parameters(), base_opt, rho=0.05)

def centralize_gradients(model):
    for p in model.parameters():
        if p.grad is not None and len(p.grad.shape) > 1:  # conv/dense weights
            p.grad.data -= p.grad.data.mean(dim=list(range(1,len(p.grad.shape))), keepdim=True)

num_epochs = 50
device = "cuda" if torch.cuda.is_available() else "cpu"

for epoch in range(num_epochs):
    model.train()
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        # ----------------------
        # Step 1: forward-backward + first SAM step
        # ----------------------
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.first_step(zero_grad=True)  # performs ascent step

        # ----------------------
        # Step 2: forward-backward + second SAM step
        # ----------------------
        criterion(model(x), y).backward()
        optimizer.second_step(zero_grad=True)  # update weights

    # ----------------------
    # Validation
    # ----------------------
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x_val, y_val in test_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            outputs = model(x_val)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_val).sum().item()
            total += y_val.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch+1:02d} | Validation Accuracy: {val_acc:.4f}")
# optimizer = torch.optim.AdamW(
#     model.parameters(),
#     lr=learning_rate,
#     betas=(0.9, 0.999),
#     eps=1e-8,
#     weight_decay= weight_decay,
# )

# train_accs, test_accs, weights = train_model(
#     model, train_loader, test_loader, criterion, optimizer, device, num_epochs
# )

# train_accs, test_accs, weights = train_model_mixup(
#     model, 
#     train_loader, 
#     test_loader, 
#     criterion, 
#     optimizer, 
#     device, 
#     num_epochs,
#     mixup_alpha=0.2   # controls how strongly the images/labels are mixed
# )



# %%

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


full_dataset = CustomImageDataset(csv_file='train_labels.csv', img_dir='train', transform=None )# we will assign transforms later )
# Collect all true labels and predictions
all_labels = []
all_preds = []

model.eval()
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = outputs.max(1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)

# Per-class statistics
report = classification_report(all_labels, all_preds, target_names=full_dataset.classes)
print("Classification Report:\n", report)

# %%
train_labels = []
train_preds = []

model.eval()  # just evaluation mode
with torch.no_grad():
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = outputs.max(1)
        train_labels.extend(labels.cpu().numpy())
        train_preds.extend(predicted.cpu().numpy())

from sklearn.metrics import accuracy_score

for i, cls in enumerate(full_dataset.classes):
    cls_train_mask = np.array(train_labels) == i
    cls_test_mask = np.array(all_labels) == i
    train_acc_cls = accuracy_score(np.array(train_labels)[cls_train_mask], np.array(train_preds)[cls_train_mask])
    test_acc_cls = accuracy_score(np.array(all_labels)[cls_test_mask], np.array(all_preds)[cls_test_mask])
    print(f"Class {cls}: Train Acc = {train_acc_cls:.2f}, Test Acc = {test_acc_cls:.2f}")
