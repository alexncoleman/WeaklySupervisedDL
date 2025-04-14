import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import resnet50


# --- Classification Model ---
class FrozenResNetCAM(nn.Module):
    def __init__(self, num_classes=37):
        super().__init__()
        resnet = resnet50(pretrained=True, replace_stride_with_dilation=[False, False, True])
        for param in resnet.parameters():
            param.requires_grad = False

        # Split layers for easier access

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        # self.layer5 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.layer0(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        # f5 = self.layer5(f4)

        pooled = self.avgpool(f4)
        flat = pooled.view(pooled.size(0), -1)
        logits = self.fc(flat)

        # Return all feature maps for LayerCAM
        return logits, [f2, f3, f4]


# def train_fc_only(model, dataloader, device, epochs=10):
#     model.to(device)
#     model.train()

#     optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)  # Only train FC layer
#     criterion = nn.CrossEntropyLoss()

#     for epoch in range(epochs):
#         total_loss, correct, total = 0.0, 0, 0
#         for imgs, (labels, _) in dataloader:
#             imgs, labels = imgs.to(device), labels.to(device)
#             logits, _ = model(imgs)
#             loss = criterion(logits, labels)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item() * imgs.size(0)
#             preds = logits.argmax(dim=1)
#             correct += (preds == labels).sum().item()
#             total += imgs.size(0)

#         print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / total:.4f} - Acc: {100 * correct / total:.2f}%")

#     model.eval()
    
def train_fc_only(model, device, epochs=10, num_classes=37):
    model.to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_dataset, val_dataset = load_split_data()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, (labels, _) in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits, _ = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

        train_acc = 100 * correct / total
        avg_loss = total_loss / total
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f} - Train Acc: {train_acc:.2f}%")

        # --- Run validation ---
        val_acc, val_f1 = evaluate_classification(model, val_loader, device, num_classes=num_classes)
        print(f"           --> Val Acc: {val_acc:.2f}% - Val F1: {val_f1:.4f}")

    model.eval()


def evaluate_classification(model, dataloader, device, num_classes=37):
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    true_positives = torch.zeros(num_classes).to(device)
    false_positives = torch.zeros(num_classes).to(device)
    false_negatives = torch.zeros(num_classes).to(device)

    for imgs, (labels, _) in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits, _ = model(imgs)
        preds = torch.argmax(logits, dim=1)

        # Accuracy
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # F1 computation (macro)
        for cls in range(num_classes):
            tp = ((preds == cls) & (labels == cls)).sum()
            fp = ((preds == cls) & (labels != cls)).sum()
            fn = ((preds != cls) & (labels == cls)).sum()

            true_positives[cls] += tp
            false_positives[cls] += fp
            false_negatives[cls] += fn

    # Avoid division by zero
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    macro_f1 = f1.mean().item()
    acc = 100.0 * correct / total

    print(f"Evaluation - Accuracy: {acc:.2f}% - F1 Score (macro): {macro_f1:.4f}")
    return acc, macro_f1
