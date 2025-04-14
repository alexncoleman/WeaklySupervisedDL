import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
from torchvision import models as segmentation
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet

# Assumes load_split_data and download_data already exist as per your earlier code

def initialize_model(num_classes=2, device=None):
    model = segmentation.deeplabv3_resnet50(weights=None, num_classes=num_classes)
    model.to(device)
    return model

def get_dataloaders(data_path='./data', train_ratio=0.85, batch_size=16, num_workers=0):
    train_dataset, val_dataset = load_split_data(pth=data_path, train_ratio=train_ratio)
    test_dataset = download_data(pth=data_path, split='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")
    return train_loader, val_loader, test_loader

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(dataloader)

@torch.no_grad()
def evaluate_model(model, dataloader, device, num_classes=2):
    model.eval()
    total_pixel_acc = 0.0
    total_iou = 0.0
    num_batches = 0
    iou_per_class = np.zeros(num_classes)
    class_counts = np.zeros(num_classes)

    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)['out']
        preds = torch.argmax(outputs, dim=1)

        pixel_acc = (preds == masks).float().mean().item()
        total_pixel_acc += pixel_acc

        ious = []
        for cls in range(num_classes):
            pred_inds = preds == cls
            target_inds = masks == cls
            intersection = (pred_inds & target_inds).sum().item()
            union = (pred_inds | target_inds).sum().item()
            if union == 0:
                iou = float('nan')
            else:
                iou = intersection / union
            ious.append(iou)

            if not np.isnan(iou):
                iou_per_class[cls] += iou
                class_counts[cls] += 1

        batch_iou = np.nanmean(ious)
        total_iou += batch_iou
        num_batches += 1

    avg_pixel_acc = total_pixel_acc / num_batches
    avg_iou = total_iou / num_batches
    return avg_pixel_acc, avg_iou

def run_supervised_training(
    data_path='./data',
    num_epochs=10,
    batch_size=16,
    train_ratio=0.85,
    num_classes=2,
    lr=1e-4,
    device=None
):
    train_loader, val_loader, test_loader = get_dataloaders(data_path, train_ratio, batch_size)
    model = initialize_model(num_classes=num_classes, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_iou = evaluate_model(model, val_loader, device, num_classes)

        print(f"Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val IoU: {val_iou:.4f}")

    # Save final model
    torch.save(model.state_dict(), 'deeplabv3_resnet50_binary_segmentation.pth')

    # Evaluate on test set
    test_runs = 3
    pixel_accs = []
    ious = []

    for run in range(test_runs):
        print(f"\nTest Run {run + 1}/{test_runs}")
        pixel_acc, iou = evaluate_model(model, test_loader, device, num_classes)
        pixel_accs.append(pixel_acc)
        ious.append(iou)
        print(f"Pixel Acc: {pixel_acc:.4f} | IoU: {iou:.4f}")

    print("\nFinal Test Results:")
    print(f"Avg Pixel Acc: {np.mean(pixel_accs):.4f} ± {np.std(pixel_accs):.4f}")
    print(f"Avg IoU: {np.mean(ious):.4f} ± {np.std(ious):.4f}")



# To run later:

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# run_supervised_training(
#     data_path='./data',
#     num_epochs=10,
#     batch_size=16,
#     train_ratio=0.85,
#     num_classes=2,
#     lr=0.0001,
#     device=device
# )
