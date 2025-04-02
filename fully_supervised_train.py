from torchvision.datasets import OxfordIIITPet
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.models.segmentation as segmentation
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def download_data(pth: str, split: str):
    assert split in ["trainval", "test"]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    target_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.long) - 1)
    ])
    dataset = OxfordIIITPet(root=pth, split=split, target_types=['segmentation'], download=True, 
                            transform=transform, target_transform=target_transform)
    return dataset

# Load datasets
train_dataset = download_data('./data', 'trainval')
test_dataset = download_data('./data', 'test')

# Data loaders with num_workers
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

# Verify loader sizes
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of testing batches: {len(test_loader)}")

# Initialize model
model = segmentation.deeplabv3_resnet50(weights=None, num_classes=3)
device = torch.device('cpu')
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with tqdm
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1}")
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Save the model
torch.save(model.state_dict(), 'deeplabv3_resnet50_fully_supervised.pth')

def evaluate_model(model, data_loader, device, num_classes=3):
    model.eval()  # Set model to evaluation mode
    total_pixel_acc = 0.0
    total_iou = 0.0
    total_f1 = 0.0
    num_batches = 0

    # Loop through the test data
    for images, masks in tqdm(data_loader, desc="Evaluating"):
        images, masks = images.to(device), masks.to(device)
        with torch.no_grad():
            outputs = model(images)['out']
        # Get predictions by taking the argmax across the class dimension
        preds = torch.argmax(outputs, dim=1)

        # Pixel-wise Accuracy
        pixel_acc = (preds == masks).float().mean().item()
        total_pixel_acc += pixel_acc

        # For each class, compute IoU and F1 Score
        ious = []
        f1_scores = []
        for cls in range(num_classes):
            # Create binary masks for the current class
            pred_inds = preds == cls
            target_inds = masks == cls

            # IoU: Intersection over Union
            intersection = (pred_inds & target_inds).sum().item()
            union = (pred_inds | target_inds).sum().item()
            if union == 0:
                iou = float('nan')  # Ignore classes with no pixels in ground truth
            else:
                iou = intersection / union
            ious.append(iou)

            # F1 Score: 2 * (precision * recall) / (precision + recall)
            tp = intersection
            # false positives: pixels predicted as the class but not in ground truth
            fp = (pred_inds & ~target_inds).sum().item()
            # false negatives: pixels in ground truth but not predicted as the class
            fn = ((~pred_inds) & target_inds).sum().item()
            if (2 * tp + fp + fn) == 0:
                f1 = float('nan')
            else:
                f1 = 2 * tp / (2 * tp + fp + fn)
            f1_scores.append(f1)

        # Average the per-class metrics (ignoring any NaNs)
        batch_iou = np.nanmean(ious)
        batch_f1 = np.nanmean(f1_scores)

        total_iou += batch_iou
        total_f1 += batch_f1
        num_batches += 1

    # Compute average metrics over all batches
    avg_pixel_acc = total_pixel_acc / num_batches
    avg_iou = total_iou / num_batches
    avg_f1 = total_f1 / num_batches

    return avg_pixel_acc, avg_f1, avg_iou

# Example: Evaluate on your test dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

test_pixel_acc, test_f1, test_iou = evaluate_model(model, test_loader, device, num_classes=3)
print(f"Test Pixel Accuracy: {test_pixel_acc:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print(f"Test IoU: {test_iou:.4f}")
