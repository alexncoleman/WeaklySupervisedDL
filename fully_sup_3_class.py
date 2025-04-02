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
    
    # Apply Resize only to the segmentation mask
    target_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
    ])

    dataset = OxfordIIITPet(root=pth, split=split, target_types=['segmentation', 'category'], 
                          download=True, transform=transform, target_transform=None) # Remove target_transform here
    
    class CustomOxfordPetDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, target_transform): # Pass target_transform to the custom dataset
            self.dataset = dataset
            self.target_transform = target_transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            image, (seg_mask, category) = self.dataset[idx]
            
            # Apply target_transform to the segmentation mask
            seg_mask = self.target_transform(seg_mask)
            
            seg_mask = torch.tensor(np.array(seg_mask), dtype=torch.long)
            if category <= 11:  # Cat
                pet_class = 0
            else:  # Dog
                pet_class = 1
            new_mask = torch.zeros_like(seg_mask, dtype=torch.long)
            new_mask[seg_mask == 1] = pet_class  # Cat (0) or Dog (1)
            new_mask[seg_mask == 2] = 2  # Background
            new_mask[seg_mask == 3] = 2  # Border merged into background
            return image, new_mask

    return CustomOxfordPetDataset(dataset, target_transform) # Pass target_transform here

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # No ignore_index since we want to classify background
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
    model.eval()
    total_pixel_acc = 0.0
    total_iou = 0.0
    total_f1 = 0.0
    num_batches = 0
    class_names = ['Cat', 'Dog', 'Background']
    iou_per_class = np.zeros(num_classes)
    f1_per_class = np.zeros(num_classes)
    class_counts = np.zeros(num_classes)

    for images, masks in tqdm(data_loader, desc="Evaluating"):
        images, masks = images.to(device), masks.to(device)
        with torch.no_grad():
            outputs = model(images)['out']
        preds = torch.argmax(outputs, dim=1)

        pixel_acc = (preds == masks).float().mean().item()
        total_pixel_acc += pixel_acc

        ious = []
        f1_scores = []
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

            tp = intersection
            fp = (pred_inds & ~target_inds).sum().item()
            fn = ((~pred_inds) & target_inds).sum().item()
            if (2 * tp + fp + fn) == 0:
                f1 = float('nan')
            else:
                f1 = 2 * tp / (2 * tp + fp + fn)
            f1_scores.append(f1)

            if not np.isnan(iou):
                iou_per_class[cls] += iou
                class_counts[cls] += 1
            if not np.isnan(f1):
                f1_per_class[cls] += f1

        batch_iou = np.nanmean(ious)
        batch_f1 = np.nanmean(f1_scores)
        total_iou += batch_iou
        total_f1 += batch_f1
        num_batches += 1

    avg_pixel_acc = total_pixel_acc / num_batches
    avg_iou = total_iou / num_batches
    avg_f1 = total_f1 / num_batches

    iou_per_class = iou_per_class / (class_counts + 1e-6)
    f1_per_class = f1_per_class / (class_counts + 1e-6)

    print("Per-class IoU:")
    for cls, iou in enumerate(iou_per_class):
        print(f"{class_names[cls]}: {iou:.4f}")
    print("Per-class F1 Score:")
    for cls, f1 in enumerate(f1_per_class):
        print(f"{class_names[cls]}: {f1:.4f}")

    return avg_pixel_acc, avg_f1, avg_iou

# Evaluate on test dataset
test_pixel_acc, test_f1, test_iou = evaluate_model(model, test_loader, device, num_classes=3)
print(f"Test Pixel Accuracy: {test_pixel_acc:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print(f"Test IoU: {test_iou:.4f}")