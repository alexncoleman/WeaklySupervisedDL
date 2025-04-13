from torchvision.datasets import OxfordIIITPet
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.models.segmentation as segmentation
import torch.nn as nn
import torch.optim as optim

def download_data(pth: str, split: str):
    assert split in ["trainval", "test"]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    target_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
    ])

    dataset = OxfordIIITPet(root=pth, split=split, target_types='segmentation',
                            download=True, transform=transform, target_transform=None)

    class CustomOxfordPetDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, target_transform):
            self.dataset = dataset
            self.target_transform = target_transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            image, seg_mask = self.dataset[idx]
            seg_mask = self.target_transform(seg_mask)
            seg_mask = torch.tensor(np.array(seg_mask), dtype=torch.long)
            new_mask = (seg_mask == 1).long()
            return image, new_mask

    return CustomOxfordPetDataset(dataset, target_transform)

train_dataset = download_data('./data', 'trainval')
test_dataset = download_data('./data', 'test')

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

print(f"Number of training batches: {len(train_loader)}")
print(f"Number of testing batches: {len(test_loader)}")

model = segmentation.deeplabv3_resnet50(weights=None, num_classes=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 10
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1}")
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), 'deeplabv3_resnet50_binary_segmentation.pth')

def evaluate_model(model, data_loader, device, num_classes=2):
    model.eval()
    total_pixel_acc = 0.0
    total_iou = 0.0
    num_batches = 0
    iou_per_class = np.zeros(num_classes)
    class_counts = np.zeros(num_classes)

    for images, masks in data_loader:
        images, masks = images.to(device), masks.to(device)
        with torch.no_grad():
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

num_runs = 3
pixel_accs = []
ious = []

for run in range(num_runs):
    print(f"Starting evaluation run {run + 1}/{num_runs}")
    pixel_acc, iou = evaluate_model(model, test_loader, device, num_classes=2)
    pixel_accs.append(pixel_acc)
    ious.append(iou)
    print(f"Run {run + 1} - Pixel Accuracy: {pixel_acc:.4f}, IoU: {iou:.4f}")

avg_pixel_acc = np.mean(pixel_accs)
std_pixel_acc = np.std(pixel_accs)
avg_iou = np.mean(ious)
std_iou = np.std(ious)

print("\nFinal Results:")
print(f"Average Pixel Accuracy: {avg_pixel_acc:.4f} ± {std_pixel_acc:.4f}")
print(f"Average IoU: {avg_iou:.4f} ± {std_iou:.4f}")