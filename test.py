import Single_stage_tools as single_stage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split


# Defining each component of SingleStageModel

class SimpleBackbone(nn.Module):
    def __init__(self):
        super(SimpleBackbone, self).__init__()
        # A simple convolutional backbone
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # (B, 16, 224, 224)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),                           # (B, 16, 112, 112)
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # (B, 32, 112, 112)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                           # (B, 32, 56, 56)
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (B, 64, 56, 56)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)                            # (B, 64, 28, 28)
        )
    def forward(self, x):
        return self.features(x)

class SimpleClassificationHead(nn.Module):
    def __init__(self, in_channels=64, num_classes=37):
        # Oxford-IIIT Pet has 37 classes.
        super(SimpleClassificationHead, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)
        # Expose weights so we can use them for CAM generation.
        self.weight = self.fc.weight  
        
    def forward(self, x):
        x = self.gap(x)          # (B, C, 1, 1)
        x = x.view(x.size(0), -1)  # (B, C)
        return self.fc(x)        # (B, num_classes)

class SimpleSegmentationHead(nn.Module):
    def __init__(self, in_channels=64, num_seg_classes=37):
        # For binary segmentation (foreground vs. background)
        super(SimpleSegmentationHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(32, num_seg_classes, kernel_size=1)
        # Upsample back to input resolution
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)  # (B, num_seg_classes, 28, 28)
        x = self.upsample(x)  # (B, num_seg_classes, 224, 224)
        return x
    

if __name__ == "__main__":
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Download training and validation datasets
    trainval_dataset = single_stage.download_data(pth="./data", split="trainval")
    train_size = int(0.8 * len(trainval_dataset))
    val_size = len(trainval_dataset) - train_size

    # Optionally, set a random seed for reproducibility
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(trainval_dataset, [train_size, val_size])

    

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Test several segmentation heads
    results = single_stage.test_segmentation_heads(train_loader, val_loader, num_epochs=5, lr=1e-3, device=device)
    



    
    


    