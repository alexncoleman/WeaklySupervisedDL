from Single_stage_tools import *
import torch
from torchvision.models import resnet18, ResNet18_Weights
import timm
import pandas as pd


class ResNetBackbone(nn.Module):
    def __init__(self, out_channels=512):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        # self.features = nn.Sequential(*list(resnet.children())[:-2])  # Output: (B, 512, 7, 7)
        # self.out_channels = out_channels

        self.features = nn.Sequential(
            resnet.conv1,      # (B, 64, 112, 112)
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,    # (B, 64, 56, 56)
            resnet.layer1, # (B, 64, 56, 56)
            resnet.layer2, # (B, 128, 28, 28)
            resnet.layer3, # (B, 256, 14, 14) 
            resnet.layer4) # (B, 512, 7, 7) 
        
        self.out_channels = out_channels

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.features(x)
    

class HRNetTimmBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        model = timm.create_model('hrnet_w18', pretrained=True, features_only=True)
        self.features = model
        self.out_channels = model.feature_info[-1]['num_chs']  # 480 for hrnet_w18

    def forward(self, x):
        feats = self.features(x)  # List of multi-scale features
        return feats[-1]          # Final layer: (B, 480, 56, 56)
    


class MobileNetTimmBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        model = timm.create_model('mobilenetv2_100', pretrained=True, features_only=True)
        self.features = model
        self.out_channels = model.feature_info[-1]['num_chs']  # 1280

    def forward(self, x):
        feats = self.features(x)
        return feats[-1]
    
# ---- 14x14 output models ----
    
class MobileNetV2Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        model = timm.create_model('mobilenetv2_100', pretrained=True, features_only=True)
        self.features = model
        self.out_channels = model.feature_info[3]['num_chs']  # 14x14 resolution

    def forward(self, x):
        feats = self.features(x)
        return feats[3]  # (B, C, 14, 14)
    
class MobileNetV3Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        model = timm.create_model('mobilenetv3_large_100', pretrained=True, features_only=True)
        self.features = model
        self.out_channels = model.feature_info[3]['num_chs']  # 14x14 resolution

    def forward(self, x):
        feats = self.features(x)
        return feats[3]  # (B, C, 14, 14)
    
class EfficientNetB0Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        model = timm.create_model('efficientnet_b0', pretrained=True, features_only=True)
        self.features = model
        self.out_channels = model.feature_info[-2]['num_chs']  # 14x14 resolution

    def forward(self, x):
        feats = self.features(x)
        return feats[-2]  # (B, C, 14, 14)
    

# --- 28x28 models ---

class MobileNetV2Backbone_28(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        model = timm.create_model('mobilenetv2_100', pretrained=True, features_only=True)
        self.features = model
        self.out_channels = model.feature_info[2]['num_chs']  # 28x28 layer

    def forward(self, x):
        feats = self.features(x)
        return feats[2]  # 28x28 resolution
    
class MobileNetV3Backbone_28(nn.Module):
    def __init__(self):
        super().__init__()
        model = timm.create_model('mobilenetv3_large_100', pretrained=True, features_only=True)
        self.features = model
        self.out_channels = model.feature_info[2]['num_chs']  # 14x14 resolution

    def forward(self, x):
        feats = self.features(x)
        return feats[2]  # (B, C, 28, 28)
    

class ConvNeXtTinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        model = timm.create_model("convnext_tiny", pretrained=True, features_only=True)
        self.features = model
        self.out_channels = model.feature_info[2]['num_chs']  # ConvNeXt stage2 -> 28x28 resolution

    def forward(self, x):
        feats = self.features(x)
        return feats[2]  # (B, 384, 28, 28)
    
class EfficientNetB4Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        model = timm.create_model('tf_efficientnet_b4', pretrained=True, features_only=True)
        self.features = model
        self.feature_index = 2  # 28x28 output
        self.out_channels = model.feature_info[self.feature_index]['num_chs']  # Typically 160

    def forward(self, x):
        feats = self.features(x)
        return feats[self.feature_index]  # (B, 160, 28, 28)
    


# --- Vanilla Model ----

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
            #nn.MaxPool2d(2),                           # (B, 32, 56, 56)
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (B, 64, 56, 56)
            nn.BatchNorm2d(64),
            nn.ReLU()
            # nn.MaxPool2d(2)                            # (B, 64, 28, 28)
        )

        self.out_channels = 64


    def forward(self, x):
        return self.features(x)

# class SimpleClassificationHead(nn.Module):
#     def __init__(self, in_channels=64, num_classes=37):
#         # Oxford-IIIT Pet has 37 classes.
#         super(SimpleClassificationHead, self).__init__()
#         self.gap = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(in_channels, num_classes)
#         # Expose weights so we can use them for CAM generation.
#         self.weight = self.fc.weight  
        
#     def forward(self, x):
#         x = self.gap(x)          # (B, C, 1, 1)
#         x = x.view(x.size(0), -1)  # (B, C)
#         return self.fc(x)        # (B, num_classes)

# class SimpleSegmentationHead(nn.Module):
#     def __init__(self, in_channels=64, num_seg_classes=37):
#         # For binary segmentation (foreground vs. background)
#         super(SimpleSegmentationHead, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
#         self.bn1   = nn.BatchNorm2d(32)
#         self.relu  = nn.ReLU()
#         self.conv2 = nn.Conv2d(32, num_seg_classes, kernel_size=1)
#         # Upsample back to input resolution
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
#     def forward(self, x):
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.conv2(x)  # (B, num_seg_classes, 28, 28)
#         x = self.upsample(x)  # (B, num_seg_classes, 224, 224)
#         return x

class SimpleClassificationHead(nn.Module):
    def __init__(self, in_channels=512, num_classes=37):
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
    def __init__(self, in_channels=512, num_seg_classes=37):
        # For binary segmentation (foreground vs. background)
        super(SimpleSegmentationHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(256)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(256, num_seg_classes, kernel_size=1)
        # Upsample back to input resolution
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)  # (B, num_seg_classes, 28, 28)
        x = self.upsample(x)  # (B, num_seg_classes, 224, 224)
        return x
    

class UNetSegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1)
        )
        self.upsample = nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False)

    def forward(self, x):
        x = self.decode(x)
        return self.upsample(x)


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    dataset = download_data(pth="./data", split="trainval")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

    # # Build model
    # backbone = SimpleBackbone()
    # cls_head = SimpleClassificationHead(in_channels=64, num_classes=37)
    # seg_head = SimpleSegmentationHead(in_channels=64, num_seg_classes=37)

    backbone = EfficientNetB0Backbone()
    cls_head = SimpleClassificationHead(in_channels=backbone.out_channels, num_classes=37)
    #seg_head = SimpleSegmentationHead(in_channels=backbone.out_channels, num_seg_classes=37)

    seg_head = UNetSegmentationHead(in_channels=backbone.out_channels, num_classes=37)

    # params = {
    #     "backbone": [backbone],
    #     "classification_head": [cls_head],
    #     "segmentation_head": [seg_head],
    #     "lr": [1e-4, 5e-4],
    #     "weightSeg": [0.5, 1.0],
    #     "weightCls": [0.5, 1.0],
    #     "num_epochs": [5],
    #     "device": device,
    #     "train_loader": train_loader,
    #     "val_loader": val_loader,
    #     "classification_loss_fn": nn.CrossEntropyLoss(),
    #     "segmentation_loss_fn": nn.CrossEntropyLoss(ignore_index=255),
    # }

    threshold_list = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

    """"
    COMMENT ON threshold_list: Testing these thresholds, thresholds in the range of [0.6, 0.8] return ignored pixel values in range ~ [30%, 70%]
    """""

    results = []
    
    #refined_threshold_list =  [0.55, 0.6, 0.7, 0.8]
    target_thresh_list = [0.3, 0.5, 0.7]
    weight_pairs = [(1.0, 1.0), (0.5, 1.5), (1.5, 0.5)]

    results_df = pd.DataFrame()


    for t in target_thresh_list: 

        for weightCls, weightSeg in weight_pairs:


            print()
            print(f"--- Training with threshold: {t}, weightCls: {weightCls}, weightSeg: {weightSeg} ---")
            print()
            model = SingleStageWeaklySupervisedModel(backbone, cls_head, seg_head, target_threshold = t)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            # Train
            _, val_metrics_list = train_single_stage_model(
                model,
                train_loader=train_loader,
                val_loader=val_loader,
                classification_loss_fn=nn.CrossEntropyLoss(),
                segmentation_loss_fn=nn.CrossEntropyLoss(ignore_index=255),
                weightCls=weightCls,
                weightSeg=weightSeg,
                optimizer=optimizer,
                device=device,
                num_epochs=3,
                print_every=1
            )

            # Validate
            # validate_single_stage_model(
            #     model,
            #     val_loader,
            #     classification_loss_fn=nn.CrossEntropyLoss(),
            #     segmentation_loss_fn=nn.CrossEntropyLoss(ignore_index=255),
            #     device=device
            # )

            for metrics in val_metrics_list:
                metrics.update({
                    "threshold": t,
                    "weightCls": weightCls,
                    "weightSeg": weightSeg
                })
                results_df = pd.concat([results_df, pd.DataFrame([metrics])], ignore_index=True)

            # Grab a batch
            images, (labels, masks) = next(iter(val_loader))
            images = images.to(device)

            # forward pass
            model.eval()
            with torch.no_grad():
                outputs = model(images)
                cams = generate_cam(model.backbone(images), outputs["logits_cls"], model.classification_head.weight)
                pseudo_masks = outputs["pseudo_masks"]

            # Visualize
            # Run model on a validation batch
            images, (labels, masks) = next(iter(val_loader))
            images = images.to(device)

            model.eval()
            with torch.no_grad():
                features = model.backbone(images)
                logits_cls = model.classification_head(features)
                cams = generate_cam(features, logits_cls, model.classification_head.weight)
                outputs = model(images)
                pseudo_masks = outputs["pseudo_masks"]

    results_df.to_csv("grid_search_results.csv", index=False)

            # Visualize both versions
            #visualize_raw_cams_and_pseudo_masks(images, cams, pseudo_masks, num_images = 4)
            #visualize_upsampled_cams_and_pseudo_masks(images, cams, pseudo_masks, num_images = 4)