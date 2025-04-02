import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
import numpy as np
import torchvision.transforms.functional as TF
from transformers import SegformerForSemanticSegmentation


def mask_transform(target):
    """Ensure segmentation masks are converted to a consistent shape."""
    if isinstance(target, (tuple, list)):
        labels, mask = target  # Extract only the segmentation mask
    else:
        mask = target
    
    mask = np.array(mask)  
    # Ensure mask is a single channel
    if len(mask.shape) == 3:  
        mask = mask[..., 0]
    # Resize mask to a fixed size 
    mask = TF.resize(TF.to_pil_image(mask), (224, 224)) 
    mask = np.array(mask).astype(np.int64)

    return labels, torch.from_numpy(mask).long()
    
def download_data(pth: str, split: str):
    assert split in ["trainval", "test"]
    
    # Transform to convert PIL images to tensors
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_dataset = OxfordIIITPet(root=pth, split=split, target_types=["category", "segmentation"], 
                                  download=True, transform=transform, target_transform=mask_transform)
    return train_dataset


class SingleStageWeaklySupervisedModel(nn.Module):
    def __init__(self, backbone: nn.Module, classification_head: nn.Module, segmentation_head: nn.Module, 
                 weight_seg: float = 1.0, weight_cls: float = 1.0):
        """
        Args:
            backbone (nn.Module): Feature extractor 
            classification_head (nn.Module): Classification head 
            segmentation_head (nn.Module): Segmentation head 
            lambda_seg (float): Weight for segmentation loss.
            lambda_cls (float): Weight for classification loss.
        """
        super().__init__()  
        self.backbone = backbone
        self.classification_head = classification_head
        self.segmentation_head = segmentation_head
        self.weight_seg = weight_seg
        self.weight_cls = weight_cls

    def forward(self, images, labels=None):
        """
        Args:
            images (Tensor): Input images of shape (B, C, H, W).
            labels (Tensor, optional): Class labels of shape (B,) for classification.
            masks (Tensor, optional): Ground-truth segmentation masks of shape (B, 1, H, W).
        Returns:
            If in training mode:
                A dict containing the classification loss, segmentation loss, 
                and total loss.
            If in eval mode:
                The predicted segmentation masks or classification logits (or both).
        """
        # Extract features from backbone
        features = self.backbone(images)  # shape depends on backbone

        # Classification forward pass
        logits_cls = self.classification_head(features)  
         # Generate pseudo masks using CAM 
        classifier_weights = self.classification_head.weight  
        pseudo_masks = generate_cam(features, logits_cls, classifier_weights) # CAMs: (B, C, H, W)
        #most simple approach: argmaxing over class labels
        confidence, labels = torch.max(pseudo_masks, dim = 1) ## (B, H, W)

        #create threshold
        ignore_index = 255 #this is defined to set any background label to be ignored in loss calculation 
        threshold = 0.5
        labels[confidence < threshold] = ignore_index  # shape: (B, H, W)
        pseudo_masks = labels
        # Upsample back to original image size. Use mode = 'nearest' to maintain discreate labels (makes it blocky though)
        pseudo_masks = F.interpolate(pseudo_masks.unsqueeze(1).float(), size = (224,224), mode = 'nearest').squeeze(1).long()

        #If the ignore_ratio is too high/low, threshold needs to be changed
        #ignore_ratio = (pseudo_masks == 255).sum() / pseudo_masks.numel()
        #print(f" Ignore pixels in pseudo-mask: {100 * ignore_ratio:.2f}%")

        # Segmentation forward pass
        logits_seg = self.segmentation_head(features)    
        return { "logits_cls": logits_cls, "logits_seg": logits_seg, "pseudo_masks": pseudo_masks}
        

def generate_cam(features, logits_cls, classifier_weights):
    """
    Generate Class Activation Maps (CAMs) from features and classifier weights.

    Args:
        features (Tensor): Feature maps before GAP, shape (B, C, H, W)
        logits_cls (Tensor): Logits from the classification head, shape (B, num_classes)
        classifier_weights (Tensor): Weights of the fully connected classification layer, shape (num_classes, C)
    
    Returns:
        Tensor: CAMs of shape (B, num_classes, H, W)
    """
    # Ensure classifier_weights has shape (num_classes, C)
    features = features[-1]
    assert classifier_weights.shape[1] == features.shape[1], "Classifier weights and feature channels mismatch"

    # Compute A_k = w_k^T f(x) 
    Ak = torch.einsum("bcHW,kc->bkHW", features, classifier_weights)  # Einstein sum performs the dot product

    Ak = F.relu(Ak)

    # Normalize by maximum value per class per image
    Ak_max = Ak.view(Ak.shape[0], Ak.shape[1], -1).max(dim=2, keepdim=True)[0]  # shape: (B, num_classes, 1)
    Ak_max = Ak_max.view(Ak.shape[0], Ak.shape[1], 1, 1)  # Reshape to match (B, num_classes, H, W)
    cam = Ak / (Ak_max + 1e-8)  # Normalize

    return cam



def train_single_stage_model(model, train_loader, val_loader = None, classification_loss_fn = None, 
                             segmentation_loss_fn = None, weightCls=1.0, weightSeg=1.0, optimizer = None, 
                             device="cpu", num_epochs=10, print_every=1):
    """
    Trains a SingleStageWeaklySupervisedModel (or a similar multi-head model).
    
    Args:
        model (nn.Module): Your single-stage weakly supervised model. Should return a dict with "logits_cls" and 
                            "logits_seg" when called in forward pass.
        train_loader (DataLoader): PyTorch DataLoader for training data, expects (images, labels, masks).
        val_loader (DataLoader): DataLoader for validation data. 
        classification_loss_fn (callable, optional): Loss function for classification. Defaults to nn.CrossEntropyLoss() 
                                                        if None.
        segmentation_loss_fn (callable, optional): Loss function for segmentation. Defaults to nn.CrossEntropyLoss() 
                                                        if None.
        lambda_cls (float): Weighting for classification loss. Default: 1.0
        lambda_seg (float): Weighting for segmentation loss. Default: 1.0
        optimizer (torch.optim.Optimizer, optional): Optimizer to use. If None, you must provide one externally.
        device (str): Device to train on. Default: "cuda".
        num_epochs (int): Number of epochs. Default: 10.
        print_every (int): Print training info every X epochs. Default: 1.
    
    Returns:
        nn.Module: The trained model (with final learned parameters).
    """

    # Provide default losses if none given
    if classification_loss_fn is None:
        classification_loss_fn = nn.CrossEntropyLoss()
    if segmentation_loss_fn is None:
        segmentation_loss_fn = nn.CrossEntropyLoss(ignore_index=255)

    # Move model to the specified device
    model = model.to(device)

    # If no optimizer is provided, raise an error (or create a default one)
    if optimizer is None:
        raise ValueError("You must provide an optimizer (e.g. torch.optim.Adam).")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_cls_loss = 0.0
        running_seg_loss = 0.0

        for batch_idx, (images, (labels, masks)) in enumerate(train_loader):
            print(batch_idx)
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            logitsCls = outputs["logits_cls"]  # shape: (B, num_classes)
            logitsSeg = outputs["logits_seg"]  # shape: (B, num_classes, H, W) or (B, 1, H, W)
            pseudoMasks = outputs["pseudo_masks"]

            # Compute classification loss
            lossCls = classification_loss_fn(logitsCls, labels)

            # Compute segmentation loss
            lossSeg = segmentation_loss_fn(logitsSeg, pseudoMasks)

            # Combine losses
            loss = weightCls * lossCls + weightSeg * lossSeg

            # Backprop
            loss.backward()
            optimizer.step()

            # Accumulate statistics
            running_loss += loss.item()
            running_cls_loss += lossCls.item()
            running_seg_loss += lossSeg.item()

        # Print training progress
        if (epoch + 1) % print_every == 0:
            avg_loss = running_loss / len(train_loader)
            avg_cls_loss = running_cls_loss / len(train_loader)
            avg_seg_loss = running_seg_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Cls Loss: {avg_cls_loss:.4f}, "
                  f"Seg Loss: {avg_seg_loss:.4f}")

        # Optional validation step
        if val_loader is not None:
            validate_single_stage_model( model=model, val_loader=val_loader, 
                                        classification_loss_fn=classification_loss_fn, 
                                        segmentation_loss_fn=segmentation_loss_fn, device=device, in_training=True)

    return model

def compute_accuracy(true_labels, pred_labels):
    """Computes accuracy"""
    correct = (true_labels == pred_labels).sum()
    total = len(true_labels)
    return correct / total


def compute_weighted_f1(true_labels, pred_labels, num_classes):
    """Computes the weighted F1 score."""
    f1_scores = []
    weights = []
    for cls in range(num_classes):
    
        tp = np.sum((pred_labels == cls) & (true_labels == cls))  # True positives
        fp = np.sum((pred_labels == cls) & (true_labels != cls))  # False positives
        fn = np.sum((pred_labels != cls) & (true_labels == cls))  # False negatives
        
        # Precision and Recall 
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1 score for current class
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
        
        # Weight by proportion of samples of this class in the true labels
        weight = np.sum(true_labels == cls) / len(true_labels)
        weights.append(weight)
    
    # Compute the weighted F1 score
    weighted_f1 = np.sum([w * f1 for w, f1 in zip(weights, f1_scores)])
    return weighted_f1

def compute_iou(y_true, y_pred, num_classes):
    """
    Compute the Intersection over Union (IoU) for each class and return the mean IoU.
    
    Args:
        y_true (np.array): Ground truth masks (flattened).
        y_pred (np.array): Predicted masks (flattened).
        num_classes (int): Number of segmentation classes.
    
    Returns:
        iou_per_class (dict): IoU per class.
        mean_iou (float): Mean IoU across classes.
    """
    iou_per_class = {}
    ious = []

    for class_idx in range(num_classes):
        true_mask = (y_true == class_idx)
        pred_mask = (y_pred == class_idx)

        intersection = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()

        if union == 0:
            iou = float("nan")  # Ignore class if it doesn't exist in either prediction or ground truth
        else:
            iou = intersection / union

        iou_per_class[class_idx] = iou
        if not np.isnan(iou):
            ious.append(iou)

    mean_iou = np.nanmean(ious)  # Compute mean IoU, ignoring NaN classes
    return iou_per_class, mean_iou

def validate_single_stage_model( model, val_loader, classification_loss_fn, segmentation_loss_fn, 
                                in_training = False, device="cuda"):
    """
    A simple validation function to evaluate classification and segmentation losses
    on the validation set.
    """
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_seg_loss = 0.0
    all_true_labels = []
    all_pred_labels = []
    all_seg_true = []
    all_seg_pred = []

    with torch.no_grad():
        for images, (labels, masks) in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            outputs = model(images)
            logits_cls = outputs["logits_cls"]
            logits_seg = outputs["logits_seg"]

            # Compute losses
            loss_cls = classification_loss_fn(logits_cls, labels)
            loss_seg = segmentation_loss_fn(logits_seg, masks)
            loss = loss_cls + loss_seg

            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_seg_loss += loss_seg.item()

            # Get classification predictions
            preds = torch.argmax(logits_cls, dim=1)
            all_true_labels.append(labels.cpu().numpy())
            all_pred_labels.append(preds.cpu().numpy())

            # Get segmentation predictions
            preds_seg = torch.argmax(logits_seg, dim=1)  # (B, H, W)
            all_seg_true.append(masks.cpu().numpy())
            all_seg_pred.append(preds_seg.cpu().numpy())


    
    # Average losses
    avg_loss = total_loss / len(val_loader)
    avg_cls_loss = total_cls_loss / len(val_loader)
    avg_seg_loss = total_seg_loss / len(val_loader)

    print(f"[Validation] Total Loss: {avg_loss:.4f}, Cls Loss: {avg_cls_loss:.4f}, Seg Loss: {avg_seg_loss:.4f}")
    if in_training == True:
        return 
    
    all_true_labels = np.concatenate(all_true_labels)
    all_pred_labels = np.concatenate(all_pred_labels)

    all_seg_true = np.concatenate(all_seg_true).ravel()  # flatten for pixel-wise evaluation
    all_seg_pred = np.concatenate(all_seg_pred).ravel()

    # Calculate accuracy and weighted F1 score
    cls_accuracy = compute_accuracy(all_true_labels, all_pred_labels)
    num_classes = logits_cls.shape[1]  
    cls_weighted_f1 = compute_weighted_f1(all_true_labels, all_pred_labels, num_classes)


    # Segmentation metrics (pixel-wise)
    # Here, num_seg_classes is assumed to be the number of channels in logits_seg.
    num_seg_classes = logits_seg.shape[1]
    seg_accuracy = compute_accuracy(all_seg_true, all_seg_pred)
    seg_weighted_f1 = compute_weighted_f1(all_seg_true, all_seg_pred, num_seg_classes)

    # Compute IoU
    iou_per_class, mean_iou = compute_iou(all_seg_true, all_seg_pred, num_seg_classes)

    print(f"[Validation] Classification -> Accuracy: {cls_accuracy*100:.2f}%, Weighted F1: {cls_weighted_f1:.4f}")
    print(f"[Validation] Segmentation   -> Pixel Accuracy: {seg_accuracy*100:.2f}%, Weighted F1: {seg_weighted_f1:.4f}, mIoU: {mean_iou:.4f}")

    return {
        "total_loss": avg_loss,
        "cls_loss": avg_cls_loss,
        "seg_loss": avg_seg_loss,
        "cls_accuracy": cls_accuracy,
        "cls_f1": cls_weighted_f1,
        "seg_accuracy": seg_accuracy,
        "seg_f1": seg_weighted_f1,
        "mean_iou": mean_iou,
        "iou_per_class": iou_per_class
    }

def random_search(params: dict, num_trials: int):
    """ Random search over parameters
    
    Args:
        params (dict): Dictionary of parameters to search over. Keys are the name of parameters to search over,
                       and values are lists of the values that the parameter can take.
        num_trials (int): Number of random trials to perform
        
    Returns:
        dict: Dictionary of best hyperparameters from search along with the corresponding validation loss.
    """
    best_paras = None
    best_loss = float('inf')
    
    # Loop for the specified number of trials
    for trial in range(num_trials):
        # Sample hyperparameters from the provided lists
        seg_head = np.random.choice(params["segmentation_head"]) if "segmentation_head" in params else None
        cls_head = np.random.choice(params["classification_head"]) if "classification_head" in params else None
        backbone = np.random.choice(params["backbone"]) if "backbone" in params else None
        lr = np.random.choice(params["lr"]) if "lr" in params else 1e-4
        weightSeg = np.random.choice(params["weightSeg"]) if "weightSeg" in params else 1.0
        weightCls = np.random.choice(params["weightCls"]) if "weightCls" in params else 1.0
        num_epochs = np.random.choice(params["num_epochs"]) if "num_epochs" in params else 5
        device = params["device"] if "device" in params else "cpu"
        
        # Use default modules if not provided via the params dictionary
        if backbone is None:
            from __main__ import SimpleBackbone  # or import from your module
            backbone = SimpleBackbone()
        if cls_head is None:
            from __main__ import SimpleClassificationHead
            cls_head = SimpleClassificationHead(in_channels=64, num_classes=37)
        if seg_head is None:
            from __main__ import SimpleSegmentationHead
            seg_head = SimpleSegmentationHead(in_channels=64, num_seg_classes=2)
        
        # Build the model with the sampled hyperparameters
        from __main__ import SingleStageWeaklySupervisedModel
        model = SingleStageWeaklySupervisedModel(backbone=backbone, classification_head=cls_head, 
                                                segmentation_head=seg_head, weight_seg=weightSeg, weight_cls=weightCls)
        
        # Set up the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Train the model using the provided training function.
        # (Assuming train_single_stage_model is defined and imported from __main__.)
        model = train_single_stage_model(model=model, train_loader=params["train_loader"], val_loader=params["val_loader"],
                                         classification_loss_fn=params.get("classification_loss_fn", nn.BCEWithLogitsLoss()),
                                         segmentation_loss_fn=params.get("segmentation_loss_fn", nn.CrossEntropyLoss()),
                                         weightCls=weightCls, weightSeg=weightSeg, optimizer=optimizer, device=device, num_epochs=num_epochs, 
                                         print_every=1)
        
        # Evaluate the trained model on the validation set
        val_loss = validate_single_stage_model(
            model,
            params["val_loader"],
            classification_loss_fn=params.get("classification_loss_fn", nn.BCEWithLogitsLoss()),
            segmentation_loss_fn=params.get("segmentation_loss_fn", nn.CrossEntropyLoss()),
            device=device
        )
        
        print(f"Trial {trial+1}: Validation loss = {val_loss:.4f}")
        
        # Update best hyperparameters if this trial is better
        if val_loss < best_loss:
            best_loss = val_loss
            best_paras = {
                "segmentation_head": seg_head,
                "classification_head": cls_head,
                "backbone": backbone,
                "lr": lr,
                "weightSeg": weightSeg,
                "weightCls": weightCls,
                "num_epochs": num_epochs,
                "validation_loss": val_loss
            }
    
    return best_paras
    
    
class SimpleSegmentationHead(nn.Module):
    def __init__(self, in_channels=64, num_seg_classes=37):
        """
        A simple segmentation head with one conv block.
        Inspired by baseline designs in [Wei et al., 2018] (https://arxiv.org/abs/1509.03150).
        
        Args:
            in_channels (int): Number of channels from the backbone.
            num_seg_classes (int): Number of segmentation classes.
        """
        super(SimpleSegmentationHead, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Conv2d(32, num_seg_classes, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        
    def forward(self, x):
        x = self.conv_block(x)
        x = self.classifier(x)
        x = self.upsample(x)
        return x

class AlternativeSegmentationHead1(nn.Module):
    def __init__(self, in_channels=64, num_seg_classes=37):
        """
        A deeper segmentation head with two convolutional layers.
        Inspired by designs in [Jo et al., 2022](https://arxiv.org/abs/2204.06754).
        
        Args:
            in_channels (int): Number of channels from the backbone.
            num_seg_classes (int): Number of segmentation classes.
        """
        super(AlternativeSegmentationHead1, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Conv2d(32, num_seg_classes, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        x = self.upsample(x)
        return x

class AlternativeSegmentationHead2(nn.Module):
    def __init__(self, in_channels=64, num_seg_classes=37, atrous_rates=(6, 12, 18)):
        """
        A segmentation head that uses parallel atrous convolutions (ASPP-style).
        Inspired by DeepLab v3 [Chen et al., 2017](https://arxiv.org/abs/1706.05587) and 
        adaptations for weakly-supervised segmentation [Zhang et al., 2022](https://link.springer.com/article/10.1007/s11263-022-01590-z).
        
        Args:
            in_channels (int): Number of channels from the backbone.
            num_seg_classes (int): Number of segmentation classes.
            atrous_rates (tuple): Dilation rates for the parallel convolutions.
        """
        super(AlternativeSegmentationHead2, self).__init__()
        # A 1x1 convolution branch
        self.branch1 = nn.Conv2d(in_channels, 32, kernel_size=1)
        
        # Atrous convolution branches
        self.branches = nn.ModuleList([
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=rate, dilation=rate)
            for rate in atrous_rates
        ])
        
        # concatenate all branches and reduce channels
        self.fuse = nn.Sequential(
            nn.Conv2d(32 * (1 + len(atrous_rates)), 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Conv2d(32, num_seg_classes, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branches = [branch1] + [branch(x) for branch in self.branches]
        x = torch.cat(branches, dim=1)
        x = self.fuse(x)
        x = self.classifier(x)
        x = self.upsample(x)
        return x

# Simple implementations for backbone and classification head.
class SimpleBackbone(nn.Module):
    def __init__(self):
        super(SimpleBackbone, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
            # nn.MaxPool2d(2)
        )
    def forward(self, x):
        return self.features(x)

class SimpleClassificationHead(nn.Module):
    def __init__(self, in_channels=64, num_classes=37):
        super(SimpleClassificationHead, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)
        # Expose weights for CAM generation
        self.weight = self.fc.weight  
    def forward(self, x):
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    

class SegFormerDecoderHead(nn.Module):
    def __init__(self, decoder, upsample_scale=4):
        """
        Wraps a SegFormer decoder and classifier as a segmentation head.
        
        Args:
            decoder (nn.Module): The pretrained SegFormer decoder.
            upsample_scale (int): Factor to upsample the output logits to match input resolution.
        """
        super(SegFormerDecoderHead, self).__init__()
        self.decoder = decoder
        self.upsample_scale = upsample_scale

    def forward(self, features):
        """
        Args:
            features (List[Tensor]): A list of feature maps from your backbone.
                                      They should match the expected input of the decoder.
        Returns:
            Tensor: The upsampled per-pixel class logits.
        """
        # The decoder expects a list of features, for example [f1, f2, f3, f4]
        logits = self.decoder(features)
        # Upsample to the desired resolution (e.g., original image size)
        logits = F.interpolate(logits, scale_factor=self.upsample_scale, mode='bilinear', align_corners=False)
        return logits


class SegFormerHead(nn.Module):
    def __init__(self, in_channels_list=[16, 32, 64, 128], embed_dim=256, num_seg_classes=37, upsample_scale=8):
        """
        SegFormer-style segmentation head with multi-level feature fusion.

        Args:
            in_channels_list (list): List of input channel sizes from different backbone stages.
            embed_dim (int): Dimensionality of the embedding.
            num_seg_classes (int): Number of segmentation classes.
            upsample_scale (int): Upsampling factor to match input resolution.
        """
        super(SegFormerHead, self).__init__()
        
        # 1x1 conv projections for each input feature map
        self.projections = nn.ModuleList([
            nn.Conv2d(in_ch, embed_dim, kernel_size=1) for in_ch in in_channels_list
        ])
        
        # Fusion layer (simple sum or concat + Conv1x1)
        self.fusion = nn.Conv2d(embed_dim * len(in_channels_list), embed_dim, kernel_size=1)
        
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Conv2d(embed_dim, num_seg_classes, kernel_size=1)
        self.upsample_scale = upsample_scale

    def forward(self, features):
        """
        Args:
            features (list of tensors): Feature maps from backbone [f1, f2, f3, f4]
        
        Returns:
            Tensor: Per-pixel class logits (B, num_seg_classes, H, W)
        """
        # Apply individual projections
        projected_features = [proj(f) for proj, f in zip(self.projections, features)]
        
        # Resize all features to match the highest resolution (f1)
        target_size = projected_features[0].shape[2:]  # Get H, W of f1
        projected_features = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
                              for f in projected_features]

        # Fuse all projected feature maps (concatenation + 1x1 conv)
        fused_features = torch.cat(projected_features, dim=1)
        fused_features = self.fusion(fused_features)  

        # Classification
        x = self.dropout(fused_features)
        x = self.classifier(x)

        # Upsample to match the original input resolution
        #x = F.interpolate(x, scale_factor=self.upsample_scale, mode='bilinear', align_corners=False)
        return x
    
# Segformer expects a list of features, hence we need to change backbone to provide this and the simple classifier to be
# able to handle this
class SegFormerBackbone(nn.Module):
    def __init__(self):
        super(SegFormerBackbone, self).__init__()
        # Stage 1: Initial low-level features, full resolution.
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        # Stage 2: Downsample by 2.
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # Stage 3: Downsample further by 2 (total factor 4).
        self.stage3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        # Each stage returns a feature map at a different resolution.
        f1 = self.stage1(x)    # Resolution: H x W, channels: 16
        f2 = self.stage2(f1)   # Resolution: H/2 x W/2, channels: 32
        f3 = self.stage3(f2)   # Resolution: H/4 x W/4, channels: 64
        return [f1, f2, f3]
    
class SimpleSegFormerClassificationHead(nn.Module):
    def __init__(self, in_channels=64, num_classes=37):
        super(SimpleSegFormerClassificationHead, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)
        # Expose weights for CAM generation
        self.weight = self.fc.weight  
    def forward(self, x):
        # Use the classification head on the highest-resolution feature (example)
        x = self.gap(x[-1])
        x = x.view(x.size(0), -1)
        return self.fc(x)


def test_segmentation_heads(train_loader, val_loader, num_epochs=5, lr=1e-4, device="cuda"):
    # Fixed backbone and classification head for these experiments
    backbone = SegFormerBackbone().to(device)
    # Freeze the backbone parameters
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    classification_head = SimpleSegFormerClassificationHead(in_channels=64, num_classes=37).to(device)
    
    # Define the segmentation heads to test along with names for identification
    # segmentation_heads = [
    #     ("SimpleSegmentationHead", SimpleSegmentationHead(in_channels=64, num_seg_classes=37)),
    #     ("AlternativeSegmentationHead1", AlternativeSegmentationHead1(in_channels=64, num_seg_classes=37)),
    #     ("AlternativeSegmentationHead2", AlternativeSegmentationHead2(in_channels=64, num_seg_classes=37))
    # ]

    num_labels = 37
    id2label = {str(i): f"class_{i}" for i in range(num_labels)}
    label2id = {v: k for k, v in id2label.items()}

    full_segformer = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # Extract the decoder and classifier.
    # In Hugging Face's implementation, these are typically stored as attributes:
    # pretrained_decoder = full_segformer.decode_head

    segmentation_heads = [("SegformerDecoderHead", SegFormerHead([16, 32, 64], embed_dim = 64))]
    
    results = {}
    
    for head_name, seg_head in segmentation_heads:
        print(f"\nTesting segmentation head: {head_name}")
        
        # Build the model using the fixed backbone and classification head and the current segmentation head.
        model = SingleStageWeaklySupervisedModel(
            backbone=backbone,
            classification_head=classification_head,
            segmentation_head=seg_head,
            weight_seg=1.0,
            weight_cls=1.0
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Train the model with the current segmentation head.
        model = train_single_stage_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            classification_loss_fn=nn.CrossEntropyLoss(),
            segmentation_loss_fn=nn.CrossEntropyLoss(ignore_index=255),
            weightCls=1.0,
            weightSeg=1.0,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            print_every=1
        )
        
        # Validate the model.
        metrics = validate_single_stage_model(
            model=model,
            val_loader=val_loader,
            classification_loss_fn=nn.CrossEntropyLoss(),
            segmentation_loss_fn=nn.CrossEntropyLoss(ignore_index=255),
            device=device
        )
        
        # Store segmentation metrics
        seg_acc = metrics.get("seg_accuracy", None)
        seg_f1 = metrics.get("seg_f1", None)
        seg_mIoU = metrics.get("mIoU", None)
        results[head_name] = {
            "total_loss": metrics["total_loss"],
            "seg_loss": metrics["seg_loss"],
            "seg_accuracy": seg_acc,
            "seg_f1": seg_f1
        }
        print(f"Segmentation Head {head_name}: Pixel Accuracy = {seg_acc*100:.2f}%, Weighted F1 = {seg_f1:.4f}, mIoU: {seg_mIoU:.4f}")
    
    # Print summary for all segmentation heads
    print("\n--- Summary of Segmentation Heads ---")
    for head_name, metrics in results.items():
        print(f"{head_name}: Loss = {metrics['total_loss']:.4f}, Seg Loss = {metrics['seg_loss']:.4f}, "
              f"Pixel Accuracy = {metrics['seg_accuracy']*100:.2f}%, Weighted F1 = {metrics['seg_f1']:.4f}")
    
    return results