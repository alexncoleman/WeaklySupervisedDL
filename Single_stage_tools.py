import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
import numpy as np



def download_data(pth: str, split: str):
    assert split in ["trainval", "test"]
    
    # Transform to convert PIL images to tensors
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_dataset = OxfordIIITPet(root=pth, split=split, target_types=["category", "segmentation"], 
                                  download=True, transform=transform)

    return train_dataset


class SingleStageWeaklySupervisedModel(nn.Module):
    def __init__(self, backbone: nn.Module, classification_head: nn.Module, segmentation_head: nn.Module, 
                 weight_seg: float = 1.0, weight_cls: float = 1.0):
        """
        Args:
            backbone (nn.Module): Feature extractor (e.g., ResNet, VGG).
            classification_head (nn.Module): Classification head that takes 
                                             the backbone features and outputs 
                                             class logits.
            segmentation_head (nn.Module): Segmentation head that takes 
                                           the backbone features and outputs 
                                           segmentation masks.
            lambda_seg (float): Weight for segmentation loss.
            lambda_cls (float): Weight for classification loss.
        """
        super().__init__()
        self.backbone = backbone
        self.classification_head = classification_head
        self.segmentation_head = segmentation_head
        self.weight_seg = weight_seg
        self.weight_cls = weight_cls

    def forward(self, images, labels=None, masks=None):
        """
        Args:
            images (Tensor): Input images of shape (B, C, H, W).
            labels (Tensor, optional): Class labels of shape (B,) for classification.
            masks (Tensor, optional): Ground-truth segmentation masks of shape (B, 1, H, W).
                                      In a weakly supervised setting, you may not always have 
                                      these, or you might only have partial or noisy masks.
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
        logits_cls = self.classification_head(features)  # (B, num_classes)
         # Generate pseudo masks using CAM 
        classifier_weights = self.classification_head.weight  
        pseudo_masks = generate_cam(features, logits_cls, classifier_weights) ## CAMs: (B, C, H, W)
        #most simple approach: argmaxing over class labels
        confidence, labels = torch.max(pseudo_masks, dim = 1) ## (B, H, W)

        #OPTIONAL: create threshold

        ignore_index = 255 #this is defined to set any background label to be ignored in loss calculation (Make sure this is consistent)
        threshold = 0.25
        labels[confidence < threshold] = ignore_index  # shape: (B, H, W)
        pseudo_masks = labels

        #If the ignore_ratio is too high/low, threshold needs to be changed
        ignore_ratio = (pseudo_masks == 255).sum() / pseudo_masks.numel()
        print(f" Ignore pixels in pseudo-mask: {100 * ignore_ratio:.2f}%")

        # Segmentation forward pass
        logits_seg = self.segmentation_head(features)    # (B, num_classes, H, W) or (B, 1, H, W)

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
    assert classifier_weights.shape[1] == features.shape[1], "Classifier weights and feature channels mismatch"

    # Compute A_k = w_k^T f(x) 
    Ak = torch.einsum("bcHW,kc->bkHW", features, classifier_weights)  # Einstein sum performs the dot product

    # Apply ReLU
    Ak = F.relu(Ak)

    # Normalize by maximum value per class per image
    Ak_max = Ak.view(Ak.shape[0], Ak.shape[1], -1).max(dim=2, keepdim=True)[0]  # (B, num_classes, 1)
    cam = Ak / (Ak_max + 1e-8)  # Avoid division by zero

    return cam



def train_single_stage_model(model, train_loader, val_loader = None, classification_loss_fn = None, 
                             segmentation_loss_fn = None, weightCls=1.0, weightSeg=1.0, optimizer = None, 
                             device="cpu", num_epochs=10, print_every=1):
    """
    Trains a SingleStageWeaklySupervisedModel (or a similar multi-head model).
    
    Args:
        model (nn.Module): Your single-stage weakly supervised model. 
                           Should return a dict with "logits_cls" and "logits_seg" 
                           when called in forward pass.
        train_loader (DataLoader): PyTorch DataLoader for training data. 
                                   Expects (images, labels, masks) or similar.
        val_loader (DataLoader, optional): DataLoader for validation data. 
                                           If provided, will run a validation step.
        classification_loss_fn (callable, optional): Loss function for classification.
            Defaults to nn.CrossEntropyLoss() if None.
        segmentation_loss_fn (callable, optional): Loss function for segmentation.
            Defaults to nn.CrossEntropyLoss() if None.
        lambda_cls (float): Weighting for classification loss. Default: 1.0
        lambda_seg (float): Weighting for segmentation loss. Default: 1.0
        optimizer (torch.optim.Optimizer, optional): Optimizer to use. If None, 
                                                     you must provide one externally.
        device (str): Device to train on ("cpu" or "cuda"). Default: "cuda".
        num_epochs (int): Number of training epochs. Default: 10.
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

        for batch_idx, (images, labels, masks) in enumerate(train_loader):
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
                                        segmentation_loss_fn=segmentation_loss_fn, device=device)

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



def validate_single_stage_model( model, val_loader, classification_loss_fn, segmentation_loss_fn, 
                                device="cuda"):
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
        for images, labels, masks in val_loader:
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

    print(f"[Validation] Total Loss: {avg_loss:.4f}, Cls Loss: {avg_cls_loss:.4f}, Seg Loss: {avg_seg_loss:.4f}")
    print(f"[Validation] Classification -> Accuracy: {cls_accuracy*100:.2f}%, Weighted F1: {cls_weighted_f1:.4f}")
    print(f"[Validation] Segmentation   -> Pixel Accuracy: {seg_accuracy*100:.2f}%, Weighted F1: {seg_weighted_f1:.4f}")

    return {
        "total_loss": avg_loss,
        "cls_loss": avg_cls_loss,
        "seg_loss": avg_seg_loss,
        "cls_accuracy": cls_accuracy,
        "cls_f1": cls_weighted_f1,
        "seg_accuracy": seg_accuracy,
        "seg_f1": seg_weighted_f1
    }


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
    
    