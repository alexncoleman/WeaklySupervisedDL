import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader



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
        pseudo_masks = generate_cam(features, logits_cls, classifier_weights)

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
        classification_loss_fn = nn.BCEWithLogitsLoss()
    if segmentation_loss_fn is None:
        segmentation_loss_fn = nn.CrossEntropyLoss()

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



def validate_single_stage_model( model, val_loader, classification_loss_fn, segmentation_loss_fn, 
                                device="cuda"):
    """
    A simple validation function to evaluate classification and segmentation losses
    on the validation set.
    """
    model.eval()
    val_loss = 0.0
    val_cls_loss = 0.0
    val_seg_loss = 0.0
    with torch.no_grad():
        for images, labels, masks in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            outputs = model(images)
            logits_cls = outputs["logits_cls"]
            logits_seg = outputs["logits_seg"]

            loss_cls = classification_loss_fn(logits_cls, labels)
            loss_seg = segmentation_loss_fn(logits_seg, masks)
            loss = loss_cls + loss_seg

            val_loss += loss.item()
            val_cls_loss += loss_cls.item()
            val_seg_loss += loss_seg.item()

    val_loss /= len(val_loader)
    val_cls_loss /= len(val_loader)
    val_seg_loss /= len(val_loader)

    print(f"[Validation] Total Loss: {val_loss:.4f}, "
          f"Cls Loss: {val_cls_loss:.4f}, Seg Loss: {val_seg_loss:.4f}")
    
