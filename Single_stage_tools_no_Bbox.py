import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
import numpy as np

# imports to visualise pseudo mask and CAMs
# matplotlib is not included in cw1-pt env so make sure to get rid in submission (only used for testing purposes for now)
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision.transforms as T

def pet_target_transform(target):
    label, mask = target
    label_bin = 0 if label == 1 else 1  # 0 = cat, 1 = dog
    mask = transforms.Resize((224, 224))(mask)
    mask = transforms.PILToTensor()(mask).squeeze(0).long()

    # Convert mask:
    # 1 → pet → set to 1 (cat) or 2 (dog) based on label
    # 2 → background → set to 0
    # 3 → border → set to 255 (ignored)
    new_mask = torch.zeros_like(mask)
    new_mask[mask == 1] = 1 + label_bin  # cat → 1, dog → 2
    new_mask[mask == 2] = 0              # background
    new_mask[mask == 3] = 255            # ignore label

    return label_bin, new_mask

def compute_dynamic_threshold(confidence_map, target_ignore_ratio=0.5):
    """
    Computes a threshold such that a given percentage of pixels fall below it.
    Args:
        confidence_map: Tensor of shape (B, H, W)
        target_ignore_ratio: float between 0 and 1
    Returns:
        threshold: float
    """
    flattened = confidence_map.view(-1)
    sorted_vals, _ = torch.sort(flattened)
    index = int(len(sorted_vals) * target_ignore_ratio)
    threshold = sorted_vals[index]
    return threshold.item()


def compute_mean_iou(true_masks, pred_masks, num_classes, ignore_index=255):
    """Compute mean Intersection-over-Union (mIoU)"""
    iou_per_class = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue

        pred_inds = (pred_masks == cls)
        target_inds = (true_masks == cls)

        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()

        if union == 0:
            iou = float('nan')
        else:
            iou = intersection.float() / union.float()

        iou_per_class.append(iou.item())

    # Return mean IoU (ignoring NaNs)
    return np.nanmean(iou_per_class)


def apply_gaussian_smoothing(cam_tensor, kernel_size=7, sigma=2.0):
    """
    Applies Gaussian smoothing to a batch of CAMs.
    Args:
        cam_tensor: Tensor of shape (B, C, H, W)
        kernel_size: Size of the Gaussian kernel
        sigma: Standard deviation of the Gaussian
    Returns:
        Smoothed tensor of same shape
    """
    # Normalize CAMs to [0, 1] for each image & class
    B, C, H, W = cam_tensor.shape
    cam_tensor = cam_tensor.reshape(B * C, 1, H, W)
    
    gaussian_blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    smoothed = gaussian_blur(cam_tensor)

    smoothed = smoothed.view(B, C, H, W)
    return smoothed

def download_data(pth: str, split: str):
    assert split in ["trainval", "test"]
    
    # Transform to convert PIL images to tensors
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    
    train_dataset = OxfordIIITPet(root=pth, split=split, target_types=["binary-category", "segmentation"], 
                                  download=True, transform=transform, target_transform = pet_target_transform)

    return train_dataset


class SingleStageWeaklySupervisedModel(nn.Module):
    def __init__(self, backbone: nn.Module, classification_head: nn.Module, segmentation_head: nn.Module, 
                 weight_seg: float = 1.0, weight_cls: float = 1.0, target_threshold = 0.3):
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
        self.target_threshold = target_threshold

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
  
        feats = self.backbone.features(images)  # List of intermediate features

        feat0 = feats[-1]  # 7×7
        feat1 = feats[-2]  # 14×14
        feat2 = feats[-3]  # 28×28
        feat3 = feats[-4]  # 56×56 ← shallower 

        logits_cls = self.classification_head(feat0)
        classifier_weights = self.classification_head.weight  # shape (num_classes, C)

        W = self.classification_head.weight
        cam0 = generate_cam(feat0, logits_cls, W[:, :feat0.shape[1]])
        cam1 = generate_cam(feat1, logits_cls, W[:, :feat1.shape[1]])
        cam2 = generate_cam(feat2, logits_cls, W[:, :feat2.shape[1]])
        cam3 = generate_cam(feat3, logits_cls, W[:, :feat3.shape[1]]) 

        # Upsample all to input resolution
        cam0_up = F.interpolate(cam0, size=(224, 224), mode='bilinear', align_corners=False)
        cam1_up = F.interpolate(cam1, size=(224, 224), mode='bilinear', align_corners=False)
        cam2_up = F.interpolate(cam2, size=(224, 224), mode='bilinear', align_corners=False)
        cam3_up = F.interpolate(cam3, size=(224, 224), mode='bilinear', align_corners=False)

        # Apply smoothing
        cam0_up = apply_gaussian_smoothing(cam0_up, kernel_size=7, sigma=2.0)
        cam1_up = apply_gaussian_smoothing(cam1_up, kernel_size=7, sigma=2.0)
        cam2_up = apply_gaussian_smoothing(cam2_up, kernel_size=7, sigma=2.0)
        cam3_up = apply_gaussian_smoothing(cam3_up, kernel_size=7, sigma=2.0)

        # Weighted fusion
        cam_combined = (
            0.1 * cam0_up +
            0.2 * cam1_up +
            0.4 * cam2_up +
            0.3 * cam3_up
        )

        #-------------------

        # Create pseudo masks from combined CAMs
        confidence, labels = torch.max(cam_combined, dim=1)
        ignore_index = 255
        threshold = compute_dynamic_threshold(confidence, target_ignore_ratio=self.target_threshold)
        labels[confidence < threshold] = ignore_index  # weak predictions set to ignore
        pseudo_masks = labels

    
        # Segmentation head takes feat1 (or fuse if you want more context)
        logits_seg = self.segmentation_head(feat1)

        # Print ignore ratio for debugging
        ignore_ratio = (pseudo_masks == ignore_index).sum() / pseudo_masks.numel()
        #print(f" Ignore pixels in pseudo-mask: {100 * ignore_ratio:.2f}%")

        return {
            "logits_cls": logits_cls,
            "logits_seg": logits_seg,
            "pseudo_masks": pseudo_masks
        }

            

def generate_cam(features, logits_cls, classifier_weights, apply_smoothing = True, sigma = 1.0):
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
    #Ak_max = Ak.view(Ak.shape[0], Ak.shape[1], -1).max(dim=2, keepdim=True)[0]  # (B, num_classes, 1)
    Ak_max = Ak.view(Ak.shape[0], Ak.shape[1], -1).max(dim=2)[0].view(Ak.shape[0], Ak.shape[1], 1, 1)
    cam = Ak / (Ak_max + 1e-8)  # Avoid division by zero

    return cam



def train_single_stage_model(model, train_loader, val_loader = None, classification_loss_fn = None, 
                             segmentation_loss_fn = None, weightCls=1.0, weightSeg=1.0, optimizer = None, 
                             device="cpu", num_epochs=10, print_every=1, initial_threshold = 0.5, threshold_type = None, decay_rate = 0.5):
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

        if threshold_type == 'exp_decay': 
            dynamic_threshold = initial_threshold * (decay_rate ** epoch)
            model.target_threshold = dynamic_threshold

        elif threshold_type == "linear_decay": 
            dynamic_threshold = initial_threshold * (1 - epoch / num_epochs)
            model.target_threshold = dynamic_threshold

        elif threshold_type == "constant": 
            dynamic_threshold = initial_threshold 
            model.target_threshold = dynamic_threshold

        running_loss = 0.0
        running_cls_loss = 0.0
        running_seg_loss = 0.0

        for batch_idx, (images, (labels, masks)) in enumerate(train_loader):
        #for images, (labels, masks) in enumerate(train_loader):
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
            val_metrics = validate_single_stage_model( model=model, val_loader=val_loader, 
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
       for batch_idx, (images, targets) in enumerate(val_loader):
            labels, masks = targets
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


    mean_iou = compute_mean_iou(torch.tensor(all_seg_true), torch.tensor(all_seg_pred), num_seg_classes)


    print(f"[Validation] Total Loss: {avg_loss:.4f}, Cls Loss: {avg_cls_loss:.4f}, Seg Loss: {avg_seg_loss:.4f}")
    print(f"[Validation] Classification -> Accuracy: {cls_accuracy*100:.2f}%, Weighted F1: {cls_weighted_f1:.4f}")
    print(f"[Validation] Segmentation   -> Pixel Accuracy: {seg_accuracy*100:.2f}%, Weighted F1: {seg_weighted_f1:.4f}")
    print(f"[Validation] Mean IoU: {mean_iou:.4f}")

    return {
        "total_loss": avg_loss,
        "cls_loss": avg_cls_loss,
        "seg_loss": avg_seg_loss,
        "cls_accuracy": cls_accuracy,
        "cls_f1": cls_weighted_f1,
        "seg_accuracy": seg_accuracy,
        "seg_f1": seg_weighted_f1
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
            cls_head = SimpleClassificationHead(in_channels=64, num_classes=2)
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


def visualize_raw_cams_and_pseudo_masks(images, cams, pseudo_masks, num_images=4):
    """
    Visualizes CAMs and pseudo masks at their original (low) resolution.
    """

    images = images.cpu()
    cams = cams.cpu()
    pseudo_masks = pseudo_masks.cpu()

    cam_max = torch.max(cams, dim=1)[0]  # (B, H_low, W_low)

    for i in range(min(num_images, images.shape[0])):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        # Original image
        axs[0].imshow(T.ToPILImage()(images[i]))
        axs[0].set_title("Input Image")
        axs[0].axis("off")

        # Raw CAM
        axs[1].imshow(cam_max[i], cmap='jet')
        axs[1].set_title("Raw CAM (low-res)")
        axs[1].axis("off")

        # Raw pseudo-mask
        axs[2].imshow(pseudo_masks[i], cmap='viridis')
        axs[2].set_title("Raw Pseudo Mask")
        axs[2].axis("off")

        plt.tight_layout()
        plt.show()

def visualize_upsampled_cams_and_pseudo_masks(images, cams, pseudo_masks, num_images=4):
    """
    Visualizes CAMs and pseudo masks after upsampling to input resolution (e.g. 224x224).
    """

    images = images.cpu()
    cams = cams.cpu()
    pseudo_masks = pseudo_masks.cpu()

    # Upsample CAMs
    cams_upsampled = F.interpolate(cams, size=(224, 224), mode='bilinear', align_corners=False)
    cam_max = torch.max(cams_upsampled, dim=1)[0]  # (B, 224, 224)

    # Upsample pseudo-masks
    pseudo_masks = pseudo_masks.unsqueeze(1).float()  # (B, 1, H, W)
    pseudo_masks_upsampled = F.interpolate(pseudo_masks, size=(224, 224), mode='bilinear').squeeze(1)

    for i in range(min(num_images, images.shape[0])):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        # Original image
        axs[0].imshow(T.ToPILImage()(images[i]))
        axs[0].set_title("Input Image")
        axs[0].axis("off")

        # Upsampled CAM over image
        axs[1].imshow(T.ToPILImage()(images[i]))
        axs[1].imshow(cam_max[i], cmap='jet', alpha=0.5)
        axs[1].set_title("Upsampled CAM")
        axs[1].axis("off")

        # Upsampled Pseudo Mask
        axs[2].imshow(pseudo_masks_upsampled[i], cmap='viridis')
        axs[2].set_title("Upsampled Pseudo Mask")
        axs[2].axis("off")

        plt.tight_layout()
        plt.show()
    
def decode_segmentation_mask(mask):
    """
    Convert segmentation mask (H, W) with labels {0, 1, 2} into RGB image (H, W, 3)
    Colors:
        - Background (0): Purple  → [128, 0, 128]
        - Cat        (1): Yellow  → [255, 255, 0]
        - Dog        (2): Green   → [0, 255, 0]
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    rgb[mask == 0] = [128, 0, 128]   # Purple
    rgb[mask == 1] = [255, 255, 0]   # Yellow
    rgb[mask == 2] = [0, 255, 0]     # Green

    return rgb

def visualize_segmentation_predictions(images, preds_seg, masks=None, num_images=4):
    """
    Visualize model predictions (and ground truth if available) on input images.
    
    Args:
        images (Tensor): Input images of shape (B, 3, H, W)
        preds_seg (Tensor): Predicted segmentation masks (B, H, W)
        masks (Tensor, optional): Ground-truth segmentation masks (B, H, W)
        num_images (int): Number of images to show
    """
    images = images.cpu()
    preds_seg = preds_seg.cpu()
    if masks is not None:
        masks = masks.cpu()

    for i in range(min(num_images, images.size(0))):
        fig, axs = plt.subplots(1, 3 if masks is not None else 2, figsize=(12, 4))

        # Original image
        axs[0].imshow(TF.to_pil_image(images[i]))
        axs[0].set_title("Input Image")
        axs[0].axis("off")

        # Prediction
        pred_rgb = decode_segmentation_mask(preds_seg[i].numpy())
        axs[1].imshow(pred_rgb)
        axs[1].set_title("Predicted Segmentation")
        axs[1].axis("off")

        # Ground truth (optional)
        if masks is not None:
            mask_rgb = decode_segmentation_mask(masks[i].numpy())
            axs[2].imshow(mask_rgb)
            axs[2].set_title("Ground Truth")
            axs[2].axis("off")

        plt.tight_layout()
        plt.show()

def plot_all_visualisations(images, cams, pseudo_masks, preds_seg=None, true_masks=None, num_images=4):
    """
    Combines raw CAMs, upsampled CAMs, pseudo masks, predicted and true segmentations
    into one subplot figure for each image.
    
    Args:
        images (Tensor): (B, 3, H, W)
        cams (Tensor): (B, C, H_low, W_low)
        pseudo_masks (Tensor): (B, H, W)
        preds_seg (Tensor, optional): (B, H, W)
        true_masks (Tensor, optional): (B, H, W)
        num_images (int): Number of examples to visualize
    """
    images = images.cpu()
    cams = cams.cpu()
    pseudo_masks = pseudo_masks.cpu()
    if preds_seg is not None:
        preds_seg = preds_seg.cpu()
    if true_masks is not None:
        true_masks = true_masks.cpu()

    cam_max_raw = torch.max(cams, dim=1)[0]  # (B, H_low, W_low)
    cam_upsampled = F.interpolate(cams, size=(224, 224), mode='bilinear', align_corners=False)
    cam_max_up = torch.max(cam_upsampled, dim=1)[0]

    pseudo_masks_up = F.interpolate(pseudo_masks.unsqueeze(1).float(), size=(224, 224), mode='nearest').squeeze(1).long()

    for i in range(min(num_images, images.shape[0])):
        cols = 4
        if preds_seg is not None:
            cols += 1
        if true_masks is not None:
            cols += 1
        fig, axs = plt.subplots(1, cols, figsize=(4 * cols, 4))

        # Original image
        axs[0].imshow(TF.to_pil_image(images[i]))
        axs[0].set_title("Input Image")
        axs[0].axis("off")

        # Raw CAM (low-res)
        axs[1].imshow(cam_max_raw[i], cmap='jet')
        axs[1].set_title("Raw CAM")
        axs[1].axis("off")

        # Upsampled CAM
        axs[2].imshow(TF.to_pil_image(images[i]))
        axs[2].imshow(cam_max_up[i], cmap='jet', alpha=0.5)
        axs[2].set_title("Upsampled CAM")
        axs[2].axis("off")

        # Pseudo mask
        axs[3].imshow(pseudo_masks_up[i], cmap='viridis')
        axs[3].set_title("Pseudo Mask")
        axs[3].axis("off")

        col_idx = 4  # After first four subplots

        if preds_seg is not None:
            pred_rgb = decode_segmentation_mask(preds_seg[i].numpy())
            axs[col_idx].imshow(pred_rgb)
            axs[col_idx].set_title("Predicted Seg")
            axs[col_idx].axis("off")
            col_idx += 1

        if true_masks is not None:
            mask_rgb = decode_segmentation_mask(true_masks[i].numpy())
            axs[col_idx].imshow(mask_rgb)
            axs[col_idx].set_title("Ground Truth")
            axs[col_idx].axis("off")

        plt.tight_layout()
        plt.show()