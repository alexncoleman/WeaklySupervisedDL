import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from PseudoMasks import generate_pseudo_masks
from SegmentationDataset import PseudoSegmentationDataset

class ConstrainToBoundaryLossSingle(nn.Module):
    def __init__(self, sigma_color=0.1, sigma_space=5, window_size=5, eps=1e-8):
        super().__init__()
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.window_size = window_size
        self.eps = eps

    def forward(self, preds, image):
        """
        preds: (C, H, W) softmax output from segmentation model
        image: (3, H, W) input image (normalized 0-1)
        returns: scalar boundary loss
        """
        C, H, W = preds.shape
        pad = self.window_size // 2
        preds_padded = F.pad(preds.unsqueeze(0), (pad, pad, pad, pad), mode='reflect').squeeze(0)
        affinities = self.compute_affinities_single(image, self.sigma_color, self.sigma_space, self.window_size)

        loss = 0.0
        idx = 0
        for dy in range(-pad, pad + 1):
            for dx in range(-pad, pad + 1):
                if dx == 0 and dy == 0:
                    continue
                shifted_preds = preds_padded[:, pad + dy:pad + dy + H, pad + dx:pad + dx + W]
                diff = (preds - shifted_preds).pow(2).sum(dim=0)  # (H, W)
                weight = affinities[idx].squeeze(0)               # (H, W)
                loss += (weight * diff).mean()
                idx += 1

        boundary_loss = loss / idx
        return boundary_loss
    
    def compute_affinities_single(image, sigma_color=0.1, sigma_space=5, window_size=5):
        """
        Compute affinity weights for a local window around each pixel.
        image: (3, H, W) tensor (single image)
        returns: list of (1, H, W) affinity weights
        """
        C, H, W = image.size()
        pad = window_size // 2
        image_padded = F.pad(image.unsqueeze(0), (pad, pad, pad, pad), mode='reflect').squeeze(0)

        affinities = []

        for dy in range(-pad, pad + 1):
            for dx in range(-pad, pad + 1):
                if dx == 0 and dy == 0:
                    continue

                shifted = image_padded[:, pad + dy:pad + dy + H, pad + dx:pad + dx + W]
                diff = (image - shifted).pow(2).sum(dim=0, keepdim=True)  # (1, H, W)
                spatial_dist = dx**2 + dy**2

                weight = torch.exp(-diff / (2 * sigma_color**2) - spatial_dist / (2 * sigma_space**2))  # (1, H, W)
                affinities.append(weight)

        return affinities  # list of (1, H, W)
    


def setup_dirs(mask_dir="/content/pseudo_masks", image_dir="/content/images"):
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    return mask_dir, image_dir

def visualize_mask(img_tensor, mask_tensor, title=""):
    img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    mask_np = mask_tensor.cpu().numpy()

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask_np, cmap='gray')
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def initialize_dataset_and_loader(mask_dir, image_dir, joint_transform, batch_size=32):
    dataset = PseudoSegmentationDataset(
        img_dir=image_dir,
        mask_dir=mask_dir,
        transform=joint_transform
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, dataloader

def initialize_model():
    num_classes = 2
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model.cuda()

def train_and_refine(model, dataset, mask_dir, lambda_boundary, momentum, sigma_color, sigma_space, epochs, visualize_idx, iteration):
    # Train model on current pseudo masks
    model.train()
    model = train_model(
        lambda_boundary=lambda_boundary,
        num_epochs=epochs,
        momentum=momentum,
        sigma_color=sigma_color,
        sigma_space=sigma_space
    )

    # Refine pseudo masks
    for idx, (img, mask) in enumerate(dataset):
        refined_mask = refine_pseudo_mask(model, img, mask, threshold=0.5, num_steps=75, lambda_boundary=lambda_boundary)
        mask_path = os.path.join(mask_dir, f"{idx}.png")
        save_image(refined_mask.unsqueeze(0).float(), mask_path)

        if idx == visualize_idx:
            visualize_mask(img, refined_mask, title=f"Refined Mask (Iter {iteration+1})")

def run_alternating_training(
    loader,
    layercam_gen,
    joint_transform,
    num_alternations=5,
    epochs_per_round=15,
    lambda_boundary=0.5,
    momentum=0.9,
    sigma_color=0.1,
    sigma_space=10,
    cam_thresh=0.5,
    alpha=1.0,
    keep_largest_masks=True,
    visualize_idx=0
):
    # Step 1: Generate initial pseudo masks
    generate_pseudo_masks(
        loader, layercam_gen,
        cam_thresh=cam_thresh,
        alpha=alpha,
        keep_largest_masks=keep_largest_masks,
    )

    # Step 2: Setup directories
    mask_dir, image_dir = setup_dirs()

    # Step 3: Visualize initial mask
    dataset, train_loader = initialize_dataset_and_loader(mask_dir, image_dir, joint_transform)
    img, mask = dataset[visualize_idx]
    visualize_mask(img, mask, title="Initial Pseudo Mask")

    # Step 4: Initialize model
    model = initialize_model()

    # Step 5: Alternating training loop
    for iteration in range(num_alternations):
        print(f"\n### Alternation {iteration+1}/{num_alternations}")
        train_and_refine(
            model=model,
            dataset=dataset,
            mask_dir=mask_dir,
            lambda_boundary=lambda_boundary,
            momentum=momentum,
            sigma_color=sigma_color,
            sigma_space=sigma_space,
            epochs=epochs_per_round,
            visualize_idx=visualize_idx,
            iteration=iteration
        )

        # Reinitialize dataset with updated masks
        dataset, train_loader = initialize_dataset_and_loader(mask_dir, image_dir, joint_transform)

    print("Alternating training and pseudo mask updates completed.")


# To run it:
# run_alternating_training(
#     loader=train_loader,
#     layercam_gen=layercam_generator,
#     joint_transform=True
# )
