import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
import os
import torch.nn.functional as F
from tqdm import tqdm

def download_data(pth=None, split="trainval"):
    image_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor()
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
        transforms.Lambda(lambda x: (x.long() - 1).clamp(0))
    ])
    dataset = OxfordIIITPet(
        root=pth,
        split=split,
        target_types=("category", "segmentation"),
        download=True,
        transform=image_transform,
        target_transform=lambda t: (t[0], mask_transform(t[1]))
    )
    return dataset

class FrozenResNetCAM(nn.Module):
    def __init__(self, num_classes=37):
        super().__init__()
        resnet = resnet50(pretrained=True, replace_stride_with_dilation=[False, False, True])
        for param in resnet.parameters():
            param.requires_grad = False

        # Split layers for easier access

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        # self.layer5 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.layer0(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        # f5 = self.layer5(f4)

        pooled = self.avgpool(f4)
        flat = pooled.view(pooled.size(0), -1)
        logits = self.fc(flat)

        # Return all feature maps for LayerCAM
        return logits, [f2, f3, f4]

class LocalNormalizedCutLoss(nn.Module):
    def __init__(self, sigma_color=0.05, window_size=5):
        super().__init__()
        self.sigma_color = sigma_color
        self.window_size = window_size

    def forward(self, preds, images):
        B, C, H, W = preds.shape
        pad = self.window_size // 2
        probs = F.softmax(preds, dim=1)
        images = images  # normalized image

        probs_padded = F.pad(probs, (pad, pad, pad, pad), mode='reflect')
        img_padded = F.pad(images, (pad, pad, pad, pad), mode='reflect')

        loss = 0.0
        count = 0

        for dy in range(-pad, pad + 1):
            for dx in range(-pad, pad + 1):
                if dx == 0 and dy == 0:
                    continue

                shifted_probs = probs_padded[:, :, pad + dy:pad + dy + H, pad + dx:pad + dx + W]
                shifted_img = img_padded[:, :, pad + dy:pad + dy + H, pad + dx:pad + dx + W]

                color_diff = (images - shifted_img).pow(2).sum(dim=1, keepdim=True)  # (B,1,H,W)
                affinity = torch.exp(-color_diff / (2 * self.sigma_color ** 2))

                for c in range(C):
                    Sk = probs[:, c:c+1]  # (B,1,H,W)
                    diff = (Sk - shifted_probs[:, c:c+1]) ** 2
                    loss += (affinity * diff).mean()

                count += 1

        return loss / (count * C)

from skimage.measure import label, regionprops
def keep_largest(mask):
    mask_np = mask.cpu().numpy()
    labeled = label(mask_np)
    props = regionprops(labeled)
    if not props: return mask
    largest = max(props, key=lambda r: r.area)
    return torch.from_numpy((labeled == largest.label).astype(np.uint8))

def train_fc_only(model, dataloader, device, epochs=10):
    print("Starting training...")
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)  # Only train FC layer
    criterion = nn.CrossEntropyLoss()
    # criterion_ncut = LocalNormalizedCutLoss(sigma_color=0.05)
    # lambda_ncut = 0.5

    # for epoch in range(epochs):
    #     total_loss, correct, total = 0.0, 0, 0
    #     print(f"Epoch {epoch + 1}/{epochs}")
        
        # for imgs, (labels, masks) in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            # imgs, labels, masks = imgs.to(device), labels.to(device), masks.to(device)
            # masks = masks.squeeze(1)  # Remove channel dimension
            # logits, outmasks = model(imgs)
            # outmasks_upsampled = F.interpolate(outmasks, size=masks.shape[1:], mode='bilinear', align_corners=False)
            # loss_ce = criterion_ce(outmasks_upsampled, masks)

            # # assert outmasks_upsampled.shape[2:] == imgs.shape[2:], "Spatial dimensions of outmasks and imgs must match."
            # # loss_ncut = criterion_ncut(outmasks_upsampled, imgs)

            # # loss = loss_ce + lambda_ncut * loss_ncut
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # total_loss += loss.item() * imgs.size(0)
            # preds = logits.argmax(dim=1)
            # correct += (preds == labels).sum().item()
            # total += imgs.size(0)
            

    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for imgs, (labels, _) in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits, _ = model(imgs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
        
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / total:.4f} - Acc: {100 * correct / total:.2f}%")

    model.eval()

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

def overlay_cam_on_image(image_tensor, cam_tensor, alpha=0.5, colormap='gray'):
    """
    Overlays a CAM heatmap on an input image.

    Args:
        image_tensor (Tensor): Tensor of shape (3, H, W)
        cam_tensor (Tensor): Tensor of shape (H, W) with values in [0, 1]
        alpha (float): Opacity of the heatmap
        colormap (str): Matplotlib colormap to use

    Returns:
        np.ndarray: Overlay image as (H, W, 3) numpy array
    """
    # Convert image to numpy (H, W, 3)
    image = TF.to_pil_image(image_tensor.cpu())
    image_np = np.array(image).astype(np.float32) / 255.0

    # Normalize cam and convert to heatmap
    try:
        cam = cam_tensor.cpu().numpy()
    except:
        cam = cam_tensor.cpu().detach().numpy()
    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    heatmap = plt.get_cmap(colormap)(cam)[:, :, :3]  # Drop alpha channel

    # Blend heatmap and image
    overlay = (1 - alpha) * image_np + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)

    return overlay

# pip install git+https://github.com/lucasb-eyer/pydensecrf.git

from pydensecrf.utils import unary_from_softmax
import pydensecrf.densecrf as dcrf

def apply_dense_crf(img_np, cam_np):
    """
    Refines a CAM heatmap using DenseCRF
    """
    img_np = np.ascontiguousarray(img_np)

    h, w = cam_np.shape
    d = dcrf.DenseCRF2D(w, h, 2)

    probs = np.stack([1 - cam_np, cam_np], axis=0)  # Background, foreground
    probs = np.clip(probs, 1e-8, 1.0)

    unary = unary_from_softmax(probs)
    d.setUnaryEnergy(unary)

    d.addPairwiseGaussian(sxy=1, compat=2)
    d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=img_np, compat=10)

    Q = d.inference(5)
    refined = np.argmax(Q, axis=0).reshape(h, w)

    return refined.astype(np.uint8)

from skimage.measure import label as lb, regionprops
def keep_largest(mask):
    labeled = lb(mask)
    regions = regionprops(labeled)
    if not regions:
        return mask
    largest = max(regions, key=lambda r: r.area)
    return (labeled == largest.label).astype(np.uint8)

import gc
class LayerCAMGenerator:
    def __init__(self, model, target_layer_names=["layer2", "layer3", "layer4"]):
        self.model = model.eval()
        self.target_layer_names = target_layer_names

        self.activations = {}
        self.gradients = {}

        self._register_hooks()

    def _register_hooks(self):
        for name in self.target_layer_names:
            layer = getattr(self.model, name)
            layer.register_forward_hook(self._make_forward_hook(name))
            layer.register_full_backward_hook(self._make_backward_hook(name))

    def _make_forward_hook(self, name):
        def hook(module, input, output):
            self.activations[name] = output
        return hook

    def _make_backward_hook(self, name):
        def hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0]
        return hook


    def generate(self, images, class_idx=None):
        self.activations.clear()
        self.gradients.clear()

        images = images.unsqueeze(0)
        images = images.requires_grad_()

        with torch.enable_grad():
            logits, _ = self.model(images)

            if class_idx is None:
                class_idx = torch.argmax(logits, dim=1)

            class_scores = logits.gather(1, class_idx.view(-1, 1)).squeeze()
            class_scores.backward(torch.ones_like(class_scores), retain_graph=True)

        layer_cams = []

        for name in self.target_layer_names:
            act = self.activations[name]     # (B, C, H, W)
            grad = self.gradients[name]      # (B, C, H, W)

            with torch.no_grad():
                weights = F.relu(grad * act)
                cam = weights.sum(dim=1)         # (B, H, W)
                cam = F.relu(cam)

                # Normalize per image (detach and operate on a copy)
                for i in range(cam.shape[0]):
                    c = cam[i]
                    c = c.detach()
                    c -= c.min()
                    c /= (c.max() + 1e-8)
                    cam[i] = c

                cam = F.interpolate(cam.unsqueeze(1), size=(224, 224), mode="bilinear", align_corners=False)
                layer_cams.append(cam.squeeze(1).detach())  # detach CAM from graph

            del act, grad, weights, cam  # free up memory

        final_cam = sum(layer_cams) / len(layer_cams)
        final_cam = final_cam.detach()

        del logits, class_scores, images, layer_cams
        torch.cuda.empty_cache()
        gc.collect()

        return final_cam  # (B, H, W)


    def generate_bg_cam(self, image_tensor, valid_class_indices, alpha=1.0):
        """
        Mimics CAMGenerator's bg+fg map output for integration with AffinityNet.
        """
        #all_cams = self.generate_all_cams(image_tensor)
        all_cams = self.generate(image_tensor, valid_class_indices)

        mask = torch.zeros_like(all_cams)
        valid_cams = all_cams

        max_obj_cam, _ = valid_cams.max(dim=0)  # (H, W)
        m_bg = 1.0 - ((1.0 - max_obj_cam).clamp(min=0.0) ** alpha)

        # Resize for consistency
        m_bg_resized = F.interpolate(
            m_bg.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
        ).squeeze()

        max_obj_cam_resized = F.interpolate(
            max_obj_cam.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
        ).squeeze()

        return m_bg_resized, max_obj_cam_resized
    
class CAMGenerator:
    def __init__(self, model):
        self.model = model.eval()
    """
    def generate_all_cams(self, image_tensor):
        with torch.no_grad():
            logits, features = self.model(image_tensor.unsqueeze(0))  # shape: (1, C, H, W)
            features = features[-1]  # shape: (C_feat, H, W)
            features = features.squeeze(0)  # (C_feat, H, W)

            all_cams = []
            for class_idx in range(logits.shape[1]):
                weights = self.model.fc.weight[class_idx]  # (C_feat,)
                cam = torch.einsum("c,chw->hw", weights, features)  # (H, W)
                cam = F.relu(cam)

                cam -= cam.min()
                cam /= cam.max() + 1e-8  # Normalize to [0,1]
                all_cams.append(cam)

            all_cams = torch.stack(all_cams, dim=0)  # Shape: (num_classes, H, W)
            return all_cams  # Return CAMs for all classes
    """
    def generate_all_cams(self, image_tensor):
        with torch.no_grad():
            logits, features = self.model(image_tensor.unsqueeze(0))  # shape: (1, C, H, W)
            features = features[-1]  # shape: (C_feat, H, W)
            features = features.squeeze(0)  # (C_feat, H, W)

            all_cams = []
            for class_idx in range(logits.shape[1]):
                
                if isinstance(class_idx, torch.Tensor):
                    class_idx = class_idx.item()

                
                weights = self.model.fc.weight[class_idx]  # (C_feat,)

                
                assert features.shape[0] == weights.shape[0], \
                    f"Features channel size {features.shape[0]} does not match weights size {weights.shape[0]}"

                
                cam = torch.einsum("c,chw->hw", weights, features)  # (H, W)
                cam = F.relu(cam)

                
                cam -= cam.min()
                cam /= cam.max() + 1e-8  # Normalize to [0,1]
                all_cams.append(cam)

            all_cams = torch.stack(all_cams, dim=0)  # Shape: (num_classes, H, W)
            return all_cams  # Return CAMs for all classes
    
    def generate_bg_cam(self, image_tensor, valid_class_indices, alpha=1.0):
        """
        image_tensor: (C, H, W)
        valid_class_indices: list of class indices (e.g., ground truth labels for this image)
        alpha: exponent for shaping M_bg
        """
        all_cams = self.generate_all_cams(image_tensor)

        # Zero out CAMs for irrelevant classes
        mask = torch.zeros_like(all_cams)
        for idx in valid_class_indices:
            mask[idx] = 1.0
        valid_cams = all_cams * mask

        # Max activation across valid classes at each (x, y)
        max_obj_cam, _ = valid_cams.max(dim=0)  # Shape: (H, W)

        # M_bg = (1 - max_c CAM_c)^alpha
        m_bg = 1.0 - ((1.0 - max_obj_cam).clamp(min=0.0) ** alpha)

        # Resize to 224x224 for visualization or consistency
        m_bg_resized = F.interpolate(
            m_bg.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
        ).squeeze()

        max_obj_cam_resized = F.interpolate(
            max_obj_cam.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
        ).squeeze()

        return m_bg_resized, max_obj_cam_resized

def compute_iou_and_acc(pred_mask, true_mask):
    """
    Computes binary IoU and pixel accuracy.
    Args:
        pred_mask: Tensor of shape (H, W), foreground = >0
        true_mask: Tensor of shape (H, W), foreground = >0
    """
    pred_fg = (pred_mask > 0)
    true_fg = (true_mask > 0)

    intersection = (pred_fg & true_fg).sum().item()
    union = (pred_fg | true_fg).sum().item()
    correct = (pred_mask == true_mask).sum().item()
    total = true_mask.numel()

    iou = intersection / (union + 1e-8)
    acc = correct / total
    return iou, acc

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import numpy as np

class PseudoSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_list = sorted(os.listdir(img_dir))
        self.mask_list = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

def joint_transform(image, mask):
    image = TF.resize(image, (256, 256))
    mask = TF.resize(mask, (256, 256), interpolation=Image.NEAREST)

    image = TF.to_tensor(image)
    image = TF.normalize(image, mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    mask = torch.as_tensor(np.array(mask), dtype=torch.long)

    return image, mask

import os
from torchvision.utils import save_image

os.environ["QT_QPA_PLATFORM"] = "offscreen"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = download_data("./data", split="trainval")
#loader = DataLoader(dataset, batch_size=16, shuffle=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

classifier = FrozenResNetCAM().to(device)

train_fc_only(classifier, loader, device, epochs=10)
print(" Classifier trained.")

save_dir = './saves'
os.makedirs(save_dir, exist_ok=True)  #  Create parent folder if it doesn't exist

save_path = os.path.join(save_dir, 'classifier_weights_without_affinity.pth')
torch.save(classifier.state_dict(), save_path)

print(f"✅Weights saved to {save_path}")

classifier.load_state_dict(torch.load(save_path, map_location=torch.device('cuda'), weights_only=True))
classifier.eval()

loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

test_dataset = download_data("./data", split="test")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

from os import PRIO_PROCESS
p = 0
cam_gen = CAMGenerator(classifier)
layercam_gen = LayerCAMGenerator(classifier, target_layer_names=["layer3", "layer4"])  # f2, f3, f4
device = "cuda"

for img, (label, true_mask) in test_loader:
    img = img[0].to(device)  # (3, H, W)
    true_mask = true_mask[0].to(device)
    for i in range(len(true_mask)):
        for j in range(len(true_mask[i])):
          for k in range(len(true_mask[i][j])):
            if true_mask[i][j][k] == 2 or true_mask[i][j][k] == 0:
              true_mask[i][j][k] = 0
            else:
              true_mask[i][j][k] = 1

    label = label[0].item() if isinstance(label[0], torch.Tensor) else label[0]

    valid_class_indices = [label]

    ### --- LAYERCAM GENERATOR ---
    class_tensor = torch.tensor(valid_class_indices).to(device)
    layercam_output = layercam_gen.generate(img, class_idx=class_tensor)  # (1, H, W)
    cam_bg, regular_cam = cam_gen.generate_bg_cam(img, class_tensor)  # (1, H, W)
    layercam_bg, _ = layercam_gen.generate_bg_cam(img, class_tensor, alpha=1.9)

    layercam = layercam_output.squeeze(0)  # (H, W)
    layercam[layercam < 0.3] = 0.0

    layercam_bg[layercam_bg < 0.5] = 0.0

    cam_bg[cam_bg <0.3] = 0.0

    # Build prediction masks
    pred_fg_mask = torch.zeros_like(cam_bg).long()
    pred_fg_mask[cam_bg > 0.0] = 1

    pred_bg_mask = torch.zeros_like(layercam_bg).long()
    pred_bg_mask[layercam_bg > 0.0] = 1

    # # Resize if needed
    if pred_fg_mask.shape != true_mask.shape:
        pred_fg_mask = F.interpolate(pred_fg_mask.unsqueeze(0).unsqueeze(0).float(), size=true_mask.shape[-2:], mode='nearest').squeeze().long()
        pred_bg_mask = F.interpolate(pred_bg_mask.unsqueeze(0).unsqueeze(0).float(), size=true_mask.shape[-2:], mode='nearest').squeeze().long()

    iou_fg, acc_fg = compute_iou_and_acc(pred_fg_mask, true_mask)
    iou_bg, acc_bg = compute_iou_and_acc(pred_bg_mask, true_mask)

    # Visualize overlays
    cam_overlay = overlay_cam_on_image(img, cam_bg, alpha=0.65, colormap='jet')
    layercam_bg_overlay = overlay_cam_on_image(img, layercam_bg, alpha=0.5, colormap='jet')

    # Convert image to uint8 for OpenCV (if not normalized)
    image_np = img.detach().cpu().permute(1, 2, 0).numpy()
    image_np = (image_np * 255).astype(np.uint8).clip(0, 255)

    image_np = img.detach().cpu().permute(1, 2, 0).numpy()
    image_np = (image_np * 255).astype(np.uint8).clip(0, 255)
    image_np = np.ascontiguousarray(image_np)

    refined_mask = apply_dense_crf(image_np, layercam_bg.squeeze().cpu().numpy())
    refined_mask = keep_largest(refined_mask)

    # Plots
    """
    fig, ax = plt.subplots(1, 6, figsize=(18, 6))

    ax[0].imshow(cam_overlay)
    ax[0].set_title("Regular CAM BG")
    print(f"Regular CAM BG: IoU: {iou_fg:.3f} | Acc: {acc_fg:.3f}")
    ax[0].axis('off')

    ax[1].imshow(layercam_bg_overlay)
    ax[1].set_title("LayerCAM (M_bg)")
    #ax[1].set_xlabel(f"IoU: {iou_bg:.3f} | Acc: {acc_bg:.3f}")
    print(f"LayerCAM BG: IoU: {iou_bg:.3f} | Acc: {acc_bg:.3f}")
    ax[1].axis('off')

    ax[2].imshow(img.cpu().permute(1, 2, 0).numpy())
    ax[2].set_title("Original Image")
    ax[2].axis('off')

    ax[3].imshow(true_mask.cpu().permute(1, 2, 0).numpy())
    ax[3].set_title("True Mask")
    ax[3].axis('off')
    ax[4].imshow(pred_bg_mask.cpu().numpy())
    ax[4].set_title("Pred Mask")
    ax[4].axis('off')

    ax[5].imshow(refined_mask, cmap='gray')
    ax[5].set_title("CSRF")
    ax[5].axis('off')


    plt.tight_layout()
    plt.show()
    """
    if p == 5:
        break
    p += 1

save_dir = "./pseudo_masks"
os.makedirs(save_dir, exist_ok=True)

image_save_dir = "./images"
os.makedirs(image_save_dir, exist_ok=True)

img_id = 0
layer_cam = LayerCAMGenerator(classifier, target_layer_names=["layer3", "layer4"])  # f2, f3, f4
for img, (label, true_mask) in loader:
    img = img[0].to(device)
    label = label[0].item() if isinstance(label[0], torch.Tensor) else label[0]
    class_tensor = torch.tensor([label]).to(device)

    cam_bg, _ = layer_cam.generate_bg_cam(img, class_tensor, alpha=1.9)
    cam_bg[cam_bg < 0.5] = 0.0

    pred_mask = torch.zeros_like(cam_bg).long()
    pred_mask[cam_bg > 0.0] = 1

    # Convert image to uint8 for OpenCV (if not normalized)
    image_np = img.detach().cpu().permute(1, 2, 0).numpy()
    image_np = (image_np * 255).astype(np.uint8).clip(0, 255)

    image_np = img.detach().cpu().permute(1, 2, 0).numpy()
    image_np = (image_np * 255).astype(np.uint8).clip(0, 255)
    image_np = np.ascontiguousarray(image_np)

    refined_mask = apply_dense_crf(image_np, cam_bg.squeeze().cpu().numpy())

    # Save pseudo mask
    mask_path = os.path.join(save_dir, f"{img_id}.png")
    save_image(torch.from_numpy(refined_mask).float().unsqueeze(0), mask_path)

    # Save input image (unnormalized)
    img_orig = img.cpu().clone()
    img_orig = (img_orig - img_orig.min()) / (img_orig.max() - img_orig.min())
    save_image(img_orig, os.path.join(image_save_dir, f"{img_id}.png"))

    img_id += 1
    if img_id >= 200:  # Save only 200 samples
        break

#del model

"""### Regularisation Loss: Boundaries

Combine segmentation loss with boundary loss to encourage segmentation to be performed around edge of object in image
"""

def compute_affinities(image, sigma_color=0.1, sigma_space=5, window_size=5):
    """
    Compute affinity weights for a local window around each pixel.
    image: (B, 3, H, W) tensor
    returns: (B, K, H, W) affinity weights (K = window_size*window_size)
    """
    B, C, H, W = image.size()
    pad = window_size // 2
    image_padded = F.pad(image, (pad, pad, pad, pad), mode='reflect')

    neighbors = []
    affinities = []

    for dy in range(-pad, pad+1):
        for dx in range(-pad, pad+1):
            if dx == 0 and dy == 0:
                continue

            shifted = image_padded[:, :, pad+dy:pad+dy+H, pad+dx:pad+dx+W]
            diff = (image - shifted).pow(2).sum(dim=1, keepdim=True)  # color distance
            spatial_dist = dx**2 + dy**2

            weight = torch.exp(-diff / (2 * sigma_color**2) - spatial_dist / (2 * sigma_space**2))
            affinities.append(weight)

    return affinities  # list of (B, 1, H, W)


from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
import torch

# Paths to image and mask directories
train_dataset = PseudoSegmentationDataset(
    img_dir="./images",             # update with your actual image path
    mask_dir="./pseudo_masks",      # update with your pseudo mask path
    transform=joint_transform
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Define model
num_classes = 2
model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
model = model.cuda()

criterion = nn.CrossEntropyLoss()
criterion_boundary = LocalNormalizedCutLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
lambda_loss = 0.5  # Weight for boundary loss
# Training loop
for epoch in range(10):  # Increase epochs as needed
    model.train()
    total_loss = 0
    for images, masks in train_loader:
        images, masks = images.cuda(), masks.cuda()
        # Make sure masks are only 0 or 1
        masks = torch.clamp(masks, max=1)

        outputs = model(images)['out']
        loss_ce = criterion(outputs, masks.long())
        loss_boundary = criterion_boundary(outputs, images)
        
        loss = loss_ce + lambda_loss * loss_boundary
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

from torchvision import transforms
import matplotlib.pyplot as plt

model.eval()

inference_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def infer_and_plot(img_path, pseudo_mask_path):
    img = Image.open(img_path).convert('RGB')
    pseudo_mask = Image.open(pseudo_mask_path)
    input_tensor = inference_transform(img).unsqueeze(0).cuda()

    with torch.no_grad():
        output = model(input_tensor)['out']
        pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
    """
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Predicted Segmentation")
    plt.imshow(pred)
    plt.axis('off')
    plt.show()

    plt.subplot(1, 3, 3)
    plt.title("Pseudo Mask")
    plt.imshow(pseudo_mask)
    plt.axis('off')
    plt.show()
    """

# The first 10 images
for i in range(10):
  infer_and_plot(f"./images/{i}.png", f"./pseudo_masks/{i}.png")

