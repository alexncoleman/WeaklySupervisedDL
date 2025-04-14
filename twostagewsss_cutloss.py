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
        if preds.dim() == 3:  # 如果是 3 维张量，添加批次维度
            preds = preds.unsqueeze(0)
            images = images.unsqueeze(0)

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
        # 打印每个 epoch 的损失和准确率
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
    img_np = np.ascontiguousarray(img_np)  # ✅ Fix non-contiguous array

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


    def generate(self, images, class_idx=None, alpha=1.0):
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
                    c = c ** alpha
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


    def generate_bg_cam(self, image_tensor, valid_class_indices, alpha=2.0):
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
                # 确保 class_idx 是标量
                if isinstance(class_idx, torch.Tensor):
                    class_idx = class_idx.item()

                # 提取对应类别的权重
                weights = self.model.fc.weight[class_idx]  # (C_feat,)

                # 确保 features 的形状正确
                assert features.shape[0] == weights.shape[0], \
                    f"Features channel size {features.shape[0]} does not match weights size {weights.shape[0]}"

                # 计算 CAM
                cam = torch.einsum("c,chw->hw", weights, features)  # (H, W)
                cam = F.relu(cam)

                # 归一化 CAM
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

        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(np.array(mask), dtype=torch.float)
        #print(f"Type of mask: {type(mask)}")
        return image, mask, os.path.basename(img_path)

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

train_fc_only(classifier, loader, device, epochs=15)
print(" Classifier trained.")

save_dir = './saves'
os.makedirs(save_dir, exist_ok=True)  #  Create parent folder if it doesn't exist

save_path = os.path.join(save_dir, 'classifier_weights_without_affinity.pth')
torch.save(classifier.state_dict(), save_path)

print(f"✅Weights saved to {save_path}")

classifier.load_state_dict(torch.load(save_path, map_location=torch.device('cuda'), weights_only=True))
classifier.eval()

test_dataset = download_data("./data", split="test")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

from os import PRIO_PROCESS
cam_gen = CAMGenerator(classifier)
layercam_gen = LayerCAMGenerator(classifier, target_layer_names=["layer3", "layer4"])  # f2, f3, f4
device = "cuda"

save_dir = "./pseudo_masks"
os.makedirs(save_dir, exist_ok=True)

image_save_dir = "./images"
os.makedirs(image_save_dir, exist_ok=True)

img_id = 0

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
    layercam_output = layercam_gen.generate(img, class_idx=class_tensor, alpha=0.5)  # (1, H, W)

    layercam = layercam_output.squeeze(0)  # (H, W)
    layercam[layercam < 0.2] = 0.0

    cam_bg = layercam

    # Build prediction masks
    pred_fg_mask = torch.zeros_like(cam_bg).long()
    pred_fg_mask[cam_bg > 0.0] = 1


    # # Resize if needed
    if pred_fg_mask.shape != true_mask.shape:
        pred_fg_mask = F.interpolate(pred_fg_mask.unsqueeze(0).unsqueeze(0).float(), size=true_mask.shape[-2:], mode='nearest').squeeze().long()
        # pred_bg_mask = F.interpolate(pred_bg_mask.unsqueeze(0).unsqueeze(0).float(), size=true_mask.shape[-2:], mode='nearest').squeeze().long()

    iou_fg, acc_fg = compute_iou_and_acc(pred_fg_mask, true_mask)
    # iou_bg, acc_bg = compute_iou_and_acc(pred_bg_mask, true_mask)

    # Visualize overlays
    cam_overlay = overlay_cam_on_image(img, cam_bg, alpha=0.65, colormap='jet')

    # Convert image to uint8 for OpenCV (if not normalized)
    image_np = img.detach().cpu().permute(1, 2, 0).numpy()
    image_np = (image_np * 255).astype(np.uint8).clip(0, 255)

    image_np = img.detach().cpu().permute(1, 2, 0).numpy()
    image_np = (image_np * 255).astype(np.uint8).clip(0, 255)
    image_np = np.ascontiguousarray(image_np)

    refined_mask = apply_dense_crf(image_np, cam_bg.squeeze().cpu().numpy())
    # refined_mask = keep_largest(refined_mask)

    mask_path = os.path.join(save_dir, f"{img_id}.png")
    save_image(torch.from_numpy(refined_mask).float().unsqueeze(0), mask_path)

    # Save input image (unnormalized)
    img_orig = img.cpu().clone()
    img_orig = (img_orig - img_orig.min()) / (img_orig.max() - img_orig.min())
    save_image(img_orig, os.path.join(image_save_dir, f"{img_id}.png"))

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
    img_id += 1
    if img_id >= 200:  # Save only 200 samples
        break

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

def evaluate_model(model, test_loader):
    """
    Evaluates a trained segmentation model on a test set.
    Returns average IoU and pixel accuracy.
    """
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    model.eval()

    acc=[]
    iou=[]

    with torch.no_grad():
        for img, (label, true_mask) in test_loader:
            img = img[0].to(device)  # (3, H, W)
            true_mask = true_mask[0].to(device)

            # Normalize ground truth to binary (0 = background, 1 = foreground)
            true_mask = true_mask.clone()
            true_mask[true_mask == 2] = 1
            true_mask[true_mask == 0] = 0
            true_mask = 1- true_mask

            # Get model prediction
            with torch.no_grad():
                input_tensor = img.unsqueeze(0)  # (1, 3, H, W)
                output = model(input_tensor)['out']  # (1, C, H, W)
                pred = torch.argmax(output.squeeze(), dim=0)  # (H, W)

            # Resize prediction if needed
            if pred.shape != true_mask.shape:
                pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0).float(),
                                    size=true_mask.shape[-2:], mode='nearest').squeeze().long()

            # Compute IoU and accuracy
            iou_seg, acc_seg = compute_iou_and_acc(pred, true_mask)
            acc.append(acc_seg)
            iou.append(iou_seg)
        average_iou = sum(iou) / len(iou)
        average_acc = sum(acc) / len(acc)
        #print(f"Segmentation Model: IoU: {average_iou:.3f} | Accuracy: {average_acc:.3f}")
    return average_iou, average_acc

def train_model(model, optimizer, criterion_ce, num_epochs = 3):
    # Define model

    # Define losses

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, masks, _ in train_loader:
            images, masks = images.cuda(), masks.cuda()
            masks = torch.clamp(masks, max=1)  # Ensure binary labels

            outputs = model(images)['out']  # (B, C, H, W)

            loss = criterion_ce(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

def refine_pseudo_mask(model, image, mask, lambda_boundary=0.1, threshold=0.5, lr=1e-2, num_steps=20,
                       sigma_color=0.1, window_size=5):
    # Ensure the image is on the same device as the model.
    device = next(model.parameters()).device
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        # Get the network's soft prediction S (shape: (1, C, H, W))
        input_tensor = image.unsqueeze(0)  # add batch dimension
        S = model(input_tensor)['out']
        S = F.softmax(S, dim=1)

    # Initialize X from the given mask (assumed to be a probability tensor with the same shape as S)
    # If mask is binary (0/1), it will be treated as logits in the KL divergence.
    # Ensure mask is on the correct device.
    num_classes = 2
    mask = (mask == 255).long()
    X_init = F.one_hot(mask.long(), num_classes=num_classes).permute(2, 0, 1).float()  # (2, H, W)
    X = X_init.unsqueeze(0).to(device).requires_grad_(True)
    optimizer_X = torch.optim.Adam([X], lr=lr)

    # Instantiate the boundary loss. You can pass additional hyperparameters if needed.
    criterion_boundary = LocalNormalizedCutLoss(sigma_color=sigma_color, window_size=window_size)

    # Optimize X over a fixed number of steps.
    total_loss = 0.0
    for step in range(num_steps):
        optimizer_X.zero_grad()
        # Project X to probability simplex
        X_norm = F.softmax(X, dim=1)

        # KL divergence between X_norm and network prediction S
        loss_kl = F.kl_div((X_norm + 1e-8).log(), S, reduction='batchmean')

        # Boundary loss on X_norm
        loss_boundary = criterion_boundary(X_norm[0], input_tensor[0])

        # print("KL:", loss_kl.item(), "Boundary:", loss_boundary.item())
        lambda_boundary_dynamic = lambda_boundary * (loss_kl.item() / (loss_boundary.item() + 1e-6))

        # Combined loss with dynamically adjusted boundary loss weight
        loss = loss_kl + lambda_boundary_dynamic * loss_boundary

        # print(loss_kl, lambda_boundary * loss_boundary)

        loss.backward()
        optimizer_X.step()
        total_loss += loss.item()

    # print(f"refine loss: {total_loss:.4f}")
    # After refinement, compute the final soft distribution.
    X_final = F.softmax(X, dim=1)

    # For binary segmentation, assume channel 1 is the foreground.
    # Threshold the foreground probability at the given threshold to obtain a binary mask.
    pseudo_mask_refined = (X_final[0, 1] > threshold).float()

    return pseudo_mask_refined

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
net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
net.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
net = net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
criterion_ce = nn.CrossEntropyLoss()

for iteration in range(10):
    # --- Step 1: Train the model on current pseudo masks ---
    train_model(net, optimizer, criterion_ce, num_epochs = 10)
    # Optionally, evaluate on a validation set to monitor improvement
    avg_iou, avg_acc = evaluate_model(net, test_loader)
    print(f"Iteration {iteration+1}: Evaluation -> Mean IoU: {avg_iou:.4f}, Mean Acc: {avg_acc:.4f}")

    # --- Step 2: Update (Refine) the pseudo masks ---
    # Loop through each image in the training set, refine the pseudo mask, and save the updated mask.
    new_mask_paths = []
    #for idx, (img, mask) in enumerate(dataset):
    #    print(f"Type of mask: {type(mask)}")
    for repeated in range(5):
        for idx, (img, mask, img_name) in enumerate(train_dataset):
            # img has shape (C, H, W); ensure it is on the proper device
            refined_mask = refine_pseudo_mask(net, img, mask, threshold=0.3, lr=1e-4, num_steps = 10, lambda_boundary=0.1)
            # Save the refined pseudo mask (overwrite the previous one)
            mask_path = os.path.join(save_dir, img_name)
            save_image(refined_mask.unsqueeze(0).float(), mask_path)
            new_mask_paths.append(mask_path)

    # Option 2: If your dataset caches the masks, you might need to reinitialize your dataset.
    train_dataset = PseudoSegmentationDataset(
        img_dir="./images",
        mask_dir=save_dir,  # now updated with refined masks
        transform=joint_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


print("Alternating training and pseudo mask updates completed.")

from torchvision import transforms
import matplotlib.pyplot as plt

net.eval()

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
        output = net(input_tensor)['out']
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

