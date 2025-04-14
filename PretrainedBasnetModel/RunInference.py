import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import BASNet
import numpy as np

# Paths
model_path = '../Weights/basnet.pth'
dataset_root = './OxfordIIITPetDataset/oxford-iiit-pet'
image_folder = os.path.join(dataset_root, 'images')
trimap_folder = os.path.join(dataset_root, 'annotations', 'trimaps')
test_txt = os.path.join(dataset_root, 'annotations', 'test.txt')
output_folder = './basnet_outputs'
os.makedirs(output_folder, exist_ok=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
net = BASNet(3, 1)
net.load_state_dict(torch.load(model_path, map_location=device))
net.to(device)
net.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Post-processing
def norm_pred(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi + 1e-8)
    return dn

def compute_metrics(pred_mask, gt_mask):
    """
    Compute IoU and pixel accuracy between two binary masks.
    """
    pred_bin = (pred_mask > 0.5).astype(np.uint8)
    gt_bin = (gt_mask == 1).astype(np.uint8)  # Convert trimap to binary (class 1 = foreground)

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    iou = intersection / union if union > 0 else 1.0

    accuracy = (pred_bin == gt_bin).sum() / pred_bin.size

    return iou, accuracy, pred_bin, gt_bin

# Read first 5 test image names from test.txt
with open(test_txt, 'r') as f:
    test_images = [line.strip().split(' ')[0] for line in f.readlines()[:10]]

results = []

# Run inference and evaluation
for fname in test_images:
    img_path = os.path.join(image_folder, f"{fname}.jpg")
    trimap_path = os.path.join(trimap_folder, f"{fname}.png")
    
    # Load image
    image = Image.open(img_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        d1, *_ = net(input_tensor)
        pred = norm_pred(d1[:, 0, :, :]).squeeze().cpu().numpy()

    # Resize prediction back to original image size
    pred_resized = np.array(Image.fromarray((pred * 255).astype(np.uint8)).resize(image.size)) / 255.0

    # Save saliency map
    pred_img = (pred * 255).astype(np.uint8)
    saliency = Image.fromarray(pred_img)
    saliency = saliency.resize(image.size)
    saliency.save(os.path.join(output_folder, f"{fname}_saliency.png"))

    # Load and resize ground truth trimap
    gt_mask = Image.open(trimap_path)
    gt_mask_resized = gt_mask.resize(image.size, resample=Image.NEAREST)
    gt_mask_np = np.array(gt_mask_resized)

    # Compute metrics
    iou, acc, pred_bin, gt_bin = compute_metrics(pred_resized, gt_mask_np)

    print(f"{fname} - IoU: {iou:.4f}, Pixel Accuracy: {acc:.4f}")

    results.append((iou, acc))

    # # Save and visualize
    # plt.figure(figsize=(15, 6))
    # plt.subplot(1, 4, 1)
    # plt.imshow(image)
    # plt.title("Original")
    # plt.axis('off')

    # plt.subplot(1, 4, 2)
    # plt.imshow(pred_bin, cmap='gray')
    # plt.title("Predicted Mask")
    # plt.axis('off')

    # plt.subplot(1, 4, 3)
    # plt.imshow(gt_bin, cmap='gray')
    # plt.title("Ground Truth Mask")
    # plt.axis('off')

    # plt.subplot(1, 4, 4)
    # plt.imshow(gt_bin, cmap='gray', alpha=0.5)
    # plt.imshow(pred_bin, cmap='jet', alpha=0.5)
    # plt.title("Overlay")
    # plt.axis('off')

    # plt.tight_layout()
    # plt.show()


mean_iou = sum(iou for iou, _ in results) / len(results)
mean_acc = sum(acc for _, acc in results) / len(results)
print(f"Mean IoU: {mean_iou:.4f}, Mean Pixel Accuracy: {mean_acc:.4f}")