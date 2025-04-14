import torch
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def custom_target_transform(target):
    category, mask = target
    mask = mask.resize((224, 224), Image.NEAREST)
    mask = torch.from_numpy(np.array(mask)).long()
    return category, mask

dataset = OxfordIIITPet(
    root="./data",
    download=True,
    target_types=("category", "segmentation"),
    transform=transform,
    target_transform=custom_target_transform
)

gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)
gdino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")

sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

os.makedirs("pseudo_masks", exist_ok=True)

def convert_3channel_to_mask(image_3ch, threshold=0.5):
    gray = np.max(image_3ch, axis=0)
    binary_mask = (gray > threshold).astype(np.uint8)
    return binary_mask

def run_segmentation_pipeline(image_pil, prompts=["a cat", "a dog"]):
    inputs = gdino_processor(images=image_pil, text=[prompts], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = gdino_model(**inputs)

    target_sizes = torch.tensor([image_pil.size[::-1]]).to(device)
    results = gdino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.3,
        text_threshold=0.25,
        target_sizes=target_sizes
    )[0]

    if len(results["boxes"]) == 0:
        return None

    input_boxes = [[box.detach().cpu().numpy().tolist() for box in results["boxes"]]]
    sam_inputs = sam_processor(image_pil, input_boxes=input_boxes, return_tensors="pt").to(device)

    with torch.no_grad():
        raw_masks = sam_model(**sam_inputs).pred_masks
        masks = raw_masks[:, 0].cpu().numpy()

    if masks.ndim == 2:
        masks = np.expand_dims(masks, axis=0)

    combined_mask = convert_3channel_to_mask(masks.squeeze(0))
    combined_mask = Image.fromarray(combined_mask * 255).resize((224, 224), Image.NEAREST)
    combined_mask = np.array(combined_mask) // 255  
    return combined_mask

def plot_segmentation(image_pil, mask_np, title="Predicted Mask"):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image_pil)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask_np, cmap='gray')
    plt.title(title)
    plt.axis("off")
    plt.show()

def calculate_iou_and_accuracy(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    iou = intersection / union if union != 0 else 1.0

    total_pixels = pred_mask.size
    correct_pixels = (pred_mask == true_mask).sum()
    pixel_accuracy = correct_pixels / total_pixels
    return iou, pixel_accuracy

def evaluate_pipeline_on_dataset(dataset, num_images=100, save_outputs=True, save_dir="pseudo_masks"):
    iou_scores = []
    acc_scores = []

    os.makedirs(save_dir, exist_ok=True)

    for idx in range(num_images):
        image_tensor, (category_label, mask_tensor) = dataset[idx]

        image_pil = transforms.ToPILImage()(image_tensor)
        gt_mask = mask_tensor.squeeze().numpy()
        gt_mask_bin = np.isin(gt_mask, [1, 3]).astype(np.uint8)

        pred_mask = run_segmentation_pipeline(image_pil)

        if pred_mask is None:
            print(f"Image {idx}: No prediction made")
            continue

        iou, acc = calculate_iou_and_accuracy(pred_mask, gt_mask_bin)
        iou_scores.append(iou)
        acc_scores.append(acc)

        print(f"[{idx+1}/{num_images}] IoU: {iou:.4f}, Accuracy: {acc:.4f}")

        if save_outputs:
            save_path = os.path.join(save_dir, f"mask_{idx}.png")
            Image.fromarray((pred_mask * 255).astype(np.uint8)).save(save_path)

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(image_pil)
            axs[0].set_title("Original Image")
            axs[0].axis("off")

            axs[1].imshow(gt_mask_bin, cmap="gray")
            axs[1].set_title("Ground Truth")
            axs[1].axis("off")

            axs[2].imshow(pred_mask, cmap="gray")
            axs[2].set_title("Predicted Mask")
            axs[2].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"viz_{idx}.png"))
            plt.close()

    avg_iou = np.mean(iou_scores)
    avg_acc = np.mean(acc_scores)
    print("\n--- Final Evaluation ---")
    print(f"Average IoU over {len(iou_scores)} images: {avg_iou:.4f}")
    print(f"Average Pixel Accuracy: {avg_acc:.4f}")

if __name__ == "__main__":
  # Full Dataset
  #evaluate_pipeline_on_dataset(dataset, num_images=len(dataset))

  # Run on 10 images for now
  evaluate_pipeline_on_dataset(dataset, num_images=10)
