from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

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


def download_data(pth=None, split="test"):
    image_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.PILToTensor()
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

def load_split_data(pth=None, train_ratio=0.8):
    """
    Downloads the Oxford-IIIT Pet dataset and splits the 'trainval' portion
    into training and validation sets.

    Args:
        pth (str): Path to store the data.
        train_ratio (float): Proportion of the data to use for training (0 < train_ratio < 1).

    Returns:
        train_dataset, val_dataset (Subset, Subset)
    """
    assert 0 < train_ratio < 1, "train_ratio must be between 0 and 1 (exclusive)"
    full_dataset = download_data(pth=pth, split="trainval")
    
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    return train_dataset, val_dataset