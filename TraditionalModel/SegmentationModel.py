from PseudoSegmentationDataset import PseudoSegmentationDataset
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from ExtraUtilities import compute_iou_and_acc, load_split_data


# def train_segmentation_model(run_id, lr=1e-4, num_epochs=10, batch_size=4):
#     """
#     Train a DeepLabV3 segmentation model on pseudo masks generated in run `run_id`.
#     Returns the trained model.
#     """

#     image_dir = f"/content/images_{run_id}"
#     mask_dir = f"/content/pseudo_masks_{run_id}"

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     train_dataset = PseudoSegmentationDataset(
#         img_dir=image_dir,
#         mask_dir=mask_dir,
#         transform=True
#     )
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#     num_classes = 2
#     model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
#     model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
#     model = model.to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
#         for images, masks in train_loader:
#           if images.size(0) == 1:
#             continue
#           images, masks = images.to(device), masks.to(device)
#           masks = torch.clamp(masks, max=1)  # Ensure binary class labels

#           outputs = model(images)['out']
#           loss = criterion(outputs, masks.long())

#           optimizer.zero_grad()
#           loss.backward()
#           optimizer.step()

#           total_loss += loss.item()

#         final_loss = total_loss

#         print(f"[Run {run_id}] Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

#     return model, final_loss

def train_segmentation_model(loss_fn, run_id, lr=1e-4, num_epochs=10, batch_size=4, val_split=0.2):
    """
    Train a DeepLabV3 segmentation model on pseudo masks generated in run `run_id`.
    Evaluates the model on a validation set after each epoch.
    Returns the trained model and final training loss.
    
    loss_fn: string Loss function to use. Options: 'cross_entropy', 'lovasz_softmax'
    """

    image_dir = f"/content/images_{run_id}"
    mask_dir = f"/content/pseudo_masks_{run_id}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = PseudoSegmentationDataset(
        img_dir=image_dir,
        mask_dir=mask_dir,
        transform=True
    )

    # Split dataset into training and validation sets
    train_dataset, val_dataset = load_split_data()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    num_classes = 2
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, masks in train_loader:
            if images.size(0) == 1:
                continue
            images, masks = images.to(device), masks.to(device)
            masks = torch.clamp(masks, max=1)  # Ensure binary class labels

            outputs = model(images)['out']
            if loss_fn == 'lovasz_softmax':
                probas = F.softmax(outputs, dim=1)
                loss = lovasz_softmax(probas, masks, classes='present', per_image=False, ignore=None)
            else:
                loss = criterion(outputs, masks.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        final_loss = total_loss
        print(f"[Run {run_id}] Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

        # 🔍 Evaluate on validation set
        avg_iou, avg_acc = evaluate_model(model, val_loader)
        print(f"[Run {run_id}] Validation IoU: {avg_iou:.4f}, Accuracy: {avg_acc:.4f}")

    return model, final_loss



def evaluate_model(model, test_loader):
    """
    Evaluates a trained segmentation model on a test set.
    Returns average IoU and pixel accuracy.
    """
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    model.eval()

    ious, accs = [], []
    with torch.no_grad():
        for img, (label, true_mask) in test_loader:
            img = img[0].to(device)
            true_mask = true_mask[0].to(device)
            true_mask = (true_mask == 1).long()  # Binarize

            output = model(img.unsqueeze(0))['out']
            pred_mask = output.argmax(dim=1).squeeze(0)

            # Resize if needed
            if pred_mask.shape != true_mask.shape:
                pred_mask = F.interpolate(pred_mask.unsqueeze(0).unsqueeze(0).float(), size=true_mask.shape[-2:], mode='nearest').squeeze().long()

            iou, acc = compute_iou_and_acc(pred_mask, true_mask)
            ious.append(iou)
            accs.append(acc)

    avg_iou = sum(ious) / len(ious)
    avg_acc = sum(accs) / len(accs)

    print(f"\n Model Evaluation on Test Set: IoU = {avg_iou:.4f} | Acc = {avg_acc:.4f}")
    return avg_iou, avg_acc