import torch
import torch.nn as nn
import torch.optim as optim
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
class TwoStageWSSS:
    def __init__(
        self, 
        classification_model=None, 
        refine_model=None, 
        grounding_model=None, 
        sam_model=None, 
        segmentation_model=None,
        mode="traditional",  # or "foundation"
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.mode = mode
        self.classification_model = classification_model
        self.refine_model = refine_model
        self.grounding_model = grounding_model
        self.sam_model = sam_model
        self.segmentation_model = segmentation_model
        self.device = device

    def train_classification(self, dataloader, epochs=5):
        """
        Train the classification model on image-level labels.
        """
        model = self.classification_model.to(self.device)
        model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(epochs):
            total_loss, total_correct, total_samples = 0.0, 0, 0

            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                total_correct += (preds == labels).sum().item()
                total_samples += images.size(0)

            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples * 100
            print(f"[Classification] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")

        model.eval()

    def generate_seeds(self, images):
        """
        Generate class activation map (CAM) style seeds from classification model.
        This is simplified: real CAMs involve feature maps + weights.
        """
        self.classification_model.eval()
        images = images.to(self.device)
        with torch.no_grad():
            outputs = self.classification_model(images)
            seeds = outputs.argmax(dim=1)  # Fake seeds per image
        return seeds  # Shape: (B,)

    def refine_seeds(self, seeds, images):
        """
        Optional seed refinement step.
        """
        if self.refine_model:
            return self.refine_model(seeds, images)
        return seeds  # No refinement used

    def generate_masks_with_foundation_models(self, images, labels):
        """
        Foundation model method using grounding + SAM.
        """
        self.grounding_model.eval()
        self.sam_model.eval()
        images = images.to(self.device)

        with torch.no_grad():
            boxes = self.grounding_model(images, labels)
            masks = self.sam_model(images, boxes)
        return masks

    def train_segmentation(self, dataloader, pseudo_masks, epochs=5):
        """
        Train the segmentation model using pseudo masks.
        Assumes `dataloader` matches `pseudo_masks` order.
        """
        model = self.segmentation_model.to(self.device)
        model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(epochs):
            total_loss = 0.0
            total_pixels = 0
            correct_pixels = 0

            for (images, _), masks in zip(dataloader, pseudo_masks):
                images = images.to(self.device)
                masks = masks.to(self.device)  # Shape: (B, H, W)

                optimizer.zero_grad()
                outputs = model(images)  # Shape: (B, C, H, W)

                if outputs.shape[-2:] != masks.shape[-2:]:
                    masks = F.interpolate(masks.unsqueeze(1).float(), size=outputs.shape[-2:], mode='nearest').squeeze(1).long()

                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_pixels += (preds == masks).sum().item()
                total_pixels += masks.numel()

            avg_loss = total_loss / len(dataloader)
            acc = 100 * correct_pixels / total_pixels
            print(f"[Segmentation] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} | Pixel Acc: {acc:.2f}%")

        model.eval()

    def run_pipeline(self, dataloader, labels=None):
        """
        Main pipeline: classification → (optional) refinement → segmentation.
        """
        pseudo_masks = []

        for images, image_labels in dataloader:
            if self.mode == "traditional":
                seeds = self.generate_seeds(images)  # Class per image
                refined = self.refine_seeds(seeds, images)

                # Dummy conversion to per-pixel masks for demo purposes
                # Real use-case would need CAM masks here
                masks = torch.stack([torch.full((images.shape[2], images.shape[3]), c, dtype=torch.long) for c in refined])
                pseudo_masks.append(masks)

            elif self.mode == "foundation":
                masks = self.generate_masks_with_foundation_models(images, labels or image_labels)
                pseudo_masks.append(masks)

        # Train segmentation model on pseudo masks
        self.train_segmentation(dataloader, pseudo_masks)


# Example use for the two stage wsss pipeline

# model = TwoStageWSSS(
#     classification_model=MyResNetCAM(), 
#     refine_model=AffinityRefineNet(),
#     segmentation_model=DeepLabV3(num_classes=21),
#     mode="traditional"
# )
# model = TwoStageWSSS(
#     grounding_model=CLIPGrounding(), 
#     sam_model=SAMWrapper(),
#     segmentation_model=UNet(),
#     mode="foundation"
# )


# Use one of the above two methods (traditional or foundation), then run_pipeline method inputting the dataloader in.