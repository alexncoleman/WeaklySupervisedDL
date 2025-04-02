from torchvision.datasets import OxfordIIITPet
import torchvision.transforms as transforms
import torchvision.models.segmentation as segmentation
import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
target_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.long)-1)
    #transforms.Lambda(lambda x: torch.tensor(np.where(np.array(x) == 2, -1, np.array(x)), dtype=torch.long))
    #transforms.Lambda(lambda x: torch.tensor(np.where(np.array(x, dtype=np.int64) == 2, -1, np.where(np.array(x, dtype=np.int64) == 255, -1, np.array(x, dtype=np.int64))), dtype=torch.long))
])
test_dataset = OxfordIIITPet(root='./data', split='test', target_types=['segmentation'], download=True,
                            transform=transform, target_transform=target_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

model = segmentation.deeplabv3_resnet50(weights=None, num_classes=3)
model.load_state_dict(torch.load('./deeplabv3_resnet50_fully_supervised.pth', map_location=torch.device('cpu')))

#inference
model.eval()
with torch.no_grad():
    for i, (images, masks) in enumerate(test_loader):
        outputs = model(images)['out']
        pred = torch.argmax(outputs, dim=1)
        pred = pred.cpu().numpy()

        for j in range(pred.shape[0]):
            original = images[j].cpu().numpy().transpose(1, 2, 0)
            original = (original * 255).astype(np.uint8)
            original = Image.fromarray(original)

            mask = pred[j]
            mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            mask_rgb[mask == 0] = [255, 255, 0]
            mask_rgb[mask == 1] = [0, 255, 0]
            mask_rgb[mask == 2] = [128, 0, 128]
            img=Image.fromarray(mask_rgb)

            combined = Image.new('RGB', (original.width + img.width+20, original.height+10))
            combined.paste(original, (5, 5))
            combined.paste(img, (original.width+15, 5))
            #combined.show()
            combined.save(f'./results/test_combined_{i*16+j}.png')

        if i >= 0:
            break