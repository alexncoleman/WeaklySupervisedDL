import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from ExtraUtils import compute_iou_and_acc

class LayerCAMGenerator:
    def __init__(self, model, target_layer_names=["layer3", "layer4"]):
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


    def generate(self, images, alpha, class_idx=None):
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
        final_cam = (final_cam).clamp(min=0.0) ** alpha

        del logits, class_scores, images, layer_cams
        torch.cuda.empty_cache()

        return final_cam  # (B, H, W)


def evaluate_layercam_on_test_set(layercam_gen, test_loader, alpha=1.0, cam_thresh=0.3):
    """
    Evaluates LayerCAM and standard CAM foreground masks on the test set using IoU and pixel accuracy.
    """
    ious_fg, accs_fg = [], []
    ious_bg, accs_bg = [], []

    for i, (img, (label, true_mask)) in enumerate(test_loader):
        img = img[0].cuda()
        true_mask = true_mask[0].cuda()

        # Binarize true mask: class 1 = foreground
        true_mask = (true_mask == 1).long()
        label = label[0].item() if isinstance(label[0], torch.Tensor) else label[0]
        class_tensor = torch.tensor([label]).to(img.device)

        # Generate CAM masks
        layercam_output = layercam_gen.generate(img, class_idx=class_tensor, alpha=alpha)

        layercam = layercam_output.squeeze(0)

        layercam[layercam < cam_thresh] = 0.0

        pred_fg_mask = (layercam > 0.0).long()

        # Resize if needed
        if pred_fg_mask.shape != true_mask.shape:
            pred_fg_mask = F.interpolate(pred_fg_mask.unsqueeze(0).unsqueeze(0).float(), size=true_mask.shape[-2:], mode='nearest').squeeze().long()

        # Compute metrics
        iou_fg, acc_fg = compute_iou_and_acc(pred_fg_mask, true_mask)

        ious_fg.append(iou_fg)
        accs_fg.append(acc_fg)

        if i >= 10:
          break # ablations taking too long so only test 10 images


    print("\n Evaluation of CAMs on test set:")
    print(f" - LayerCam FG: Avg IoU: {sum(ious_fg)/len(ious_fg):.4f} | Acc: {sum(accs_fg)/len(accs_fg):.4f}")


    return {
        "layercam_fg_iou": sum(ious_fg) / len(ious_fg),
        "layercam_fg_acc": sum(accs_fg) / len(accs_fg)
    }