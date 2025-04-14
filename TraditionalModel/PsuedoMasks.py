from skimage.measure import label as lb, regionprops
import os 
import numpy as np

def delete_dir_recursive(path):
    if os.path.exists(path):
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(path)


def keep_largest(mask):
    labeled = lb(mask)
    regions = regionprops(labeled)
    if not regions:
        return mask
    largest = max(regions, key=lambda r: r.area)
    return (labeled == largest.label).astype(np.uint8)

def generate_pseudo_masks(
    loader,
    layercam_gen,
    cam_thresh=0.3,
    alpha=1.0,
    keep_largest_masks=True,
    run_id="default"):

    save_dir = f"/content/pseudo_masks_{run_id}"
    image_save_dir = f"/content/images_{run_id}"
    for d in [save_dir, image_save_dir]:
      if os.path.exists(d):
        delete_dir_recursive(d)
      os.makedirs(d)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_id = 0

    for imgs, (labels, _) in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        batch_size = imgs.size(0)

        for i in range(batch_size):

          if img_id >= 500:
            break

          img = imgs[i]
          label = labels[i].item()

          #labels = label.item() if isinstance(label[0], torch.Tensor) else label[0]
          class_tensor = torch.tensor([label]).to(device)

          layercam = layercam_gen.generate(img, class_idx=class_tensor)
          layercam = layercam.squeeze(0)
          layercam[layercam < cam_thresh] = 0.0

          refined_mask = (layercam.cpu().numpy() > 0).astype(np.uint8)

          if keep_largest_masks:
              refined_mask = keep_largest(refined_mask)

          # Save pseudo mask
          mask_path = os.path.join(save_dir, f"{img_id}.png")
          save_image(torch.from_numpy(refined_mask).float().unsqueeze(0), mask_path)

          # Save image
          img_orig = img.cpu().clone()
          img_orig = (img_orig - img_orig.min()) / (img_orig.max() - img_orig.min())
          save_image(img_orig, os.path.join(image_save_dir, f"{img_id}.png"))

          img_id += 1
    print(f"Pseudo masks saved to: {save_dir}")
    print(f"Images saved to: {image_save_dir}")
    return image_save_dir, save_dir
