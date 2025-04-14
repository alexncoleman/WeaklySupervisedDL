from LayerCAM import LayerCAMGenerator #, evaluate_layercam_on_test_set
from SegmentationModel import evaluate_model, train_segmentation_model
from PseudoMasks import generate_pseudo_masks, keep_largest
import torch
from statistics import mean, stdev
import itertools
from ClassficationModel import FrozenResNetCAM

def run_ablation(classifier, save_path, loader, test_loader, cam_method, cam_thresh, alpha, lr, keep_largest, run_id):

    classifier.load_state_dict(torch.load(save_path))
    classifier.eval()

    # cam_gen = CAMGenerator(classifier)
    layercam_gen = LayerCAMGenerator(classifier, target_layer_names=["layer3", "layer4"])

    # 1. (Optional) Evaluate LayerCAMs pre-training
    #evaluate_layercam_on_test_set(layercam_gen, cam_gen, test_loader, alpha, cam_thresh, cam_thresh_bg)

    # 2. Generate pseudo masks
    generate_pseudo_masks(loader, layercam_gen, cam_thresh, alpha, keep_largest, run_id)

    # 3. Train model
    model, final_loss = train_segmentation_model(run_id, lr, num_epochs = 5)

    # 4. Evaluate model
    iou, acc = evaluate_model(model, test_loader)

    return {"run_id": run_id, "iou": iou, "acc": acc, "final_loss": final_loss}


def run_abalation_experiment(all_combinations):
    results = []
    num_repeats = 3

    for combo_id, (method, cam_thresh, alpha, lr, keep_largest_opt) in enumerate(all_combinations):
        run_results = []
        for repeat in range(num_repeats):
            run_id = f"abl_{combo_id:03d}_r{repeat}"
            print(f"\n Running {run_id}...")

            classifier = FrozenResNetCAM(num_classes=37, pretrained=True, freeze=True, use_cam=True)
            save_path = f"../Weights/classifier_weights_without_affinity.pth"
            result = run_ablation(
                    cam_method=method,
                    cam_thresh=cam_thresh,
                    alpha=alpha,
                    lr=lr,
                    keep_largest=keep_largest,
                    run_id=run_id
                )
            result.update({
                    "cam_method": method,
                    "cam_thresh": cam_thresh,
                    "alpha": alpha,
                    "learning_rate": lr,
                    "keep_largest": keep_largest_opt
                })
            results.append(result)
            run_results.append(result)

        if run_results:
            ious = [r["iou"] for r in run_results]
            accs = [r["acc"] for r in run_results]
            losses = [r["final_loss"] for r in run_results]

            summary = {
                "combo_id": combo_id,
                "cam_method": method,
                "cam_thresh": cam_thresh,
                "alpha": alpha,
                "learning_rate": lr,
                "keep_largest": keep_largest,
                "iou_mean": mean(ious),
                "iou_std": stdev(ious) if len(ious) > 1 else 0.0,
                "acc_mean": mean(accs),
                "acc_std": stdev(accs) if len(accs) > 1 else 0.0,
                "loss_mean": mean(losses),
                "loss_std": stdev(losses) if len(losses) > 1 else 0.0
            }
            results.append(summary)


if __name__ == "__main__":

    cam_methods = ['LayerCAM']
    cam_thresholds = [0.3, 0.5, 0.7]
    alphas = [1.0]
    lrs = [1e-2, 1e-3, 1e-4, 1e-5]
    keep_largest_opts = [True]

    # Create all combinations
    all_combinations = list(itertools.product(
        cam_methods, cam_thresholds, alphas, lrs, keep_largest_opts
    ))
    run_abalation_experiment(all_combinations)