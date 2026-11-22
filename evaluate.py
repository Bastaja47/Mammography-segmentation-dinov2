"""
Finalna evaluacija na test skupu (pokrece se tek kad je konfiguracija izabrana
na osnovu validacionog skupa)

Evaluiraju se sva tri istrenirana modela (po jedan za svaki seed), a ne samo
najbolji, da ne bismo birali model na osnovu test rezultata
"""

import csv

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CHECKPOINT_DIR, SPLIT_JSON, SUMMARY_DIR
from dataset import InbreastMassDataset
from model import DinoSegmentationModel
from losses import hard_dice_score, hard_iou_score, hard_sensitivity_score, hard_precision_score

BACKBONE_NAME = "dinov2_vits14"
N_UNFROZEN_BLOCKS = 6
SEEDS = [123, 456, 789]
CHECKPOINT_TEMPLATE = "faza5_finetune_6blocks_seed{seed}_model.pt"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Koristim uredjaj: {device}")

    test_ds = InbreastMassDataset(SPLIT_JSON, "test")
    print(f"Test skup: {len(test_ds)} slika")

    # Ceo test skup ide u1 batch (17 slika). Tako je prosek metrika po slikama
    # tacan i ne zavisi od toga kako se skup podeli na batch-eve razlicite velicine
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

    all_results = []

    for seed in SEEDS:
        checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_TEMPLATE.format(seed=seed)
        print(f"\nUcitavam: {checkpoint_path}")

        model = DinoSegmentationModel(backbone_name=BACKBONE_NAME, n_unfrozen_blocks=N_UNFROZEN_BLOCKS)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        batch = next(iter(test_loader))
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        with torch.no_grad():
            logits = model(images)

        metrics = {
            "seed": seed,
            "dice": hard_dice_score(logits, masks),
            "iou": hard_iou_score(logits, masks),
            "sensitivity": hard_sensitivity_score(logits, masks),
            "precision": hard_precision_score(logits, masks),
        }
        all_results.append(metrics)

        print(f"  Dice={metrics['dice']:.4f}  IoU={metrics['iou']:.4f}  "
              f"Sensitivity={metrics['sensitivity']:.4f}  Precision={metrics['precision']:.4f}")

    print(f"\n{'='*60}")
    print("FINALNI TEST REZULTATI (prosek +- std, 3 seed-a)")
    print(f"{'='*60}")
    for metric_name in ["dice", "iou", "sensitivity", "precision"]:
        values = np.array([r[metric_name] for r in all_results])
        print(f"{metric_name:12s}: {values.mean():.4f} +/- {values.std():.4f}")

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUMMARY_DIR / "final_test_evaluation.csv"
    with open(out_path, "w", newline="") as f:
        fieldnames = ["seed", "dice", "iou", "sensitivity", "precision"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"\nSacuvano: {out_path}")


if __name__ == "__main__":
    main()
