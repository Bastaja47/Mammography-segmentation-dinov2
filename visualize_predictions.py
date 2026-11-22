"""
Vizuelizacija predikcija modela na test slikama.

Bira se model (seed) cija je test Dice vrednost NAJBLIZA prosecnoj, a ne
najbolji, da vizuelizacija ne bi bila "cherry-picking". Prikazuju se cetiri
slike: najgora, dve tipicne i najbolja (po Dice-u).
"""

import csv

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import CHECKPOINT_DIR, FIGURES_DIR, SPLIT_JSON, SUMMARY_DIR
from dataset import InbreastMassDataset, IMAGENET_MEAN, IMAGENET_STD
from model import DinoSegmentationModel
from losses import hard_dice_score

BACKBONE_NAME = "dinov2_vits14"
N_UNFROZEN_BLOCKS = 6
CHECKPOINT_TEMPLATE = "faza5_finetune_6blocks_seed{seed}_model.pt"


def load_test_dice_per_seed():
    """Cita test Dice po seed-u iz results/summary/final_test_evaluation.csv (izlaz evaluate.py)."""
    with open(SUMMARY_DIR / "final_test_evaluation.csv", newline="") as f:
        return {int(r["seed"]): float(r["dice"]) for r in csv.DictReader(f)}


def denormalize_for_display(image_t):
    """Vraca normalizovanu sliku u opseg [0, 1] da bi mogla da se prikaze."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = image_t * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_test_dice = load_test_dice_per_seed()
    mean_dice = float(np.mean(list(seed_test_dice.values())))
    chosen_seed = min(seed_test_dice, key=lambda s: abs(seed_test_dice[s] - mean_dice))
    print(f"Koristim seed={chosen_seed} (najblizi proseku {mean_dice:.4f}) - "
          f"NE najbolji seed, da izbegnemo cherry-picking u vizuelizaciji.")

    checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_TEMPLATE.format(seed=chosen_seed)
    model = DinoSegmentationModel(backbone_name=BACKBONE_NAME, n_unfrozen_blocks=N_UNFROZEN_BLOCKS)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    test_ds = InbreastMassDataset(SPLIT_JSON, "test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # Predikcija za svaku test sliku, sa Dice-om po slici
    per_image_results = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            dice = hard_dice_score(logits, masks)

            per_image_results.append({
                "id": batch["id"][0],
                "dice": dice,
                "image": images[0].cpu(),
                "mask": masks[0, 0].cpu().numpy(),
                "pred": (torch.sigmoid(logits[0, 0]) > 0.5).float().cpu().numpy(),
            })

    per_image_results.sort(key=lambda r: r["dice"])
    n = len(per_image_results)
    print(f"\nDice po test slici (sortirano, najgori->najbolji):")
    for r in per_image_results:
        print(f"  ID {r['id']}: {r['dice']:.4f}")

    # Najgora, dve iz sredine (tipicne) i najbolja
    selected_idx = [0, n // 3, (2 * n) // 3, n - 1]
    selected = [per_image_results[i] for i in selected_idx]
    labels = ["Najgori primer", "Tipican primer 1", "Tipican primer 2", "Najbolji primer"]

    fig, axes = plt.subplots(len(selected), 3, figsize=(10, 3.4 * len(selected)))
    for row, (res, label) in enumerate(zip(selected, labels)):
        img_display = denormalize_for_display(res["image"])

        axes[row][0].imshow(img_display)
        axes[row][0].contour(res["mask"], levels=[0.5], colors="lime", linewidths=2)
        axes[row][0].contour(res["pred"], levels=[0.5], colors="red", linewidths=2, linestyles="dashed")
        axes[row][0].set_title(
            f"{label} (ID {res['id']}, Dice={res['dice']:.3f})\nzeleno=prava masa, crveno(isprek.)=predikcija",
            fontsize=9,
        )
        axes[row][0].axis("off")

        axes[row][1].imshow(res["mask"], cmap="gray")
        axes[row][1].set_title("Prava maska", fontsize=9)
        axes[row][1].axis("off")

        axes[row][2].imshow(res["pred"], cmap="gray")
        axes[row][2].set_title("Predikcija modela", fontsize=9)
        axes[row][2].axis("off")

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "predikcije_primeri.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSacuvano: {out_path}")


if __name__ == "__main__":
    main()
