"""
visualize_predictions.py

Vizuelizacija predikcija pobednickog modela (fine-tuning N=6) na test
skupu - za slike u finalnom izvestaju.

"""

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import InbreastMassDataset, IMAGENET_MEAN, IMAGENET_STD
from model import DinoSegmentationModel
from losses import hard_dice_score

SPLIT_JSON = "inbreast_split.json"
BACKBONE_NAME = "dinov2_vits14"
N_UNFROZEN_BLOCKS = 6
CHECKPOINT_TEMPLATE = "faza5_finetune_6blocks_seed{seed}_model.pt"

# Test Dice po seed-u, iz evaluate.py izlaza - koristimo da izaberemo
# NAJBLIZI PROSEKU seed za vizuelizaciju (ne najbolji - cherry-picking).
SEED_TEST_DICE = {123: 0.7885, 456: 0.7702, 789: 0.7851}
MEAN_DICE = 0.7813


def denormalize_for_display(image_t):
    """Poništava ImageNet normalizaciju - SAMO za prikaz, mreza i dalje
    dobija normalizovane vrednosti u stvarnom radu."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = image_t * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    chosen_seed = min(SEED_TEST_DICE, key=lambda s: abs(SEED_TEST_DICE[s] - MEAN_DICE))
    print(f"Koristim seed={chosen_seed} (najblizi proseku {MEAN_DICE:.4f}) - "
          f"NE najbolji seed, da izbegnemo cherry-picking u vizuelizaciji.")

    checkpoint_path = CHECKPOINT_TEMPLATE.format(seed=chosen_seed)
    model = DinoSegmentationModel(backbone_name=BACKBONE_NAME, n_unfrozen_blocks=N_UNFROZEN_BLOCKS)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    test_ds = InbreastMassDataset(SPLIT_JSON, "test")
    # batch_size=1 OVDE namerno (razlicito od evaluate.py) - treba nam
    # Dice PO POJEDINACNOJ slici da bismo mogli da rangiramo primere.
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    per_image_results = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            dice = hard_dice_score(logits, masks)  # batch=1 -> Dice TE JEDNE slike

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

    # Biramo 4 primera: najgori, dva "tipicna" (blizu medijane), najbolji -
    # NE samo lepe primere.
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
    plt.savefig("predikcije_primeri.png", dpi=150)
    print("\nSacuvano: predikcije_primeri.png")


if __name__ == "__main__":
    main()
