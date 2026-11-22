"""
Trening jedne konfiguracije (jedan seed) segmentacionog modela.

Poziva se iz run_multi_seed.py za svaki seed i svaku konfiguraciju, a moze i
direktno (`python train.py`) kao brzi test na jednom seed-u.
"""

import csv
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CHECKPOINT_DIR, LOGS_DIR, SPLIT_JSON
from dataset import InbreastMassDataset
from model import DinoSegmentationModel
from losses import BCEDiceLoss, hard_dice_score, hard_iou_score

BATCH_SIZE = 8
NUM_EPOCHS = 60
LR_HEAD = 1e-3       # glava je nasumicno inicijalizovana pa sme veci learning rate
LR_BACKBONE = 1e-5   # odmrznuti blokovi: 100x manji LR da ne "zaboravimo" pretrenirano znanje
PATIENCE = 10        # early stopping na validacionom Dice-u


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    """Jedna epoha nad loader-om. Vraca prosecan loss, Dice i IoU."""
    model.train(train)
    total_loss, total_dice, total_iou, n_batches = 0.0, 0.0, 0.0, 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        # Gradijenti se racunaju samo tokom treninga
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, masks)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_dice += hard_dice_score(logits, masks)
        total_iou += hard_iou_score(logits, masks)
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "dice": total_dice / n_batches,
        "iou": total_iou / n_batches,
    }


def train_one_run(seed, experiment_name, augment, n_unfrozen_blocks=0,
                   backbone_name="dinov2_vits14"):
    """
    Trenira model za dati seed i konfiguraciju.

    Cuva najbolji checkpoint (po val Dice-u) u checkpoints/ i log po epohama u
    results/logs/. Vraca recnik sa najboljim rezultatima za run_multi_seed.py.
    """
    # Fiksiramo sve izvore slucajnosti da bi run bio ponovljiv
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Eksperiment: {experiment_name} | seed={seed} | augment={augment} "
          f"| n_unfrozen_blocks={n_unfrozen_blocks}")
    print(f"{'='*60}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{experiment_name}_seed{seed}_model.pt"
    log_csv_path = LOGS_DIR / f"{experiment_name}_seed{seed}_log.csv"

    # Augmentacija se ukljucuje samo za train, val ostaje neizmenjen
    train_ds = InbreastMassDataset(SPLIT_JSON, "train", augment=augment)
    val_ds = InbreastMassDataset(SPLIT_JSON, "val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = DinoSegmentationModel(backbone_name=backbone_name, n_unfrozen_blocks=n_unfrozen_blocks)
    model.to(device)

    criterion = BCEDiceLoss(bce_weight=0.5)

    # Dve grupe parametara sa razlicitim learning rate-om (vidi konstante gore)
    head_params = list(model.head.parameters())
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]

    param_groups = [{"params": head_params, "lr": LR_HEAD}]
    if len(backbone_params) > 0:
        param_groups.append({"params": backbone_params, "lr": LR_BACKBONE})
        print(f"Fine-tuning: {len(backbone_params)} tenzora u odmrznutom delu backbone-a")

    optimizer = torch.optim.AdamW(param_groups)

    best_val_dice = 0.0
    best_val_iou = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_stats = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_stats = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(
            f"Epoha {epoch:3d}/{NUM_EPOCHS} | "
            f"train dice={train_stats['dice']:.4f} | "
            f"val dice={val_stats['dice']:.4f} iou={val_stats['iou']:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_stats["loss"], "train_dice": train_stats["dice"],
            "val_loss": val_stats["loss"], "val_dice": val_stats["dice"], "val_iou": val_stats["iou"],
        })

        # Model biramo po validacionom Dice-u (test skup se ovde NE koristi)
        if val_stats["dice"] > best_val_dice:
            best_val_dice = val_stats["dice"]
            best_val_iou = val_stats["iou"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping (nema poboljsanja {PATIENCE} epoha).")
            break

    with open(log_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print(f"Zavrseno. Najbolji val Dice: {best_val_dice:.4f} (epoha {best_epoch})")

    return {
        "experiment": experiment_name,
        "seed": seed,
        "augment": augment,
        "n_unfrozen_blocks": n_unfrozen_blocks,
        "best_val_dice": best_val_dice,
        "best_val_iou": best_val_iou,
        "best_epoch": best_epoch,
        "checkpoint_path": str(checkpoint_path),
    }


if __name__ == "__main__":
    # Brzi test jedne konfiguracije (fine-tuning 2 bloka, jedan seed)
    train_one_run(seed=123, experiment_name="quick_test_finetune", augment=True, n_unfrozen_blocks=2)
