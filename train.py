"""
train.py
 
Trening logika - REFAKTORISANA u funkciju train_one_run(), da moze da se
poziva VISE PUTA (razliciti seed/konfiguracija) bez dupliranja koda.
Orkestracija vise poziva (3 seed-a x N konfiguracija) je u run_multi_seed.py.
 
=======================================================================
STA MORAS DA RAZUMES:
- Zasto je main() postao funkcija SA PARAMETRIMA (seed, experiment_name,
  augment) umesto koda direktno u if __name__ bloku - da bi mogla da se
  POZOVE vise puta iz drugog fajla, kao obicna funkcija, bez copy-paste
  cele trening petlje za svaki seed
- Sta funkcija VRACA (recnik rezultata) i zasto - POZIVALAC
  (run_multi_seed.py) sakuplja te recnike iz svih poziva da bi izracunao
  prosek/std na kraju
- Zasto se seed postavlja UNUTAR funkcije (na pocetku), ne samo jednom
  na vrhu fajla - svaki poziv mora da krene od SVOG kontrolisanog stanja
 
STA MOZES DA GUGLAS:
- Python default argumenti, kako tacno rade **kwargs
=======================================================================
"""
 
import csv
import random
 
import numpy as np
import torch
from torch.utils.data import DataLoader
 
from dataset import InbreastMassDataset
from model import DinoSegmentationModel
from losses import BCEDiceLoss, hard_dice_score, hard_iou_score
 
# Hiperparametri ZAJEDNICKI za sve eksperimente u ovoj fazi - ne menjamo
# ih izmedju konfiguracija, da poredjenje ostane fer (menjamo SAMO ono sto
# je predmet konkretnog eksperimenta - seed i augment).
SPLIT_JSON = "inbreast_split.json"
BATCH_SIZE = 8
NUM_EPOCHS = 60
LR = 1e-3
PATIENCE = 10
 
 
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    """Jedna epoha - trening ili validacija (nepromenjeno od ranije)."""
    model.train(train)
    total_loss, total_dice, total_iou, n_batches = 0.0, 0.0, 0.0, 0
 
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
 
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
 
 
def train_one_run(seed, experiment_name, augment, backbone_name="dinov2_vits14"):
    """Trenira JEDNU konfiguraciju, sa DATIM seed-om.
 
    Vraca recnik sa najboljim rezultatima - ovo je KLJUCNO za orkestraciju:
    run_multi_seed.py ce ovo pozvati 3x (razliciti seed) po konfiguraciji
    i sakupiti povratne vrednosti da izracuna prosek +- std.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Eksperiment: {experiment_name} | seed={seed} | augment={augment}")
    print(f"{'='*60}")
 
    checkpoint_path = f"{experiment_name}_seed{seed}_head.pt"
    log_csv_path = f"{experiment_name}_seed{seed}_log.csv"
 
    train_ds = InbreastMassDataset(SPLIT_JSON, "train", augment=augment)
    val_ds = InbreastMassDataset(SPLIT_JSON, "val")
 
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
 
    model = DinoSegmentationModel(backbone_name=backbone_name, freeze_backbone=True)
    model.to(device)
 
    criterion = BCEDiceLoss(bce_weight=0.5)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR)
 
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
 
        if val_stats["dice"] > best_val_dice:
            best_val_dice = val_stats["dice"]
            best_val_iou = val_stats["iou"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.head.state_dict(), checkpoint_path)
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
        "best_val_dice": best_val_dice,
        "best_val_iou": best_val_iou,
        "best_epoch": best_epoch,
        "checkpoint_path": checkpoint_path,
    }
 
 
if __name__ == "__main__":
    # Brzo, POJEDINACNO testiranje jedne konfiguracije (npr. za debug) -
    # za PRAVO poredjenje sa vise seed-ova, pokreni run_multi_seed.py
    train_one_run(seed=123, experiment_name="quick_test", augment=False)