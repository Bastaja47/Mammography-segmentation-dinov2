"""
Sanity-check za arhitekturu modela (potpuno zamrznut backbone): koji delovi se
treniraju, koliko ima parametara i da li se oblik izlaza poklapa sa maskom.
"""

import torch
from torch.utils.data import DataLoader

from config import SPLIT_JSON
from dataset import InbreastMassDataset
from model import DinoSegmentationModel


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Koristim uredjaj: {device}")

    train_ds = InbreastMassDataset(SPLIT_JSON, "train")
    loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    batch = next(iter(loader))

    # n_unfrozen_blocks=0 znaci potpuno zamrznut backbone (baseline konfiguracija)
    model = DinoSegmentationModel(backbone_name="dinov2_vits14", n_unfrozen_blocks=0)
    model.to(device)
    model.train()

    # Zamrznut backbone mora ostati u eval modu cak i kad je model u train modu
    print("Backbone.training (ocekujemo False):", model.backbone.training)
    print("Head.training (ocekujemo True):     ", model.head.training)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nUkupno parametara u modelu:      {n_total:,}")
    print(f"Trenabilnih parametara (head):    {n_trainable:,}")
    print(f"Procenat trenabilnih:             {100 * n_trainable / n_total:.4f}%")

    images = batch["image"].to(device)
    masks = batch["mask"].to(device)

    logits = model(images)

    print(f"\nOblik ulaznih slika:   {images.shape}")
    print(f"Oblik izlaznih logita:  {logits.shape}")
    print(f"Oblik pravih maski:     {masks.shape}")

    assert logits.shape == masks.shape, "Oblik logita i maski MORA da se poklopi za racunanje loss-a!"
    print("\nOblici se poklapaju - model je spreman za trening.")

    if device == "cuda":
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nMaksimalno zauzeta GPU memorija u ovom testu: {mem_gb:.2f} GB")


if __name__ == "__main__":
    main()
