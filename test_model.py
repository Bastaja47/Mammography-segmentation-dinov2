"""
test_model.py

Proverava da DinoSegmentationModel ispravno radi na PRAVIM podacima iz naseg
Dataset-a (ne izmisljenom tenzoru), PRE nego sto pisemo trening petlju.
Ne racunamo jos loss - samo proveravamo da se oblici (shapes) poklapaju
i da nam parametri izgledaju kako ocekujemo (mali broj trenabilnih).
"""

import torch
from torch.utils.data import DataLoader

from dataset import InbreastMassDataset
from model import DinoSegmentationModel

SPLIT_JSON = "inbreast_split.json"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Koristim uredjaj: {device}")

    train_ds = InbreastMassDataset(SPLIT_JSON, "train")
    # mali batch_size=2 - ovo je SAMO test da vidimo da li radi, ne trening;
    # pravi batch_size za trening biramo u sledecem koraku, uz obzir na
    # 6GB VRAM ogranicenje
    loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    batch = next(iter(loader))

    model = DinoSegmentationModel(backbone_name="dinov2_vits14", freeze_backbone=True)
    model.to(device)
    model.train()  # namerno testiramo bas u train() modu - da potvrdimo da
                    # nas override ispravno drzi backbone u eval()-u

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
    print("\nOblici se poklapaju - model je spreman za pisanje trening petlje.")

    if device == "cuda":
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nMaksimalno zauzeta GPU memorija u ovom testu: {mem_gb:.2f} GB")


if __name__ == "__main__":
    main()
