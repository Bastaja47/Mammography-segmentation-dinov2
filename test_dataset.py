
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import IMAGENET_MEAN, IMAGENET_STD, InbreastMassDataset

SPLIT_JSON = "inbreast_split.json"

def denormalize_for_display(image_t):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = image_t * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()

def main():
    train_ds = InbreastMassDataset(SPLIT_JSON, "train")
    val_ds = InbreastMassDataset(SPLIT_JSON, "val")
    test_ds = InbreastMassDataset(SPLIT_JSON, "test")

    print(f"train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")

    loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    batch = next(iter(loader))

    print("\nOblik batch-a slika:", batch["image"].shape)
    print("Oblik batch-a maski: ", batch["mask"].shape)
    print("Opseg vrednosti slike (posle normalizacije): min=%.2f max=%.2f"
          % (batch["image"].min(), batch["image"].max()))
    print("Jedinstvene vrednosti u maski:", torch.unique(batch["mask"]))
    print("ID-jevi u batch-u:", batch["id"])

    fig, axes = plt.subplots(4, 3, figsize=(10, 13))
    for i in range(4):
        img_display = denormalize_for_display(batch["image"][i])
        mask_display = batch["mask"][i, 0].numpy()

        axes[i][0].imshow(img_display)
        axes[i][0].set_title(f"Slika (ID {batch['id'][i]})")
        axes[i][1].imshow(mask_display, cmap="gray")
        axes[i][1].set_title("Maska (nakon resize)")
        axes[i][2].imshow(img_display)
        axes[i][2].contour(mask_display, levels=[0.5], colors="red", linewidths=1.5)
        axes[i][2].set_title("Preklop")
        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig("test_dataset_output.png", dpi=130)
    print("\nSacuvano: test_dataset_output.png")

if __name__ == "__main__":
    main()
