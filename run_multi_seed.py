"""
Pokrece svaku konfiguraciju sa vise seed-ova i racuna prosek +- std val Dice-a.

Jedan run na ovako malom skupu (13 val slika) ima veliku varijansu, pa se
svaka konfiguracija trenira sa 3 seed-a i poredi se prosek, a ne pojedinacni broj.
"""

import csv
from collections import defaultdict

import numpy as np

from config import SUMMARY_DIR
from train import train_one_run

SEEDS = [123, 456, 789]

# Sve konfiguracije iz eksperimenta. Imena odgovaraju logovima u results/logs/.
# Za pokretanje samo dela eksperimenta ostavi samo zeljene stavke u listi.
CONFIGS = [
    {"experiment_name": "faza1_baseline", "augment": False, "n_unfrozen_blocks": 0},
    {"experiment_name": "faza5_augmentacija", "augment": True, "n_unfrozen_blocks": 0},
    {"experiment_name": "faza5_finetune_2blocks", "augment": True, "n_unfrozen_blocks": 2},
    {"experiment_name": "faza5_finetune_4blocks", "augment": True, "n_unfrozen_blocks": 4},
    {"experiment_name": "faza5_finetune_6blocks", "augment": True, "n_unfrozen_blocks": 6},
]


def main():
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    batch_id = "_".join(cfg["experiment_name"] for cfg in CONFIGS)
    raw_path = SUMMARY_DIR / f"summary_raw__{batch_id}.csv"
    aggregated_path = SUMMARY_DIR / f"summary_aggregated__{batch_id}.csv"

    all_results = []

    for cfg in CONFIGS:
        for seed in SEEDS:
            result = train_one_run(seed=seed, **cfg)
            all_results.append(result)

    # Sirovi rezultati: jedan red po (konfiguracija, seed)
    with open(raw_path, "w", newline="") as f:
        fieldnames = ["experiment", "seed", "augment", "n_unfrozen_blocks",
                      "best_val_dice", "best_val_iou", "best_epoch"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in fieldnames})

    grouped = defaultdict(list)
    for r in all_results:
        grouped[r["experiment"]].append(r["best_val_dice"])

    print(f"\n{'='*60}")
    print("SUMARNI REZULTATI (prosek preko 3 seed-a)")
    print(f"{'='*60}")

    # Agregirani rezultati: prosek i std po konfiguraciji
    summary_rows = []
    for exp_name, dices in grouped.items():
        arr = np.array(dices)
        print(f"{exp_name:25s}: {arr.mean():.4f} +/- {arr.std():.4f}  "
              f"(pojedinacno: {[round(d, 4) for d in dices]})")
        summary_rows.append({
            "experiment": exp_name,
            "mean_val_dice": arr.mean(),
            "std_val_dice": arr.std(),
            "n_seeds": len(dices),
        })

    with open(aggregated_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "mean_val_dice", "std_val_dice", "n_seeds"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSacuvano: {raw_path} i {aggregated_path}")


if __name__ == "__main__":
    main()
