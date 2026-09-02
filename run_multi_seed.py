"""
run_multi_seed.py

Orkestrira VISE treninga: za svaku konfiguraciju (baseline, augmentacija...),
trenira sa 3 razlicita seed-a, pa racuna prosek +- std - da poredjenje
bude STATISTICKI POUZDANO, ne slucajno.

Podsetnik ZASTO nam ovo treba: dva ISTA baseline treninga (razlicit seed)
dala su val Dice 0.4748 i 0.5026 - razlika 0.0278, VECA od "poboljsanja"
koje smo pogresno pripisali augmentaciji (0.0103) na osnovu SAMO jednog
pokretanja svake konfiguracije.

=======================================================================
STA MORAS DA RAZUMES:
- Zasto vise seed-ova po konfiguraciji (upravo objasnjeno gore)
- Sta prosek/std govore: mala std = pouzdan, ponovljiv rezultat;
  velika std = rezultat je uglavnom "sum", ne treba mu previse verovati
- Kako citati finalnu tabelu da odlucis da li je razlika izmedju dve
  konfiguracije STVARNA (npr. razlika proseka > zbir std-ova) ili sum

STA MOZES DA GUGLAS:
- collections.defaultdict sintaksa
- numpy .mean()/.std()
=======================================================================
"""

import csv
from collections import defaultdict

import numpy as np

from train import train_one_run

SEEDS = [123, 456, 789]  # fiksiran, ponovljiv skup - isti seed-ovi za SVAKU
                           # konfiguraciju, da poredjenje bude "parno"
                           # (baseline sa seed=123 vs. augmentacija sa seed=123, itd.)

# Konfiguracije koje poredimo - svaka ce biti trenirana 3x (jednom po seed-u)
CONFIGS = [
    {"experiment_name": "faza1_baseline", "augment": False},
    {"experiment_name": "faza5_augmentacija", "augment": True},
]


def main():
    all_results = []

    for cfg in CONFIGS:
        for seed in SEEDS:
            result = train_one_run(seed=seed, **cfg)
            all_results.append(result)

    # Sirovi rezultati - SVAKO pojedinacno pokretanje, da vidis i samu
    # varijansu, ne samo prosek (korisno za "Rezultati i diskusija").
    with open("experiment_summary_raw.csv", "w", newline="") as f:
        fieldnames = ["experiment", "seed", "augment", "best_val_dice", "best_val_iou", "best_epoch"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in fieldnames})

    # Agregacija po konfiguraciji - grupisi po experiment_name (recnik
    # {ime_eksperimenta: [dice1, dice2, dice3]}), pa izracunaj prosek/std.
    grouped = defaultdict(list)
    for r in all_results:
        grouped[r["experiment"]].append(r["best_val_dice"])

    print(f"\n{'='*60}")
    print("SUMARNI REZULTATI (prosek preko 3 seed-a)")
    print(f"{'='*60}")

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

    with open("experiment_summary_aggregated.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "mean_val_dice", "std_val_dice", "n_seeds"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nSacuvano: experiment_summary_raw.csv i experiment_summary_aggregated.csv")


if __name__ == "__main__":
    main()
