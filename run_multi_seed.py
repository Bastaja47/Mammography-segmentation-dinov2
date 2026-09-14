"""
run_multi_seed.py

Orkestrira vise treninga: za svaku konfiguraciju (baseline, augmentacija...),
trenira sa 3 razlicita seed-a, pa racuna prosek +- std - da poredjenje
bude statisticki pouzdano, ne slucajno


"""

import csv
from collections import defaultdict

import numpy as np

from train import train_one_run

SEEDS = [123, 456, 789]  # fiksiran, ponovljiv skup - isti seed-ovi za SVAKU
                           # konfiguraciju, da poredjenje bude "parno"
                           # (baseline sa seed=123 vs. augmentacija sa seed=123, itd.)

# Konfiguracije za OVO pokretanje - baseline, augmentacija, N=2 i N=4 su
# vec zavrseni (0.4995, 0.5169, 0.5532, 0.5768) - ne ponavljamo ih ovde.
# N=4 je NASTAVIO da poboljsava rezultat (suprotno hipotezi da bi opao) -
# probamo JOS JEDNU tacku, N=6, da vidimo da li trend i dalje raste ili
# konacno staje/opada.
CONFIGS = [
    {"experiment_name": "faza5_finetune_6blocks", "augment": True, "n_unfrozen_blocks": 6},
]


def main():
    # Ime sumarnih fajlova IZVODIMO automatski iz CONFIGS (spoj imena svih
    # eksperimenata u ovom pokretanju) - umesto da nam treba JOS JEDNA
    # promenljiva koju bismo mogli da zaboravimo da promenimo (isti obrazac
    # greske koji smo vec nekoliko puta ponovili sa EXPERIMENT_NAME).
    batch_id = "_".join(cfg["experiment_name"] for cfg in CONFIGS)
    raw_path = f"summary_raw__{batch_id}.csv"
    aggregated_path = f"summary_aggregated__{batch_id}.csv"

    all_results = []

    for cfg in CONFIGS:
        for seed in SEEDS:
            result = train_one_run(seed=seed, **cfg)
            all_results.append(result)

    # Sirovi rezultati - SVAKO pojedinacno pokretanje, da vidis i samu
    # varijansu, ne samo prosek (korisno za "Rezultati i diskusija").
    with open(raw_path, "w", newline="") as f:
        fieldnames = ["experiment", "seed", "augment", "n_unfrozen_blocks",
                      "best_val_dice", "best_val_iou", "best_epoch"]
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

    with open(aggregated_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "mean_val_dice", "std_val_dice", "n_seeds"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSacuvano: {raw_path} i {aggregated_path}")


if __name__ == "__main__":
    main()
