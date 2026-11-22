"""
Statisticko poredjenje uzastopnih konfiguracija (baseline -> augmentacija ->
fine-tuning N=2 -> N=4 -> N=6).

Za svaku konfiguraciju uzima najbolji val Dice po seed-u iz results/logs/ i
poredi je sa prethodnom konfiguracijom UPARENO (isti seed sa obe strane):
  - upareni t-test (dvostrani),
  - sign test (jednostrani: koliko seed-ova je poboljsano od ukupno 3).
Sa samo 3 seed-a ove p-vrednosti su orijentacione, ne tvrd dokaz.

Rezultat: results/summary/paired_ttests.csv
"""

import csv

from scipy import stats

from config import LOGS_DIR, SUMMARY_DIR

SEEDS = [123, 456, 789]

# Redosled je bitan: svaka konfiguracija se poredi sa prethodnom u listi
EXPERIMENTS = [
    "faza1_baseline",
    "faza5_augmentacija",
    "faza5_finetune_2blocks",
    "faza5_finetune_4blocks",
    "faza5_finetune_6blocks",
]


def best_val_dice(experiment, seed):
    """Najbolji val Dice kroz epohe za jedan run (isto sto bira early stopping)."""
    path = LOGS_DIR / f"{experiment}_seed{seed}_log.csv"
    with open(path, newline="") as f:
        return max(float(row["val_dice"]) for row in csv.DictReader(f))


def main():
    scores = {exp: [best_val_dice(exp, s) for s in SEEDS] for exp in EXPERIMENTS}

    rows = []
    for prev, curr in zip(EXPERIMENTS, EXPERIMENTS[1:]):
        diffs = [c - p for c, p in zip(scores[curr], scores[prev])]
        t_res = stats.ttest_rel(scores[curr], scores[prev])
        n_better = sum(d > 0 for d in diffs)
        sign_p = stats.binomtest(n_better, len(diffs), 0.5, alternative="greater").pvalue

        rows.append({
            "comparison": f"{curr} vs {prev}",
            "mean_diff_dice": sum(diffs) / len(diffs),
            "p_paired_ttest": t_res.pvalue,
            "seeds_improved": f"{n_better}/{len(diffs)}",
            "p_sign_test_one_sided": sign_p,
        })

    for r in rows:
        print(f"{r['comparison']:50s} diff={r['mean_diff_dice']:+.4f}  "
              f"t-test p={r['p_paired_ttest']:.4f}  "
              f"sign {r['seeds_improved']} p={r['p_sign_test_one_sided']:.3f}")

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUMMARY_DIR / "paired_ttests.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSacuvano: {out_path}")


if __name__ == "__main__":
    main()
