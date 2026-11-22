"""
Deli INBreast slike (one koje imaju masku mase) na train/val/test.

Podela je po PACIJENTU, a ne po slici: isti pacijent ima u proseku ~2 slike,
pa bi podela po slici mogla da stavi jednu u train a drugu u test (curenje
podataka, lazno dobar rezultat). Pacijenta prepoznajemo po hash delu imena
DICOM fajla, jer je kolona 'Patient ID' u originalnom CSV-u uklonjena.

Rezultat je inbreast_split.json sa putanjama relativnim u odnosu na INBREAST_ROOT.
"""

import os
import re
import json
import random

from config import DICOM_DIR, MASK_DIR, SPLIT_JSON

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# Primer imena: 20586908_6c613a14b80a8591_MG_R_CC_ANON.dcm
# -> id slike, hash pacijenta, strana (L/R), projekcija (CC/MLO)
FILENAME_RE = re.compile(r"^(\d+)_([0-9a-fA-F]+)_MG_([LR])_(\w+)_ANON\.dcm$")


def build_dicom_index():
    """Mapira id slike -> relativna putanja i metapodaci iz imena fajla."""
    index = {}
    for fname in os.listdir(DICOM_DIR):
        m = FILENAME_RE.match(fname)
        if not m:
            continue
        img_id, patient_hash, lat, view = m.groups()
        index[img_id] = {
            "path": f"AllDICOMs/{fname}",
            "patient_hash": patient_hash,
            "laterality": lat,
            "view": view,
        }
    return index


def build_mask_index():
    """Mapira id slike -> relativna putanja do maske mase."""
    index = {}
    for fname in os.listdir(MASK_DIR):
        m = re.match(r"(\d+)_mask", fname)
        if m:
            index[m.group(1)] = f"extras/MassSegmentationMasks/{fname}"
    return index


def main():
    all_dcm = [f for f in os.listdir(DICOM_DIR) if f.lower().endswith(".dcm")]
    dicom_index = build_dicom_index()
    print(f"Ukupno .dcm fajlova u folderu: {len(all_dcm)}")
    print(f"Uspesno isparsirano regex-om: {len(dicom_index)}")
    if len(dicom_index) < len(all_dcm):
        print("!! Neki fajlovi se ne poklapaju sa ocekivanim formatom imena - proveri ih.")

    # Ciljni skup: samo slike koje imaju i DICOM i masku mase
    mask_index = build_mask_index()
    common_ids = sorted(set(dicom_index) & set(mask_index))
    print(f"\nSlika sa maskom (nas ciljni skup za segmentaciju): {len(common_ids)}")

    # Grupisanje slika po pacijentu
    groups = {}
    for img_id in common_ids:
        ph = dicom_index[img_id]["patient_hash"]
        groups.setdefault(ph, []).append(img_id)

    patient_hashes = list(groups.keys())
    print(f"Broj razlicitih pacijenata/pregleda: {len(patient_hashes)}")
    print(f"Prosecno slika po pacijentu: {len(common_ids) / len(patient_hashes):.2f}")

    # Mesamo PACIJENTE (fiksan seed = ponovljiva podela), pa ih delimo 70/15/15
    rng = random.Random(SEED)
    rng.shuffle(patient_hashes)

    n = len(patient_hashes)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    train_patients = set(patient_hashes[:n_train])
    val_patients = set(patient_hashes[n_train:n_train + n_val])
    test_patients = set(patient_hashes[n_train + n_val:])

    split = {"train": [], "val": [], "test": []}
    for img_id in common_ids:
        ph = dicom_index[img_id]["patient_hash"]
        if ph in train_patients:
            split["train"].append(img_id)
        elif ph in val_patients:
            split["val"].append(img_id)
        else:
            split["test"].append(img_id)

    print()
    for name, ids in split.items():
        n_patients = len(set(dicom_index[i]["patient_hash"] for i in ids))
        print(f"{name:5s}: {len(ids):3d} slika, {n_patients:2d} pacijenata")

    records = []
    for name, ids in split.items():
        for img_id in ids:
            records.append({
                "id": img_id,
                "split": name,
                "dicom_path": dicom_index[img_id]["path"],
                "mask_path": mask_index[img_id],
                "patient_hash": dicom_index[img_id]["patient_hash"],
                "laterality": dicom_index[img_id]["laterality"],
                "view": dicom_index[img_id]["view"],
            })

    with open(SPLIT_JSON, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSacuvano: {SPLIT_JSON} ({len(records)} zapisa)")


if __name__ == "__main__":
    main()
