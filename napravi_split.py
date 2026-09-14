
import os
import re
import json
import random

ROOT = r"C:\Users\XYZ\Desktop\Principi dubokog ucenja\Projekat\INBreast\INBreast"

DICOM_DIR = os.path.join(ROOT, "AllDICOMs")
MASK_DIR = os.path.join(ROOT, "extras", "MassSegmentationMasks")

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

FILENAME_RE = re.compile(r"^(\d+)_([0-9a-fA-F]+)_MG_([LR])_(\w+)_ANON\.dcm$")

def build_dicom_index():
    index = {}
    for fname in os.listdir(DICOM_DIR):
        m = FILENAME_RE.match(fname)
        if not m:
            continue
        img_id, patient_hash, lat, view = m.groups()
        index[img_id] = {
            "path": os.path.join(DICOM_DIR, fname),
            "patient_hash": patient_hash,
            "laterality": lat,
            "view": view,
        }
    return index

def build_mask_index():
    index = {}
    for fname in os.listdir(MASK_DIR):
        m = re.match(r"(\d+)_mask", fname)
        if m:
            index[m.group(1)] = os.path.join(MASK_DIR, fname)
    return index

def main():
    all_dcm = [f for f in os.listdir(DICOM_DIR) if f.lower().endswith(".dcm")]
    dicom_index = build_dicom_index()
    print(f"Ukupno .dcm fajlova u folderu: {len(all_dcm)}")
    print(f"Uspesno isparsirano regex-om: {len(dicom_index)}")
    if len(dicom_index) < len(all_dcm):
        print("!! Neki fajlovi se ne poklapaju sa ocekivanim formatom imena - proveri ih.")

    mask_index = build_mask_index()
    common_ids = sorted(set(dicom_index) & set(mask_index))
    print(f"\nSlika sa maskom (nas ciljni skup za segmentaciju): {len(common_ids)}")

    groups = {}
    for img_id in common_ids:
        ph = dicom_index[img_id]["patient_hash"]
        groups.setdefault(ph, []).append(img_id)

    patient_hashes = list(groups.keys())
    print(f"Broj razlicitih pacijenata/pregleda: {len(patient_hashes)}")
    print(f"Prosecno slika po pacijentu: {len(common_ids) / len(patient_hashes):.2f}")

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

    out_path = "inbreast_split.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSacuvano: {out_path} ({len(records)} zapisa)")

if __name__ == "__main__":
    main()
