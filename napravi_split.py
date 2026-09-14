"""
napravi_split.py

Deli 107 slika sa Mass maskama na train/val/test, GRUPISANO PO PACIJENTU
(ne po pojedinacnoj slici), da izbegnemo curenje podataka (data leakage)

"""

import os
import re
import json
import random

ROOT = r"C:\Users\XYZ\Desktop\Principi dubokog ucenja\Projekat\INBreast\INBreast"

DICOM_DIR = os.path.join(ROOT, "AllDICOMs")
MASK_DIR = os.path.join(ROOT, "extras", "MassSegmentationMasks")


#Hiperparametri
# SEED pocetna vrednost generatora slucajnih brojeva, kad je fiksiran na 42 uvek se dobija identtican shuffle pacijenata
# TRAIN_FRAC to je procenat koji ide u trening
# VAL_FRAC procenat za validaciju
SEED = 42             # fiksiran broj -> IDENTICAN split svaki put kad pokrenes skriptu.
                       # Bez ovoga, svako pokretanje bi dalo drugaciju podelu, i ne bismo
                       # mogli da uporedimo rezultate izmedju razlicitih dana rada.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test frac je ostatak (~0.15) - racuna se automatski, ne pisemo ga eksplicitno

# Primer imena fajla koje ovaj regex mora da razume:
#   20586908_6c613a14b80a8591_MG_R_CC_ANON.dcm
#   \___1___/ \______2______/    3  \4/
# 1 = ID slike, 2 = "hash" pregleda/pacijenta, 3 = strana (L/R), 4 = pogled (CC/MLO)
#
# Zasto bas ova 4 dela: ID nam treba da spojimo sliku sa njenom maskom (isto
# kao ranije). Hash pacijenta nam treba za GRUPISANJE (objasnjeno gore). L/R i
# pogled ne koristimo direktno u splitu, ali ih cuvamo - mogu biti korisni
# kasnije (npr. za analizu "da li model radi bolje na CC ili MLO pogledu").
FILENAME_RE = re.compile(r"^(\d+)_([0-9a-fA-F]+)_MG_([LR])_(\w+)_ANON\.dcm$")


def build_dicom_index():
    """Vraca recnik: {id_slike: {path, patient_hash, laterality, view}}.

    Za razliku od ranije skripte (gde smo cuvali samo putanju), ovde cuvamo
    ceo mali paket informacija po slici - jer nam dalje u kodu treba i
    patient_hash, ne samo putanja do fajla
    """
    index = {}
    for fname in os.listdir(DICOM_DIR):
        m = FILENAME_RE.match(fname)
        if not m:
            continue  # ime fajla se ne poklapa sa ocekivanim obrascem - preskacemo
        img_id, patient_hash, lat, view = m.groups()  # .groups() vraca torku
                                                          # (id, hash, lat, view)
                                                          # jer regex ima 4 zagrade ( )
        index[img_id] = {
            "path": os.path.join(DICOM_DIR, fname),
            "patient_hash": patient_hash,
            "laterality": lat,
            "view": view,
        }
    return index


def build_mask_index():
    """Isto sto i ranije - {id_slike: putanja_do_maske}."""
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
        # Ovo NIJE nuzno greska - moze biti par fajlova drugacijeg formata
        # imena (npr. dodatni/testni snimci). Ali MORAMO da znamo da postoje,
        # ne da ih "tiho" izgubimo iz analize bez ikakvog traga.
        print("!! Neki fajlovi se ne poklapaju sa ocekivanim formatom imena - proveri ih.")

    mask_index = build_mask_index()
    common_ids = sorted(set(dicom_index) & set(mask_index))
    print(f"\nSlika sa maskom (nas ciljni skup za segmentaciju): {len(common_ids)}")

    # --- Grupisanje ID-jeva slika PO PACIJENTU ---
    # groups je recnik oblika: {"6c613a14b80a8591": ["20586908", "20586934", ...], ...}
    # Za svaki hash pacijenta, pravimo LISTU svih ID-jeva slika koje mu pripadaju.
    groups = {}
    for img_id in common_ids:
        ph = dicom_index[img_id]["patient_hash"]
        # .setdefault(ph, []) znaci: "ako ph JOS NIJE kljuc u recniku, napravi
        # mu praznu listu []; ako VEC jeste, samo vrati postojecu listu."
        # Onda na tu listu (bilo novu ili postojecu) dodajemo trenutni img_id.
        # Ovo je standardan Python idiom za "grupisanje stavki po kljucu"
        # bez potrebe da rucno proveravamo da li kljuc vec postoji.
        groups.setdefault(ph, []).append(img_id)

    patient_hashes = list(groups.keys())
    print(f"Broj razlicitih pacijenata/pregleda: {len(patient_hashes)}")
    print(f"Prosecno slika po pacijentu: {len(common_ids) / len(patient_hashes):.2f}")

    # --- Mesamo PACIJENTE (liste slika kao celinu), ne pojedinacne slike ---
    # random.Random(SEED) pravi NEZAVISAN generator slucajnih brojeva sa
    # fiksnim seed-om (za razliku od globalnog random.shuffle koji bi mogao
    # da bude pod uticajem drugog koda koji negde drugde u programu takodje
    # koristi random modul - odvojeni generator je cistije i predvidljivije).
    rng = random.Random(SEED)
    rng.shuffle(patient_hashes)  # menja listu "in-place" (ne pravi novu)

    n = len(patient_hashes)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    # ostatak (n - n_train - n_val) automatski ide u test

    # Isecanje liste (slicing) na 3 dela, po vec izmesanom redosledu:
    train_patients = set(patient_hashes[:n_train])
    val_patients = set(patient_hashes[n_train:n_train + n_val])
    test_patients = set(patient_hashes[n_train + n_val:])
    # pretvaramo u set() (ne listu) jer cemo odmah ispod raditi mnogo
    # "da li je ph U ovom skupu" provera - set proverava clanstvo skoro
    # trenutno (O(1)), dok bi provera nad listom bila znatno sporija
    # kako lista raste (O(n)) - kod nas razlika je zanemarljiva (107 slika),
    # ali je ovo opsti dobar refleks kad pises ovakav kod.

    # --- Sad prolazimo kroz SLIKE (ne pacijente) i raspodeljujemo ih
    # u odgovarajuci skup, prema tome kom skupu pripada NJIHOV pacijent ---
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

    # --- Cuvamo rezultat u JSON, spreman za citanje u Dataset klasi ---
    # Zasto bas JSON, a ne npr. samo isprintati brojeve: zelimo da ovaj
    # split bude TRAJAN i PONOVO UPOTREBLJIV - sledeci put kad pisemo
    # trening kod (baseline, pa adaptacija), citamo ISTI fajl, umesto da
    # split racunamo iznova (sto bi, i pored fiksnog seed-a, unelo rizik
    # da nesto omaskom promenimo i dobijemo drugaciju podelu).
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
