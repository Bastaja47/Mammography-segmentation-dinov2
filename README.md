# Semantička segmentacija masa u mamografskim slikama pomoću DINOv2 modela

Adaptacija pretreniranog, self-supervised **DINOv2** modela (Oquab i dr., 2024, ViT-S/14)
za semantičku segmentaciju masa u mamografskim slikama iz **INBreast** baze podataka.
Projekat realizovan u okviru predmeta *Principi dubokog učenja*.

## Sadržaj

- [Opis problema](#opis-problema)
- [Pristup](#pristup)
- [Struktura repozitorijuma](#struktura-repozitorijuma)
- [Instalacija](#instalacija)
- [Upotreba](#upotreba)
- [Rezultati](#rezultati)
- [Metodološke napomene](#metodološke-napomene)
- [Reference](#reference)

## Opis problema

Mamografija je standardni skrining metod za rano otkrivanje raka dojke. Precizna
segmentacija masa (lezija) pomaže dijagnostičkom procesu, ali ručna anotacija zahteva
iskusnog radiologa i skupa je da se sprovede na velikoj skali. Ovaj projekat ispituje
koliko se dobro **generičke, self-supervised naučene vizuelne osobine** (DINOv2,
treniran isključivo na prirodnim fotografijama, bez ijedne medicinske slike) mogu
preneti na segmentaciju masa uz minimalnu količinu dodatnih anotiranih podataka.

## Pristup

1. **Baseline** — potpuno zamrznut DINOv2 backbone (ViT-S/14) + trenabilna linearna
   segmentaciona glava (1×1 konvolucija nad patch tokenima).
2. **Augmentacija podataka** — geometrijske i intenzitetske transformacije nad trening
   skupom, radi smanjenja overfitting-a na malom broju primera.
3. **Postepen fine-tuning** — odmrzavanje poslednjih *N* transformer blokova backbone-a
   (N = 2, 4, 6), sa zasebnim, manjim learning rate-om od segmentacione glave.
4. **Statistička validacija** — svaka konfiguracija trenirana je sa tri nezavisna
   random seed-a; poređenja između konfiguracija rađena su uparenim t-testom.
5. **Finalna evaluacija** — jednokratna, na potpuno nezavisnom test skupu.

## Struktura repozitorijuma

```
├── napravi_split.py              # Deli INBreast podatke na train/val/test — GRUPISANO PO PACIJENTU
├── provera_maski_detaljno.py     # Kvantitativna i vizuelna verifikacija segmentacionih maski
├── dataset.py                    # PyTorch Dataset — učitavanje DICOM, normalizacija, augmentacija
├── model.py                      # Segmentacioni model — zamrznut/delimično odmrznut backbone + head
├── losses.py                     # Loss funkcija (BCE + Dice) i metrike (Dice, IoU, Sensitivity, Precision)
├── train.py                      # Trening jedne konfiguracije (train_one_run)
├── run_multi_seed.py             # Orkestracija treninga preko više random seed-ova i konfiguracija
├── evaluate.py                   # Finalna evaluacija na test skupu
├── visualize_predictions.py      # Vizuelizacija predikcija na test primerima
├── test_dataset.py               # Sanity-check za Dataset klasu
├── test_model.py                 # Sanity-check za arhitekturu modela
└── inbreast_split.json           # Sačuvana podela podataka (generiše se pomoću napravi_split.py)
```

## Instalacija

Zahtevani su Python 3.11 i PyTorch sa CUDA podrškom.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install pydicom scipy pillow numpy matplotlib
```

DINOv2 backbone (`dinov2_vits14`) preuzima se automatski pri prvom pokretanju, preko
`torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")`.

Potrebno je lokalno preuzeti **INBreast** bazu podataka i podesiti putanju (`ROOT`) na
vrhu `napravi_split.py` i `provera_maski_detaljno.py`.

## Upotreba

```bash
# 1. Verifikacija ispravnosti gotovih segmentacionih maski (opciono, ali preporučeno)
python provera_maski_detaljno.py

# 2. Podela podataka na train/val/test (generiše inbreast_split.json)
python napravi_split.py

# 3. Brz test jedne konfiguracije (npr. baseline)
python train.py

# 4. Pun eksperimentalni protokol - izmeniti CONFIGS listu u fajlu za željene konfiguracije
python run_multi_seed.py

# 5. Finalna evaluacija pobedničke konfiguracije na test skupu
python evaluate.py

# 6. Vizuelizacija primera predikcija
python visualize_predictions.py
```

## Rezultati

| Konfiguracija | Val Dice | Std |
|---|---|---|
| Baseline (zamrznut backbone) | 0.4995 | 0.0137 |
| + Augmentacija | 0.5169 | 0.0047 |
| + Fine-tuning (N=2 bloka) | 0.5532 | 0.0149 |
| + Fine-tuning (N=4 bloka) | 0.5768 | 0.0115 |
| + Fine-tuning (N=6 blokova) | 0.6709 | 0.0151 |

**Finalna evaluacija na test skupu** (pobednička konfiguracija, prosek 3 seed-a):

| Metrika | Vrednost | Std |
|---|---|---|
| Dice | 0.7813 | 0.0079 |
| IoU | 0.6789 | 0.0109 |
| Sensitivity (Recall) | 0.7546 | 0.0106 |
| Precision | 0.8924 | 0.0115 |

## Metodološke napomene

- **Podela po pacijentu, ne po slici** — sprečava curenje podataka između trening i
  test skupa (svaki pacijent ima u proseku više od jedne snimljene slike).
- **Ponavljanje sa tri random seed-a** — pojedinačna pokretanja iste konfiguracije na
  ovako malom skupu podataka pokazala su varijansu (do ±0.03 Dice) uporedivu sa
  razlikama koje se pripisuju testiranim promenama, pa je poređenje isključivo na
  osnovu proseka i uparenog statističkog testa.
- **Test skup korišćen samo jednom** — finalna evaluacija sprovedena je tek nakon što
  je konfiguracija izabrana isključivo na osnovu validacionog skupa.

## Reference

- Oquab, M. et al. *DINOv2: Learning Robust Visual Features without Supervision.*
  arXiv:2304.07193 (2024). https://doi.org/10.48550/arXiv.2304.07193
- https://github.com/facebookresearch/dinov2
- INBreast baza podataka — Moreira, I. C. et al. *INbreast: toward a full-field
  digital mammographic database.* Academic Radiology (2012).

---

*Autor: Stojan Bastaja*
