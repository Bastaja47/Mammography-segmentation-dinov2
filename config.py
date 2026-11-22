"""
Putanje za ceo projekat na jednom mestu.

U kodu nema apsolutnih putanja sa mog racunara: sve se racuna u odnosu na
folder projekta. Ako je INBreast baza negde drugde, dovoljno je postaviti
promenljivu okruzenja INBREAST_ROOT (folder u kome su AllDICOMs/ i extras/).

    PowerShell:  $env:INBREAST_ROOT = "D:\\podaci\\INBreast"
    Linux/Mac:   export INBREAST_ROOT=/putanja/do/INBreast
"""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# Podrazumevano ocekujemo bazu u <projekat>/INBreast (taj folder je u .gitignore)
INBREAST_ROOT = Path(os.environ.get("INBREAST_ROOT", PROJECT_DIR / "INBreast"))
DICOM_DIR = INBREAST_ROOT / "AllDICOMs"
MASK_DIR = INBREAST_ROOT / "extras" / "MassSegmentationMasks"

# Podela na train/val/test. Putanje slika u njoj su relativne u odnosu na INBREAST_ROOT
SPLIT_JSON = PROJECT_DIR / "inbreast_split.json"

# Izlazi treninga i evaluacije
RESULTS_DIR = PROJECT_DIR / "results"
LOGS_DIR = RESULTS_DIR / "logs"
SUMMARY_DIR = RESULTS_DIR / "summary"
FIGURES_DIR = RESULTS_DIR / "figures"

# Istrenirani modeli (.pt) su veliki (~86 MB) pa nisu u Git-u
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
