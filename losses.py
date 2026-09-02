"""
losses.py

Loss funkcija (ono sto MREZA optimizuje) i metrike (ono sto MI citamo).

=======================================================================
STA MORAS DA RAZUMES IZ OVOG FAJLA (koncept - ovo je "tvoja teorija"):
- Razlika izmedju LOSS-a (mora biti diferencijabilan, mreza ga koristi
  da racuna gradijente) i METRIKE (ne mora biti diferencijabilna, MI je
  citamo da procenimo kvalitet)
- Zasto kombinujemo BCE + Dice, umesto da koristimo samo jedno
- Sta znaci "soft" (za loss) nasuprot "hard" (za metriku) Dice

STA MOZES SLOBODNO DA GUGLAS (sintaksa - ovo niko ne pamti napamet):
- Tacni argumenti nn.BCEWithLogitsLoss()
- Kako tacno radi torch.sigmoid, .reshape(), .sum(dim=...)
=======================================================================
"""

import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    """Kombinacija Binary Cross-Entropy i (mekog) Dice loss-a.

    ZASTO KOMBINACIJA (ovo MORAS da umes da objasnis):
    - BCE gleda SVAKI piksel nezavisno i daje stabilan, koristan gradijent
      cak i na samom pocetku treninga, kad model jos nista ne pogadja
      (kad je preklapanje sa pravom maskom = 0, Dice gradijent postaje
      manje informativan bas u tom trenutku).
    - Dice loss direktno optimizuje METRIKU koja nas na kraju zanima
      (preklapanje) i "zna" da je masa mala/retka regija - BCE to ne zna,
      pa BCE sam moze biti "zadovoljan" i kad model uglavnom predvidja
      pozadinu (neuravnotezenost klasa, o cemu smo pricali ranije).

    Ova kombinacija je standardna, proverena praksa u medicinskoj
    segmentaciji (npr. podrazumevani izbor u nnU-Net okviru).
    """

    def __init__(self, bce_weight=0.5):  #bce 0.5 znaci da je BCE i Dice jednako vazan, moze se podesiti(da je 1 bilo bi samo bce da je 0 samo dice)
        super().__init__() #obavezan korak kad se koristi nn.Module, poziva se konstruktor roditelja
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss() #Obezbedjuje da se vracaju logiti(sirovi brojevi)
        # BCEWithLogitsLoss ocekuje SIROVE logite (ne sigmoid) - primenjuje
        # sigmoid interno, numericki stabilnije nego rucni sigmoid()+BCELoss.

    def forward(self, logits, targets):  #klasa koja nasledjuje nn.Module mora da implementira forward metodu, koja definise kako se ulazni podaci prolaze kroz mrezu i kako se dobija izlaz
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)  # logiti -> verovatnoce [0,1]
        dice_loss = 1.0 - soft_dice_score(probs, targets)
        # Dice SCORE: "vise je bolje" (1 = savrseno poklapanje).
        # LOSS mora biti "manje je bolje" (mreza minimizuje) -> 1 - dice.
        #logits su sirovi brojevi izlaz modela, a targets su binarne maske (0 ili 1)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


def soft_dice_score(probs, targets, eps=1e-6):
    """Dice racunat sa VEROVATNOCAMA (soft/meko) - MORA ostati
    diferencijabilno da bi moglo da se koristi u loss-u.

    Umesto strogog preseka (koji zahteva binarne 0/1 vrednosti), koristimo
    mnozenje verovatnoca kao glatku (diferencijabilnu) aproksimaciju.
    """
    probs = probs.reshape(probs.shape[0], -1)     # (B, H*W) - spljosti svaku sliku
    targets = targets.reshape(targets.shape[0], -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)
    # eps sprecava deljenje nulom ako je neka slika u batch-u bez
    # ijednog pozitivnog piksela (union bi bio 0)
    return dice.mean()


@torch.no_grad()
def hard_dice_score(logits, targets, threshold=0.5, eps=1e-6):
    """PRAVI Dice, binarne (thresholded) predikcije - metrika koju
    IZVESTAVAMO (u tabelama/izvestaju), ne koristi se za trening."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.reshape(preds.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()  # .item(): tenzor sa 1 vrednoscu -> obican Python broj


@torch.no_grad()
def hard_iou_score(logits, targets, threshold=0.5, eps=1e-6):
    """IoU (Jaccard): IoU = |A∩B| / |A∪B|, gde je |A∪B| = |A|+|B|-|A∩B|"""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.reshape(preds.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()
