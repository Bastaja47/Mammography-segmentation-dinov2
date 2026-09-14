"""
losses.py

Loss funkcija (ono sto MREZA optimizuje) i metrike (ono sto MI citamo).


"""

import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    

    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        # BCEWithLogitsLoss ocekuje SIROVE logite (ne sigmoid) - primenjuje
        # sigmoid interno, numericki stabilnije nego rucni sigmoid()+BCELoss.

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)  # logiti -> verovatnoce [0,1]
        dice_loss = 1.0 - soft_dice_score(probs, targets)
        # Dice SCORE: "vise je bolje" (1 = savrseno poklapanje).
        # LOSS mora biti "manje je bolje" (mreza minimizuje) -> 1 - dice.

        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


def soft_dice_score(probs, targets, eps=1e-6):
    """Dice mora ostati
    diferencijabilno da bi moglo da se koristi u loss-u.

    Umesto strogog preseka (koji zahteva binarne 0/1 vrednosti), koristimo
    mnozenje verovatnoca kao glatku (diferencijabilnu) aproksimaciju
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


@torch.no_grad()
def hard_sensitivity_score(logits, targets, threshold=0.5, eps=1e-6):
    """Sensitivity (Recall) = TP / (TP + FN).

    Klinicko znacenje: od svih piksela koji SU stvarno masa, koliki
    procenat je model pronasao? Moram da proverim 
    Visok sensitivity = model retko
    propusta masu  ovo je obicno vaznije od precision-a,
    jer je propustena masa (lazno negativno) ozbiljnija greska
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.reshape(preds.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)

    tp = (preds * targets).sum(dim=1)
    fn = ((1 - preds) * targets).sum(dim=1)

    sensitivity = (tp + eps) / (tp + fn + eps)
    return sensitivity.mean().item()


@torch.no_grad()
def hard_precision_score(logits, targets, threshold=0.5, eps=1e-6):
    """Precision = TP / (TP + FP).

    Klinicko znacenje: od svega sto je model oznacio kao masu, koliki
    procenat je zaista bio tacan pogodak (ne lazni alarm na zdravom tkivu)
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.reshape(preds.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)

    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1 - targets)).sum(dim=1)

    precision = (tp + eps) / (tp + fp + eps)
    return precision.mean().item()
