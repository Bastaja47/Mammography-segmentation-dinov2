"""
Loss funkcija (BCE + Dice) i metrike za segmentaciju (Dice, IoU, Sensitivity, Precision)

Sve metrike se racunaju po slici, pa se usrednjavaju preko batch-a
Metrike sa prefiksom "hard_" koriste prag 0.5 na verovatnoci (binarna predikcija),
a "soft_dice_score" radi direktno nad verovatnocama pa se moze koristiti u lossu
"""

import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    """
    Kombinacija BCE (greska po pikselu) i Dice lossa (preklapanje maski)



    Mase zauzimaju mali deo slike, pa sam BCE bi favorizovao kao da je sve pozadina
    Dice kaznjava slabo preklapanje, pa zajedno daju stabilniji trening
    """

    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        dice_loss = 1.0 - soft_dice_score(probs, targets)

        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


def soft_dice_score(probs, targets, eps=1e-6):
    """Dice nad verovatnocama (diferencijabilan), srednja vrednost po slikama u batchu"""
    probs = probs.reshape(probs.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()


@torch.no_grad()
def hard_dice_score(logits, targets, threshold=0.5, eps=1e-6):
    """Dice nad binarnom predikcijom (verovatnoca > prag)"""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.reshape(preds.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


@torch.no_grad()
def hard_iou_score(logits, targets, threshold=0.5, eps=1e-6):
    """IoU (Jaccard): preklapanje /unija predikcije i prave maske"""
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
    """Sensitivity (recall): koji deo prave mase je model pokrio"""
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
    """Precision: koji deo predvidjene mase stvarno pripada pravoj masi"""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.reshape(preds.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)

    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1 - targets)).sum(dim=1)

    precision = (tp + eps) / (tp + fp + eps)
    return precision.mean().item()
