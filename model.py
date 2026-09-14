"""
model.py

Segmentacioni model - prosiren da podrzi delimican fine-tuning: umesto
"sve ili nista", moze da se odmrzne poslednjih N transformer blokova
backbone-a, dok ostatak (patch embedding, rani blokovi) ostaje zamrznut.


"""

import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH_SIZE = 14  # sve dinov2_*14 varijante koriste patch 14x14


class DinoSegmentationModel(nn.Module):
    def __init__(self, backbone_name="dinov2_vits14", n_unfrozen_blocks=0):
        """
        n_unfrozen_blocks=0  -> puno zamrznut backbone (isto kao baseline/
                                  augmentacija eksperimenti do sad)
        n_unfrozen_blocks=N  -> poslednjih N transformer blokova (+ finalni
                                  norm sloj) je TRENABILNO, ostatak zamrznut
        """
        super().__init__()

        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
        embed_dim = self.backbone.embed_dim

        self.n_unfrozen_blocks = n_unfrozen_blocks
        self.fully_frozen = (n_unfrozen_blocks == 0)

        # KORAK 1: zamrzni SVE (isto kao ranije)
        for p in self.backbone.parameters():
            p.requires_grad = False          #Tad pytorch nece racunati gradijente, backbone potpuno zamrznut

        # KORAK 2: ako trazimo delimican fine-tuning, odmrzni SAMO
        # poslednjih n_unfrozen_blocks transformer blokova + finalni norm.
        if n_unfrozen_blocks > 0:
            for block in self.backbone.blocks[-n_unfrozen_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True
            # finalni normalizacioni sloj (poslednja transformacija pre
            # nego sto tokeni izadju iz backbone-a) - i njega odmrzavamo,
            # ako postoji pod ovim imenom u ucitanom modelu.
            if hasattr(self.backbone, "norm"):            #Proveravamo ima li zavrsni normalizacioni sloj jer i njega treba odmrznuti
                for p in self.backbone.norm.parameters():
                    p.requires_grad = True

        self.head = nn.Conv2d(embed_dim, 1, kernel_size=1)  #Pretvara izlaz iz backbone u masku za segmentaciju (broj ulaznih kanala, broj izlaznih kanala, velicina filtera)

    def train(self, mode=True):
        """Kad je backbone potpuno zamrznut, prisilno ga drzimo u eval()
        modu (kao i ranije - bezbednosna mera). Kad je delimicnoa odmrznut,
        puštamo standardno train()/eval() ponasanje na celom modelu"""
        super().train(mode)
        if self.fully_frozen:
            self.backbone.eval()
        return self

    def forward(self, images):
        B, C, H, W = images.shape

        if self.fully_frozen:
            # Ovde JOS UVEK koristimo no_grad() - dodatna optimizacija
            # memorije kad znamo da BAS NISTA u backbone-u ne treba gradijent.
            with torch.no_grad():
                features = self.backbone.forward_features(images)
        else:
            # KLJUCNA IZMENA: BEZ no_grad() ovde! Ako bismo ga zadrzali,
            # PyTorch fizicki ne bi mogao da izracuna gradijente ni za
            # odmrznute poslednje blokove - no_grad() "presece" mogucnost
            # gradijenta za SVE unutar sebe, bez izuzetka.
            #
            # PyTorch je ipak pametan sam po sebi: pošto ulazna slika
            # requires_grad=False, i prvi (zamrznuti) blokovi takodje
            # requires_grad=False, PyTorch automatski NE gradi graf za
            # gradijente kroz te rane slojeve - tek kad racunanje stigne
            # do PRVOG odmrznutog bloka, graf pocinje da se prati. Dakle,
            # i dalje dobijamo (skoro) istu usteda memorije, bez rucnog
            # no_grad() koji bi ovde bio STETAN, ne koristan.
            features = self.backbone.forward_features(images)

        patch_tokens = features["x_norm_patchtokens"]  # (B, N, embed_dim)

        h = H // PATCH_SIZE
        w = W // PATCH_SIZE
        patch_grid = patch_tokens.reshape(B, h, w, -1).permute(0, 3, 1, 2)

        logits_small = self.head(patch_grid)
        logits_full = F.interpolate(
            logits_small, size=(H, W), mode="bilinear", align_corners=False
        )
        return logits_full
# 1. Izvuci obeležja slike kroz backbone (uz no_grad() štedimo memoriju ako je skroz zamrznut).
# 2. Izdvoji patch tokene (kockice 14x14) i preuredi ih iz niza u 2D grid [B, Embed_dim, h, w].
# 3. Konvolucioni head pretvara taj grid u smanjenu masku (logits_small).
# 4. Interpolacijom (bilinear) razvuci malu masku nazad na originalnu rezoluciju slike (H, W).