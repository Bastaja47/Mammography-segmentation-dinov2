"""
model.py

Segmentacioni model za Fazu 1 (baseline): zamrznut DINOv2 backbone
+ mali trenabilni head (1x1 konvolucija). Backbone se NE trenira -
samo izvlaci patch-tokene, isto sto smo radili pri istrazivanju repoa.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Sve dinov2_*14 varijante (vits14, vitb14, vitl14, vitg14) koriste patch
# velicine 14x14. Ovo NE citamo iz atributa ucitanog modela (naziv atributa
# se moze razlikovati izmedju verzija repoa), nego eksplicitno hardkodiramo -
# ionako smo ovo vec potvrdili u Fazi 0 (518/14 = 37 patch-eva po strani).
PATCH_SIZE = 14


class DinoSegmentationModel(nn.Module):
    def __init__(self, backbone_name="dinov2_vits14", freeze_backbone=True):
        super().__init__()

        # Ucitavamo GOTOV pretreniran backbone (isto kao ranije preko
        # torch.hub) - ovo je "diplomirani" model iz price o teacher/student,
        # mi ga ovde NE treniramo iznova.
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)

        # Broj kanala u izlazu zavisi od velicine backbone-a (384 za ViT-S,
        # 768 za ViT-B...). Citamo ga IZ ucitanog modela (embed_dim atribut
        # sigurno postoji, koristi se svuda u repo kodu), umesto da ga rucno
        # upisujemo - ako kasnije promenimo backbone_name, kod i dalje radi
        # bez izmena.
        embed_dim = self.backbone.embed_dim

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # FAZA 1 HEAD - najprostije moguce: 1x1 konvolucija koja svaki
        # patch-vektor (embed_dim brojeva) svede na JEDAN broj (logit:
        # "koliko ovaj deo slike lici na masu"). Matematicki isto sto i
        # linearni sloj primenjen nezavisno na svaki patch.
        self.head = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def train(self, mode=True):
        """Override: zamrznut backbone MORA ostati u eval() modu UVEK,
        cak i kad pozovemo model.train() da treniramo head. Bez ovoga bi se
        backbone (iako mu tezine ne rastu) i dalje prebacio u 'training'
        stanje, sto moze uticati na slojeve osetljive na train/eval mod."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self
        #Ovako postizemo da je backbone uvek u eval() modu, a head u train() modu kad pozovemo model.train()


    def forward(self, images):
        """
        images: (B, 3, H, W) - vec preprocesirano u Dataset klasi
        vraca:  (B, 1, H, W) - SIROVI logiti (bez sigmoida), iste rezolucije
                 kao ulaz - spremno da se poredi sa pravom maskom.
        """
        B, C, H, W = images.shape

        if self.freeze_backbone:
            # torch.no_grad() dodatno stedi memoriju - ne cuvamo gradijente
            # za deo mreze koji svakako necemo azurirati. Ovo je RAZLICITO
            # od requires_grad=False: requires_grad kaze "ne racunaj gradijent
            # ZA OVE parametre", a no_grad() kaze "ne cuvaj intermedijarne
            # aktivacije potrebne ZA bilo kakav gradijent" - kombinacija
            # obe stedi znacajno GPU memorije, sto nam je bitno na 6GB kartici.
            with torch.no_grad():
                features = self.backbone.forward_features(images)
        else:
            features = self.backbone.forward_features(images)

        patch_tokens = features["x_norm_patchtokens"]  # (B, N, embed_dim)

        # Patch tokeni su spljosten niz, poredjani red-po-red (isti redosled
        # kao kad citas tekst: levo->desno, pa red ispod). Reshape ih vraca
        # u 2D "tablu": (B, N, C) -> (B, h, w, C). Zatim permute prebacuje
        # kanale na drugo mesto: (B, h, w, C) -> (B, C, h, w) - jer Conv2d
        # ocekuje format (batch, kanali, visina, sirina).
        h = H // PATCH_SIZE
        w = W // PATCH_SIZE
        patch_grid = patch_tokens.reshape(B, h, w, -1).permute(0, 3, 1, 2)

        logits_small = self.head(patch_grid)  # (B, 1, h, w) - gruba rezolucija (npr. 37x37)

        # Upsample NAZAD na punu ulaznu rezoluciju. Bilinear (ne nearest!) je
        # ispravan izbor ovde jer su ovo KONTINUALNI logiti (verovatnoce),
        # ne binarna maska - za binarnu masku smo koristili nearest (setimo
        # se dataset.py), ali ovde radimo obrnutu operaciju nad drugacijim
        # tipom podatka, pa je bilinear ispravan.
        logits_full = F.interpolate(
            logits_small, size=(H, W), mode="bilinear", align_corners=False
        )

        return logits_full
