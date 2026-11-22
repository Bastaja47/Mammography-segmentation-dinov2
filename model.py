"""
Segmentacioni model: DINOv2 backbone + 1x1 konvoluciona glava (head).

Backbone je ili potpuno zamrznut (n_unfrozen_blocks=0), ili se odmrzava
poslednjih N transformer blokova (fine-tuning). Glava se uvek trenira.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ViT-S/14 deli sliku na patch-eve 14x14 piksela
PATCH_SIZE = 14


class DinoSegmentationModel(nn.Module):
    def __init__(self, backbone_name="dinov2_vits14", n_unfrozen_blocks=0):
        super().__init__()

        # Pretrenirani DINOv2 se preuzima preko torch.hub pri prvom pokretanju
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
        embed_dim = self.backbone.embed_dim

        self.n_unfrozen_blocks = n_unfrozen_blocks
        self.fully_frozen = (n_unfrozen_blocks == 0)

        # 1) Zamrzni sve parametre backbone-a
        for p in self.backbone.parameters():
            p.requires_grad = False

        # 2) Odmrzni samo poslednjih N blokova (i zavrsni LayerNorm).
        #    Rani slojevi uce opste obrasce (ivice, teksture) pa ih ne diramo,
        #    a poslednji su specificniji za domen i njih prilagodjavamo.
        if n_unfrozen_blocks > 0:
            for block in self.backbone.blocks[-n_unfrozen_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True
            if hasattr(self.backbone, "norm"):
                for p in self.backbone.norm.parameters():
                    p.requires_grad = True

        # Glava: 1x1 konvolucija koja svaki patch vektor (384 broja) pretvara u jedan logit
        self.head = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def train(self, mode=True):
        super().train(mode)
        # Potpuno zamrznut backbone drzimo u eval modu i tokom treninga
        if self.fully_frozen:
            self.backbone.eval()
        return self

    def forward(self, images):
        B, C, H, W = images.shape

        # Kad je backbone zamrznut, nema potrebe da se cuva graf za gradijente
        if self.fully_frozen:
            with torch.no_grad():
                features = self.backbone.forward_features(images)
        else:
            features = self.backbone.forward_features(images)

        # (B, 1369, 384) -> (B, 384, 37, 37): patch tokeni slozeni nazad u 2D mrezu
        patch_tokens = features["x_norm_patchtokens"]
        h = H // PATCH_SIZE
        w = W // PATCH_SIZE
        patch_grid = patch_tokens.reshape(B, h, w, -1).permute(0, 3, 1, 2)

        # Predikcija po patch-u, pa bilinearno povecanje na punu rezoluciju slike
        logits_small = self.head(patch_grid)
        logits_full = F.interpolate(
            logits_small, size=(H, W), mode="bilinear", align_corners=False
        )
        return logits_full
