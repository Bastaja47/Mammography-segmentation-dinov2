
import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH_SIZE = 14

class DinoSegmentationModel(nn.Module):
    def __init__(self, backbone_name="dinov2_vits14", n_unfrozen_blocks=0):
        super().__init__()

        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
        embed_dim = self.backbone.embed_dim

        self.n_unfrozen_blocks = n_unfrozen_blocks
        self.fully_frozen = (n_unfrozen_blocks == 0)

        for p in self.backbone.parameters():
            p.requires_grad = False

        if n_unfrozen_blocks > 0:
            for block in self.backbone.blocks[-n_unfrozen_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True
            if hasattr(self.backbone, "norm"):
                for p in self.backbone.norm.parameters():
                    p.requires_grad = True

        self.head = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def train(self, mode=True):
        super().train(mode)
        if self.fully_frozen:
            self.backbone.eval()
        return self

    def forward(self, images):
        B, C, H, W = images.shape

        if self.fully_frozen:
            with torch.no_grad():
                features = self.backbone.forward_features(images)
        else:
            features = self.backbone.forward_features(images)

        patch_tokens = features["x_norm_patchtokens"]

        h = H // PATCH_SIZE
        w = W // PATCH_SIZE
        patch_grid = patch_tokens.reshape(B, h, w, -1).permute(0, 3, 1, 2)

        logits_small = self.head(patch_grid)
        logits_full = F.interpolate(
            logits_small, size=(H, W), mode="bilinear", align_corners=False
        )
        return logits_full
