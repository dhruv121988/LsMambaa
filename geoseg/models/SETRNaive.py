import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SETRNaive(nn.Module):
    """
    SETR-Naive: Pure Vision Transformer Semantic Segmentation Baseline.
    Reference:
      Zheng et al., "Rethinking Semantic Segmentation from a Sequence-to-Sequence
      Perspective with Transformers", CVPR 2021.
      Official SETR: https://github.com/fudan-zvg/SETR

    Architecture:
      - Encoder: ViT-B/16 (12 Transformer blocks, 768 hidden dim, 12 attention heads),
        initialized with official ImageNet-1k pretrained weights.
      - Naive Decoder Head: Progressive projection head with 1x1 conv + 3x3 conv + BN + ReLU,
        followed by 16x direct bilinear upsampling to input resolution.
      - Positional Embedding: Bicubic 2D interpolation to dynamically support any input resolution
        (e.g., 512x512 during training crop, 1024x1024 during LoveDA validation).
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        drop_rate: float = 0.1,
        in_channels: int = 3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.patch_size = 16
        self.pretrained_loaded = pretrained

        # Load official ImageNet-1k pretrained ViT-B/16 backbone
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        vit = models.vit_b_16(weights=weights)

        self.conv_proj = vit.conv_proj
        self.class_token = vit.class_token
        self.encoder = vit.encoder

        # SETR-Naive Decoder Head (CVPR 2021)
        self.head = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_rate),
            nn.Conv2d(512, num_classes, kernel_size=1),
        )

        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def interpolate_pos_encoding(self, x, w, h):
        """
        Dynamically interpolate 2D positional embeddings for arbitrary image sizes.
        """
        npatch = x.shape[1] - 1
        N = self.encoder.pos_embedding.shape[1] - 1
        if npatch == N and w == h:
            return self.encoder.pos_embedding

        class_pos_embed = self.encoder.pos_embedding[:, 0]
        patch_pos_embed = self.encoder.pos_embedding[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(N ** 0.5)

        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            size=(h0, w0),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(1), patch_pos_embed), dim=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. Patch projection
        feat = self.conv_proj(x)
        Hp, Wp = feat.shape[2], feat.shape[3]
        feat = feat.flatten(2).transpose(1, 2)  # [B, N, 768]

        # 2. Add class token
        batch_class_token = self.class_token.expand(B, -1, -1)
        feat = torch.cat([batch_class_token, feat], dim=1)

        # 3. Add dynamically interpolated positional embedding
        pos_embed = self.interpolate_pos_encoding(feat, W, H)
        feat = feat + pos_embed
        feat = self.encoder.dropout(feat)

        # 4. Standard Transformer Encoder layers
        feat = self.encoder.layers(feat)
        feat = self.encoder.ln(feat)

        # 5. Extract spatial tokens (discard class token) and reshape to 2D
        spatial_feat = feat[:, 1:].transpose(1, 2).reshape(B, -1, Hp, Wp)

        # 6. SETR-Naive projection head + 16x bilinear upsampling
        out = self.head(spatial_feat)
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=True)
        return out
