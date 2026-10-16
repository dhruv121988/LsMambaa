import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """
    Make a 2D bilinear kernel suitable for transposed convolution upsampling.
    Reference: Long, Shelhamer, Darrell (CVPR 2015).
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float32)
    for i in range(min(in_channels, out_channels)):
        weight[i, i, :, :] = filt
    return torch.from_numpy(weight).float()


class FCN8s(nn.Module):
    """
    Pure CNN baseline: FCN-8s with VGG-16 backbone.
    Reference: Long, Shelhamer, & Darrell, "Fully Convolutional Networks for Semantic Segmentation", CVPR 2015.
    As benchmarked in the official LoveDA paper (Wang et al., 2021).

    Combines features across multiple scales via skip connections:
      - pool3 (stride 8, 256 channels)
      - pool4 (stride 16, 512 channels)
      - pool5 / fc7 (stride 32, 4096 channels)
    """

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 3,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained_loaded = pretrained

        # Load VGG-16 backbone
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        vgg = models.vgg16(weights=weights)
        features = list(vgg.features.children())

        # Stage 1: input -> pool3 (stride 8, 256 channels)
        # layers 0..16: conv1_1 to pool3
        self.stage1 = nn.Sequential(*features[:17])

        # Stage 2: pool3 -> pool4 (stride 16, 512 channels)
        # layers 17..23: conv4_1 to pool4
        self.stage2 = nn.Sequential(*features[17:24])

        # Stage 3: pool4 -> pool5 (stride 32, 512 channels)
        # layers 24..30: conv5_1 to pool5
        self.stage3 = nn.Sequential(*features[24:31])

        # Fully convolutional classification layers (fc6, fc7)
        self.fc6 = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
        )
        self.fc7 = nn.Sequential(
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
        )

        # 1x1 convolutions for class scoring at each skip level
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)

        # Transposed convolution upsampling layers
        self.upscore2 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.upscore_pool4 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.upscore8 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=16, stride=8, padding=4, bias=False
        )

        # Initialize weights
        self._init_weights(vgg if pretrained else None)

    def _init_weights(self, vgg=None):
        if vgg is not None:
            # Transfer fc6 and fc7 weights from pretrained VGG16 classifier
            with torch.no_grad():
                self.fc6[0].weight.data.copy_(vgg.classifier[0].weight.data.view(4096, 512, 7, 7))
                self.fc6[0].bias.data.copy_(vgg.classifier[0].bias.data)
                self.fc7[0].weight.data.copy_(vgg.classifier[3].weight.data.view(4096, 4096, 1, 1))
                self.fc7[0].bias.data.copy_(vgg.classifier[3].bias.data)

        # Initialize score layers with Xavier / normal
        for m in [self.score_fr, self.score_pool4, self.score_pool3]:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        # Initialize upsampling layers with bilinear weights
        self.upscore2.weight.data.copy_(
            get_upsampling_weight(self.num_classes, self.num_classes, 4)
        )
        self.upscore_pool4.weight.data.copy_(
            get_upsampling_weight(self.num_classes, self.num_classes, 4)
        )
        self.upscore8.weight.data.copy_(
            get_upsampling_weight(self.num_classes, self.num_classes, 16)
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]

        # Backbone features
        pool3 = self.stage1(x)       # Stride 8  (H/8, W/8)
        pool4 = self.stage2(pool3)   # Stride 16 (H/16, W/16)
        pool5 = self.stage3(pool4)   # Stride 32 (H/32, W/32)

        # Deepest features + fc
        fc6 = self.fc6(pool5)
        fc7 = self.fc7(fc6)
        score_fr = self.score_fr(fc7)  # (H/32, W/32)

        # FCN-16s fusion: upsample score_fr x2 and add pool4 score
        upscore2 = self.upscore2(score_fr)
        score_pool4 = self.score_pool4(pool4)
        if upscore2.shape[2:] != score_pool4.shape[2:]:
            upscore2 = F.interpolate(upscore2, size=score_pool4.shape[2:], mode="bilinear", align_corners=True)
        fuse_pool4 = upscore2 + score_pool4

        # FCN-8s fusion: upsample fuse_pool4 x2 and add pool3 score
        upscore_pool4 = self.upscore_pool4(fuse_pool4)
        score_pool3 = self.score_pool3(pool3)
        if upscore_pool4.shape[2:] != score_pool3.shape[2:]:
            upscore_pool4 = F.interpolate(upscore_pool4, size=score_pool3.shape[2:], mode="bilinear", align_corners=True)
        fuse_pool3 = upscore_pool4 + score_pool3

        # Final upsampling x8 to original image resolution
        out = self.upscore8(fuse_pool3)
        if out.shape[2:] != (h, w):
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)

        return out
