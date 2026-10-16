import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class DeepLabV3Plus(nn.Module):
    """
    DeepLabv3+ model with ResNet-50 backbone.
    Encoder-Decoder with Atrous Spatial Pyramid Pooling (ASPP).
    """
    def __init__(
        self,
        num_classes=7,
        in_channels=3,
        encoder_name="resnet50",
        encoder_weights="imagenet",
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)
