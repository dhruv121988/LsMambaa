import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class SegFormer(nn.Module):
    """
    SegFormer model with Hierarchical Mix Transformer (MiT-B0) backbone.
    Lightweight all-MLP decoder for efficient semantic segmentation.
    Paper: Xie et al., SegFormer: Simple and Efficient Design for Semantic
           Segmentation with Transformers, NeurIPS 2021.
    """
    def __init__(
        self,
        num_classes=7,
        in_channels=3,
        encoder_name="mit_b0",
        encoder_weights="imagenet",
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.model = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)
