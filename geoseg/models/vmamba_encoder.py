"""
VMamba-Tiny Encoder for GeoSeg
==============================

Wraps the official VMamba backbone (MzeroMiko/VMamba) as a multi-scale
feature-extraction encoder.  Uses the native 2D-Selective-Scan (SS2D /
cross-scan) – NOT mamba_ssm's 1-D Mamba block.

Outputs
-------
Given a 512×512×3 input the encoder returns four skip features:

    res1 : (B, 96,  128, 128)   – 1/4  scale
    res2 : (B, 192,  64,  64)   – 1/8  scale
    res3 : (B, 384,  32,  32)   – 1/16 scale
    res4 : (B, 768,  16,  16)   – 1/32 scale

Usage
-----
    from geoseg.models.vmamba_encoder import VMambaEncoder

    encoder = VMambaEncoder(pretrained_ckpt="/path/to/vssm_tiny_0230_ckpt_epoch_262.pth")
    features = encoder(x)          # x: (B, 3, 512, 512)
    res1, res2, res3, res4 = features

    # Layerwise LR (GeoSeg convention) – the encoder stores everything under
    # self.backbone so that the regex ``backbone.*`` in process_model_params
    # automatically assigns a separate (lower) learning rate.
"""

import os
import math
import copy
import logging
from functools import partial
from typing import Optional, List, Dict, Any
from collections import OrderedDict

import torch
import torch.hub
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.models.layers import DropPath, trunc_normal_

logger = logging.getLogger(__name__)


# =====================================================================
# Utilities from MzeroMiko/VMamba  (self-contained, no mamba_ssm import)
# =====================================================================

class Linear2d(nn.Linear):
    """Linear layer that operates on channel-first (B, C, H, W) tensors."""
    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm that works on channel-first (B, C, H, W) tensors."""
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x


# =====================================================================
# Cross-Scan / Cross-Merge  (pure-PyTorch fallback – no Triton needed)
# =====================================================================

def cross_scan_forward(x: torch.Tensor):
    """
    Unfold a (B, C, H, W) feature map into 4 scan directions → (B, 4, C, L).

    Directions:
        0 – row-major (H,W)
        1 – column-major (W,H)
        2 – reverse row-major
        3 – reverse column-major
    """
    B, C, H, W = x.shape
    L = H * W
    # row-major & column-major views
    x_hw = x.reshape(B, C, L)                                    # (B, C, L)
    x_wh = x.transpose(2, 3).contiguous().reshape(B, C, L)       # (B, C, L)
    xs = torch.stack([x_hw, x_wh, x_hw.flip(-1), x_wh.flip(-1)], dim=1)  # (B, 4, C, L)
    return xs


def cross_merge_forward(ys: torch.Tensor, H: int, W: int):
    """
    Merge 4 scan-direction outputs (B, 4, C, L) back into (B, C, H, W).
    """
    B, K, C, L = ys.shape
    y_hw  = ys[:, 0]
    y_wh  = ys[:, 1]
    y_hwr = ys[:, 2].flip(-1)
    y_whr = ys[:, 3].flip(-1)

    # column-major → row-major
    y_wh  = y_wh.reshape(B, C, W, H).transpose(2, 3).reshape(B, C, L)
    y_whr = y_whr.reshape(B, C, W, H).transpose(2, 3).reshape(B, C, L)

    y = y_hw + y_wh + y_hwr + y_whr                              # (B, C, L)
    return y.reshape(B, C, H, W)


# =====================================================================
# Selective-Scan  (pure-PyTorch recurrence – works on CPU & any GPU)
# =====================================================================

def selective_scan_ref(u, delta, A, B, C, D=None, delta_bias=None,
                       delta_softplus=False, chunk_size=1024):
    """
    Pure-PyTorch chunked parallel selective scan via cumulative sum.
    Chunks the sequence length L to strictly bound peak activation memory.

    Args:
        u:     (B, D, L)    input
        delta: (B, D, L)    Δ
        A:     (D, N)       state matrix (log-space negated externally)
        B:     (B, N, L)    or (B, G, N, L)
        C:     (B, N, L)    or (B, G, N, L)
        D:     (D,)         skip connection
        delta_bias: (D,)    added to delta before softplus
        chunk_size: int     sequence chunk size for bounded memory (default 512)

    Returns:
        y:     (B, D, L)
    """
    B_batch, D_dim, L = u.shape
    N = A.shape[1]

    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(0).unsqueeze(-1)  # (B, D, L)
    if delta_softplus:
        delta = F.softplus(delta)

    G = B.shape[1] if B.dim() == 4 else 1
    d_per_g = D_dim // G
    A_exp = A.unsqueeze(0).unsqueeze(2)  # (1, D, 1, N)

    num_chunks = (L + chunk_size - 1) // chunk_size
    ys = []
    x_prev = torch.zeros(B_batch, D_dim, N, device=u.device, dtype=u.dtype)

    for c in range(num_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, L)
        Lc = end - start

        u_c = u[:, :, start:end]
        delta_c = delta[:, :, start:end]

        if B.dim() == 4:
            b_c = B[:, :, :, start:end].unsqueeze(2).expand(B_batch, G, d_per_g, N, Lc).reshape(B_batch, D_dim, N, Lc).permute(0, 1, 3, 2)
            c_c = C[:, :, :, start:end].unsqueeze(2).expand(B_batch, G, d_per_g, N, Lc).reshape(B_batch, D_dim, N, Lc).permute(0, 1, 3, 2)
        else:
            b_c = B[:, :, start:end].unsqueeze(2).expand(B_batch, D_dim, 1, Lc).permute(0, 1, 3, 2)
            c_c = C[:, :, start:end].unsqueeze(2).expand(B_batch, D_dim, 1, Lc).permute(0, 1, 3, 2)

        delta_A_c = delta_c.unsqueeze(-1) * A_exp  # (B, D, Lc, N)
        log_a_cum = torch.cumsum(delta_A_c, dim=2)
        log_a_cum = torch.clamp(log_a_cum, min=-40.0, max=0.0)
        a_cum = torch.exp(log_a_cum)

        # Fused memory-efficient formulation: avoids large intermediate tensor allocations
        du_c = (delta_c * u_c).unsqueeze(-1)  # (B, D, Lc, 1)
        v_scaled = (b_c * torch.exp(-log_a_cum)) * du_c  # (B, D, Lc, N)
        x_intra = a_cum * (torch.cumsum(v_scaled, dim=2) + x_prev.unsqueeze(2))

        y_c = (x_intra * c_c).sum(dim=-1)
        ys.append(y_c)
        x_prev = x_intra[:, :, -1]

    y = torch.cat(ys, dim=-1)
    if D is not None and isinstance(D, torch.Tensor):
        y = y + u * D.unsqueeze(0).unsqueeze(-1)


try:
    from geoseg.models.csm_triton import selective_scan_fn
except ImportError:
    selective_scan_fn = selective_scan_ref


# =====================================================================
# SS2D – 2D Selective Scan (cross-scan based)
# =====================================================================

class SS2D(nn.Module):
    """
    2D Selective-Scan block with cross-scan / cross-merge.

    This is a self-contained re-implementation of MzeroMiko/VMamba's SS2D
    that uses pure-PyTorch cross-scan and selective-scan (no mamba_ssm,
    no Triton kernels required).
    """

    def __init__(
        self,
        d_model: int = 96,
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        dt_rank: str = "auto",
        act_layer: type = nn.SiLU,
        d_conv: int = 3,
        conv_bias: bool = True,
        dropout: float = 0.0,
        bias: bool = False,
        forward_type: str = "v0",
        channel_first: bool = False,
        **kwargs,
    ):
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank_val = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank_val
        self.channel_first = channel_first
        k_group = 4  # 4 scan directions

        # ---- input projection ----
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)
        self.act = act_layer()
        self.conv2d = nn.Conv2d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=(d_conv - 1) // 2, groups=d_inner, bias=conv_bias,
        )

        # ---- x-projection (shared across 4 scan dirs) ----
        self.x_proj_weight = nn.Parameter(
            torch.randn(k_group, dt_rank_val + d_state * 2, d_inner) * 0.02
        )

        # ---- dt / A / D initialisation ----
        self.dt_projs_weight, self.dt_projs_bias = self._init_dt(
            dt_rank_val, d_inner, k_group)
        self.A_logs = self._init_A(d_state, d_inner, k_group)
        self.Ds = self._init_D(d_inner, k_group)

        # ---- output projection ----
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    # ---- static initialisers (from MzeroMiko/VMamba mamba_init) ----

    @staticmethod
    def _init_dt(dt_rank, d_inner, k_group,
                 dt_scale=1.0, dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        weights, biases = [], []
        for _ in range(k_group):
            w = torch.empty(d_inner, dt_rank)
            std = dt_rank ** -0.5 * dt_scale
            nn.init.uniform_(w, -std, std)
            dt = torch.exp(
                torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            weights.append(w)
            biases.append(inv_dt)
        return (
            nn.Parameter(torch.stack(weights, dim=0)),   # (K, d_inner, dt_rank)
            nn.Parameter(torch.stack(biases, dim=0)),     # (K, d_inner)
        )

    @staticmethod
    def _init_A(d_state, d_inner, k_group):
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).repeat(d_inner, 1)  # (D, N)
        A_log = torch.log(A)
        A_log = A_log.unsqueeze(0).repeat(k_group, 1, 1).flatten(0, 1)  # (K*D, N)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def _init_D(d_inner, k_group):
        D = torch.ones(d_inner).unsqueeze(0).repeat(k_group, 1).flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, H, W, C)   [channel-last]  or  (B, C, H, W) [channel-first]
        Returns:
            (B, H, W, C)  or  (B, C, H, W) matching input layout
        """
        if self.channel_first:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)  # → (B, H, W, C)
        else:
            B, H, W, C = x.shape

        xz = self.in_proj(x)                     # (B, H, W, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)         # each (B, H, W, d_inner)
        z = self.act(z)

        # depthwise conv
        x_inner = x_inner.permute(0, 3, 1, 2).contiguous()  # (B, d_inner, H, W)
        x_inner = self.act(self.conv2d(x_inner))

        # cross-scan → (B, 4, d_inner, L)
        xs = cross_scan_forward(x_inner)
        K = 4
        D_dim = self.d_inner
        N = self.d_state
        R = self.dt_rank
        L = H * W

        # x-projection for dt, B, C
        x_dbl = torch.einsum("b k d l, k c d -> b k c l",
                             xs, self.x_proj_weight)        # (B, K, R+2N, L)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l",
                           dts, self.dt_projs_weight)       # (B, K, D, L)

        # flatten K into D dimension for batched selective-scan
        xs_flat = xs.reshape(B, K * D_dim, L)
        dts_flat = dts.reshape(B, K * D_dim, L)

        dtype = xs_flat.dtype
        As = -self.A_logs.to(dtype).exp()                    # (K*D, N)
        Ds = self.Ds.to(dtype)                               # (K*D,)
        dt_bias = self.dt_projs_bias.to(dtype).reshape(-1)   # (K*D,)

        # selective scan (Triton accelerated)
        ys = selective_scan_fn(
            xs_flat, dts_flat,
            As, Bs.to(dtype).reshape(B, K, N, L),
            Cs.to(dtype).reshape(B, K, N, L),
            D=None, delta_bias=dt_bias, delta_softplus=True,
        )
        ys = ys + xs_flat * Ds.unsqueeze(0).unsqueeze(-1)

        # reshape back to (B, K, D, L) and cross-merge
        ys = ys.reshape(B, K, D_dim, L)
        y = cross_merge_forward(ys, H, W)                   # (B, d_inner, H, W)

        y = y.permute(0, 2, 3, 1).to(x.dtype)               # (B, H, W, d_inner)
        y = self.out_norm(y)
        y = y * z
        y = self.out_proj(y)
        y = self.dropout(y)

        if self.channel_first:
            y = y.permute(0, 3, 1, 2)

        return y                                             # (B, H, W, C)


# =====================================================================
# VSSBlock – Vision State-Space block
# =====================================================================

class VSSBlock(nn.Module):
    """
    Single VMamba block: LN → SS2D → residual,  LN → MLP → residual.
    """

    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0.0,
        norm_layer: type = nn.LayerNorm,
        channel_first: bool = False,
        ssm_d_state: int = 16,
        ssm_ratio: float = 2.0,
        ssm_dt_rank: str = "auto",
        ssm_act_layer: type = nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias: bool = True,
        ssm_drop_rate: float = 0.0,
        ssm_init: str = "v0",
        forward_type: str = "v0",
        mlp_ratio: float = 4.0,
        mlp_act_layer: type = nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp: bool = False,
        use_checkpoint: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.norm = norm_layer(hidden_dim)
        self.op = SS2D(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            forward_type=forward_type,
            channel_first=channel_first,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # MLP (optional – VMamba-Tiny vanilla uses mlp_ratio=0)
        self.has_mlp = mlp_ratio > 0.0
        if self.has_mlp:
            _Mlp = gMlp if gmlp else Mlp
            self.norm2 = norm_layer(hidden_dim)
            self.mlp = _Mlp(
                in_features=hidden_dim,
                hidden_features=int(hidden_dim * mlp_ratio),
                act_layer=mlp_act_layer,
                drop=mlp_drop_rate,
                channels_first=channel_first,
            )

    def _forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.op(self.norm(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, x: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


# =====================================================================
# VSSM – full VMamba backbone  (classification head removed)
# =====================================================================

class VSSM(nn.Module):
    """
    VMamba backbone (Backbone_VSSM from MzeroMiko/VMamba).

    Constructs 4 stages with increasing channel dimensions and decreasing
    spatial resolution, suitable for use as an encoder in UNet-style
    segmentation architectures.
    """

    def __init__(
        self,
        patch_size: int = 4,
        in_chans: int = 3,
        depths: list = (2, 2, 9, 2),
        dims: int = 96,
        # SSM params
        ssm_d_state: int = 16,
        ssm_ratio: float = 2.0,
        ssm_dt_rank: str = "auto",
        ssm_act_layer: str = "silu",
        ssm_conv: int = 3,
        ssm_conv_bias: bool = True,
        ssm_drop_rate: float = 0.0,
        ssm_init: str = "v0",
        forward_type: str = "v0",
        # MLP params
        mlp_ratio: float = 0.0,
        mlp_act_layer: str = "gelu",
        mlp_drop_rate: float = 0.0,
        gmlp: bool = False,
        # regularisation
        drop_path_rate: float = 0.2,
        patch_norm: bool = True,
        norm_layer: str = "ln",
        downsample_version: str = "v1",
        patchembed_version: str = "v1",
        use_checkpoint: bool = False,
        # feature extraction
        out_indices: tuple = (0, 1, 2, 3),
        pretrained: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        self.channel_first = norm_layer.lower() in ("bn", "ln2d")
        self.num_layers = len(depths)

        if isinstance(dims, int):
            dims = [int(dims * 2 ** i) for i in range(self.num_layers)]
        self.dims = dims
        self.out_indices = out_indices

        _NORMLAYERS = {"ln": nn.LayerNorm, "ln2d": LayerNorm2d, "bn": nn.BatchNorm2d}
        _ACTLAYERS = {"silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}

        norm_layer_cls = _NORMLAYERS[norm_layer.lower()]
        ssm_act_cls = _ACTLAYERS.get(ssm_act_layer.lower(), nn.SiLU)
        mlp_act_cls = _ACTLAYERS.get(mlp_act_layer.lower(), nn.GELU)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # ---- patch embedding ----
        _make_patch_embed = {
            "v1": self._make_patch_embed,
            "v2": self._make_patch_embed_v2,
        }[patchembed_version]
        self.patch_embed = _make_patch_embed(
            in_chans, dims[0], patch_size, patch_norm,
            norm_layer_cls, channel_first=self.channel_first)

        # ---- downsample strategy ----
        _make_downsample = {
            "v1": self._make_downsample_pm,
            "v2": self._make_downsample_conv,
            "v3": self._make_downsample_conv3,
        }[downsample_version]

        # ---- stages ----
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            downsample = (
                _make_downsample(dims[i], dims[i + 1], norm_layer_cls, self.channel_first)
                if i < self.num_layers - 1
                else nn.Identity()
            )
            depth_dp = dpr[sum(depths[:i]):sum(depths[:i + 1])]
            blocks = []
            for d in range(len(depth_dp)):
                blocks.append(VSSBlock(
                    hidden_dim=dims[i],
                    drop_path=depth_dp[d],
                    norm_layer=norm_layer_cls,
                    channel_first=self.channel_first,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_cls,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act_cls,
                    mlp_drop_rate=mlp_drop_rate,
                    gmlp=gmlp,
                    use_checkpoint=use_checkpoint,
                ))
            self.layers.append(nn.Sequential(OrderedDict(
                blocks=nn.Sequential(*blocks),
                downsample=downsample,
            )))

        # ---- per-stage output norms (for feature extraction) ----
        for i in out_indices:
            layer = norm_layer_cls(dims[i])
            self.add_module(f"outnorm{i}", layer)

        self.apply(self._init_weights)

        # ---- load pretrained weights ----
        self._pretrained_loaded = False
        if pretrained is not None:
            self._pretrained_loaded = bool(self.load_pretrained(pretrained))

    # ---- weight init ----
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # ---- patch-embed variants ----
    @staticmethod
    def _make_patch_embed(in_chans, embed_dim, patch_size, patch_norm,
                          norm_layer, channel_first=False):
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                      stride=patch_size, bias=True),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans, embed_dim, patch_size, patch_norm,
                             norm_layer, channel_first=False):
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            (nn.Identity() if (channel_first or not patch_norm)
             else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or not patch_norm)
             else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    # ---- downsample variants ----
    @staticmethod
    def _make_downsample_pm(dim, out_dim, norm_layer, channel_first):
        """PatchMerging-style (v1): 2×2 patch merge → linear."""
        Linear = Linear2d if channel_first else nn.Linear
        _pad = (VSSM._pm_pad_cf if channel_first
                else VSSM._pm_pad_cl)
        return _PatchMerging(dim, out_dim, norm_layer, Linear, _pad)

    @staticmethod
    def _pm_pad_cl(x):
        H, W, _ = x.shape[-3:]
        if W % 2 != 0 or H % 2 != 0:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]
        x1 = x[..., 1::2, 0::2, :]
        x2 = x[..., 0::2, 1::2, :]
        x3 = x[..., 1::2, 1::2, :]
        return torch.cat([x0, x1, x2, x3], -1)

    @staticmethod
    def _pm_pad_cf(x):
        H, W = x.shape[-2:]
        if W % 2 != 0 or H % 2 != 0:
            x = F.pad(x, (0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2]
        x1 = x[..., 1::2, 0::2]
        x2 = x[..., 0::2, 1::2]
        x3 = x[..., 1::2, 1::2]
        return torch.cat([x0, x1, x2, x3], 1)

    @staticmethod
    def _make_downsample_conv(dim, out_dim, norm_layer, channel_first):
        """Conv2d stride-2 downsample (v2)."""
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_downsample_conv3(dim, out_dim, norm_layer, channel_first):
        """Conv2d 3×3 stride-2 downsample (v3)."""
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    # ---- checkpoint loading ----
    def load_pretrained(self, ckpt_path: str, key: str = "model") -> bool:
        """
        Load an official ImageNet-pretrained VMamba-Tiny checkpoint.

        Parameters
        ----------
        ckpt_path : str
            Path to the .pth checkpoint file. If not found locally, candidate
            locations and the official GitHub release URL will be checked.
        key : str
            Key in checkpoint dict holding the state dict (default 'model').

        Returns
        -------
        bool
            True if weights were successfully loaded, False otherwise.
        """
        resolved_path = None
        candidates = [
            ckpt_path,
            os.path.join(os.getcwd(), ckpt_path),
            os.path.join(os.getcwd(), "model_weights", os.path.basename(ckpt_path)),
            os.path.join(os.getcwd(), "model_weights", "vmamba_tiny_e292.pth"),
            "/home/admin/Desktop/mamba/third_party/VMamba/checkpoints/vmamba_tiny_e292.pth",
        ]
        for c in candidates:
            if c and os.path.isfile(c):
                resolved_path = c
                break

        if resolved_path is None:
            # Attempt download from official MzeroMiko/VMamba releases
            official_url = "https://github.com/MzeroMiko/VMamba/releases/download/%23v2/vmamba_tiny_e292.pth"
            target_dir = os.path.join(os.getcwd(), "model_weights")
            os.makedirs(target_dir, exist_ok=True)
            target_path = os.path.join(target_dir, "vmamba_tiny_e292.pth")
            print(f"[VMambaEncoder] Checkpoint not found at '{ckpt_path}'. Downloading official VMamba-Tiny from:\n  {official_url}\n  -> {target_path}")
            try:
                torch.hub.download_url_to_file(official_url, target_path)
                resolved_path = target_path
            except Exception as e:
                print(f"[VMambaEncoder] ERROR: Failed to download official VMamba-Tiny checkpoint: {e}")
                return False

        print(f"\n{'=' * 75}")
        print(f"Loading VMamba-Tiny Pretrained Checkpoint: {resolved_path}")
        print(f"{'=' * 75}")

        try:
            ckpt = torch.load(resolved_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[VMambaEncoder] ERROR: Could not read checkpoint file: {e}")
            return False

        if isinstance(ckpt, dict):
            if key in ckpt:
                state_dict = ckpt[key]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        model_dict = self.state_dict()
        matched_weights = {}
        ignored_head_keys = []
        unmatched_ckpt_keys = []
        shape_mismatch_keys = []

        for k, v in state_dict.items():
            # 1. Skip ImageNet classifier head keys
            if k.startswith(("head.", "classifier.")):
                ignored_head_keys.append(k)
                continue

            # 2. Key remapping to match VSSM module names
            new_k = k
            if new_k == "patch_embed.proj.weight":
                new_k = "patch_embed.0.weight"
            elif new_k == "patch_embed.proj.bias":
                new_k = "patch_embed.0.bias"
            elif new_k == "patch_embed.norm.weight":
                new_k = "patch_embed.2.weight"
            elif new_k == "patch_embed.norm.bias":
                new_k = "patch_embed.2.bias"
            elif new_k.startswith("layers."):
                new_k = new_k.replace(".self_attention.", ".op.")
                new_k = new_k.replace(".ln_1.", ".norm.")
            elif new_k == "norm.weight":
                new_k = "outnorm3.weight"
            elif new_k == "norm.bias":
                new_k = "outnorm3.bias"

            # 3. Validation against model parameter layout
            if new_k in model_dict:
                if model_dict[new_k].shape == v.shape:
                    matched_weights[new_k] = v
                else:
                    shape_mismatch_keys.append(
                        f"{k} -> {new_k}: ckpt {tuple(v.shape)} vs model {tuple(model_dict[new_k].shape)}"
                    )
            else:
                unmatched_ckpt_keys.append(f"{k} (mapped as: {new_k})")

        unmatched_model_keys = [k for k in model_dict if k not in matched_weights]

        # Detailed verification reporting
        print(f"  Total keys in checkpoint     : {len(state_dict)}")
        print(f"  Total keys in encoder model  : {len(model_dict)}")
        print(f"  Successfully matched keys    : {len(matched_weights)} / {len(model_dict)}")
        print(f"  Ignored classification keys  : {len(ignored_head_keys)} ({', '.join(ignored_head_keys)})")

        if shape_mismatch_keys:
            print("  [ERROR] Shape mismatches encountered:")
            for sm in shape_mismatch_keys:
                print(f"    - {sm}")
            raise ValueError(f"Shape mismatches while loading pretrained VMamba backbone: {shape_mismatch_keys}")

        if unmatched_ckpt_keys:
            print(f"  [WARNING] Unmatched keys in checkpoint ({len(unmatched_ckpt_keys)}):")
            for uk in unmatched_ckpt_keys:
                print(f"    - {uk}")

        if unmatched_model_keys:
            print(f"  Unmatched encoder keys (kept random/identity init) ({len(unmatched_model_keys)}):")
            for um in sorted(unmatched_model_keys):
                print(f"    - {um}")

        # Update model dict and load with strict=True
        model_dict.update(matched_weights)
        self.load_state_dict(model_dict, strict=True)
        print(f"{'=' * 75}")
        print(f"SUCCESS: Pretrained VMamba-Tiny backbone weights successfully loaded! ({len(matched_weights)} parameters)")
        print(f"{'=' * 75}\n")
        return True

    # ---- forward ----
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            list of 4 feature maps, each (B, C_i, H_i, W_i)
        """
        x = self.patch_embed(x)   # (B, H/4, W/4, C0) or channel-first

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer.blocks(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f"outnorm{i}")
                out = norm_layer(x)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2)  # → (B, C, H, W)
                outs.append(out.contiguous())
            x = layer.downsample(x)

        return outs


class _PatchMerging(nn.Module):
    """PatchMerging downsample (VMamba v1 style)."""

    def __init__(self, dim, out_dim, norm_layer, Linear, pad_fn):
        super().__init__()
        self.reduction = Linear(4 * dim, out_dim, bias=False)
        self.norm = norm_layer(4 * dim)
        self._pad = pad_fn

    def forward(self, x):
        x = self._pad(x)
        x = self.norm(x)
        x = self.reduction(x)
        return x


# =====================================================================
# VMambaEncoder – the public API for GeoSeg
# =====================================================================

class VMambaEncoder(nn.Module):
    """
    VMamba-Tiny feature-extraction encoder for GeoSeg segmentation models.

    Everything lives under ``self.backbone`` so that GeoSeg's
    ``process_model_params(net, {"backbone.*": dict(lr=backbone_lr, ...)})``
    automatically assigns a separate (lower) learning rate to the encoder.

    Parameters
    ----------
    pretrained_ckpt : str or None
        Path to a VMamba-Tiny ImageNet classification checkpoint
        (e.g. ``vssm_tiny_0230_ckpt_epoch_262.pth``).
    depths : list
        Depths per stage.  Default ``[2, 2, 9, 2]`` (VMamba-Tiny).
    dims : int
        Base channel width.  Default ``96`` → auto-expanded to
        ``[96, 192, 384, 768]``.
    drop_path_rate : float
        Stochastic depth rate.
    use_checkpoint : bool
        Gradient checkpointing.
    """

    # Channel widths exposed for downstream decoders.
    channels = (96, 192, 384, 768)

    def __init__(
        self,
        pretrained_ckpt: Optional[str] = None,
        depths: list = (2, 2, 9, 2),
        dims: int = 96,
        drop_path_rate: float = 0.2,
        use_checkpoint: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.backbone = VSSM(
            patch_size=4,
            in_chans=3,
            depths=list(depths),
            dims=dims,
            drop_path_rate=drop_path_rate,
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v0",
            mlp_ratio=0.0,   # VMamba-Tiny vanilla has no MLP
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            patch_norm=True,
            norm_layer="ln",
            downsample_version="v1",
            patchembed_version="v1",
            use_checkpoint=use_checkpoint,
            out_indices=(0, 1, 2, 3),
            pretrained=pretrained_ckpt,
        )

        # Expose dims for downstream code
        self.channels = tuple(
            self.backbone.dims[i] for i in self.backbone.out_indices
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (B, 3, 512, 512)

        Returns:
            [res1, res2, res3, res4] with shapes:
                res1: (B,  96, 128, 128)
                res2: (B, 192,  64,  64)
                res3: (B, 384,  32,  32)
                res4: (B, 768,  16,  16)
        """
        return self.backbone(x)

    def load_pretrained(self, ckpt_path: str, key: str = "model") -> bool:
        """Load pretrained VMamba-Tiny ImageNet weights into backbone."""
        return self.backbone.load_pretrained(ckpt_path, key=key)

    def get_param_groups(self, backbone_lr: float, backbone_wd: float):
        """
        Return parameter groups suitable for ``torch.optim`` with a
        separate (usually lower) learning rate for the backbone.

        This is an alternative to using ``process_model_params`` —
        call it directly when you need fine-grained control.

        Example::

            groups = encoder.get_param_groups(backbone_lr=6e-5, backbone_wd=0.01)
            optimizer = torch.optim.AdamW(groups, lr=6e-4, weight_decay=0.01)
        """
        return [
            {"params": list(self.backbone.parameters()),
             "lr": backbone_lr, "weight_decay": backbone_wd},
        ]


# =====================================================================
# Quick self-test
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VMambaEncoder – shape verification test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = VMambaEncoder(pretrained_ckpt=None).to(device)

    x = torch.randn(2, 3, 512, 512, device=device)
    features = encoder(x)

    expected = [
        (2, 96, 128, 128),
        (2, 192, 64, 64),
        (2, 384, 32, 32),
        (2, 768, 16, 16),
    ]

    print(f"\nInput  : {tuple(x.shape)}")
    for i, (feat, exp) in enumerate(zip(features, expected)):
        shape = tuple(feat.shape)
        status = "✓" if shape == exp else "✗"
        print(f"res{i+1}   : {shape}  (expected {exp})  {status}")

    assert len(features) == 4
    for feat, exp in zip(features, expected):
        assert tuple(feat.shape) == exp, f"Shape mismatch: {tuple(feat.shape)} != {exp}"

    # Parameter group test
    groups = encoder.get_param_groups(backbone_lr=6e-5, backbone_wd=0.01)
    n_params = sum(p.numel() for p in groups[0]["params"])
    print(f"\nBackbone params : {n_params:,}")
    print(f"Channels       : {encoder.channels}")
    print("\nAll assertions passed ✓")
