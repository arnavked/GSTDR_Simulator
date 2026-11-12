"""
Bucket 4 -- model architecture.

Consumes exactly what Bucket 3 produces: a raw trace [1168], a signed
heatmap target [1168] (Gaussian bumps at each fault/end-reflection delay,
scaled by gamma), and an object-list label (num_faults, per-fault
location/gamma/regime flags, end_reflection, pairwise resolution info).

Two things decided here that were EXPLICITLY DEFERRED from Bucket 3 rather
than fixed in the generator (see bucket3_dataset_generator.py's module
docstring):

1. Heatmap amplitude imbalance (open/short peaks ~1.0, soft-fault peaks
   ~0.02-0.1). Fixed HERE via a two-head design instead of a single signed
   regression target:
     - a PRESENCE map (unsigned, unit-amplitude bumps at every peak
       regardless of gamma) trained with a class-imbalance-aware loss, so
       detecting a peak doesn't depend on how strong its gamma is.
     - a signed GAMMA map, trained with a masked, RELATIVE-error loss
       (normalized by target magnitude) restricted to true peak locations
       -- this is what stops small-gamma peaks from being drowned out by
       large ones in a plain MSE, without distorting the physics upstream.

2. Heatmap sigma was fixed at 2.5 samples in the generator (also flagged as
   a deferred choice). Not addressed here either -- still a fixed-width
   target. Noted again since it's still open.

Architecture: small 1D U-Net (encoder-decoder with skip connections) for
the two dense heads, plus a linear classification head off the bottleneck
for num_faults (0-4). Kept deliberately small -- the input is 1168 samples,
not an image; a deep architecture is unnecessary and would overfit a
dataset this size before it helps.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=9):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class GSTDRNet(nn.Module):
    """
    Input:  [B, 1, 1168] raw trace
    Outputs:
        presence_logits: [B, 1, 1168] -- pre-sigmoid, peak/no-peak per sample
        gamma_map:        [B, 1, 1168] -- signed gamma value per sample
        num_faults_logits:[B, 5]       -- classification over {0,1,2,3,4} faults
    """

    def __init__(self, base_ch: int = 16, in_channels: int = 1):
        super().__init__()
        # Encoder (3 downsample stages -- 1168 -> 584 -> 292 -> 146)
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool1d(2)

        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)

        # Decoder with skip connections
        self.up3 = nn.ConvTranspose1d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)  # concat with enc3 skip
        self.up2 = nn.ConvTranspose1d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)  # concat with enc2 skip
        self.up1 = nn.ConvTranspose1d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)  # concat with enc1 skip

        self.presence_head = nn.Conv1d(base_ch, 1, kernel_size=1)
        self.gamma_head = nn.Conv1d(base_ch, 1, kernel_size=1)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_ch * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 5),  # num_faults in {0,1,2,3,4}
        )

    def forward(self, x):
        # x: [B, 1, N] ; N=1168 -- pad to a multiple of 8 for clean 3x downsampling
        n = x.shape[-1]
        pad_to = ((n + 7) // 8) * 8
        if pad_to != n:
            x = F.pad(x, (0, pad_to - n))

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        presence_logits = self.presence_head(d1)[..., :n]
        gamma_map = self.gamma_head(d1)[..., :n]
        num_faults_logits = self.classifier(b)

        return presence_logits, gamma_map, num_faults_logits


def gstdr_loss(presence_logits, gamma_map, num_faults_logits,
               presence_target, gamma_target, num_faults_target,
               presence_pos_weight: float = 20.0, gamma_eps: float = 0.05,
               w_presence: float = 1.0, w_gamma: float = 1.0, w_class: float = 0.5):
    """
    presence_target: [B, 1, N] binary (1 at/near any true peak, 0 elsewhere)
    gamma_target:     [B, 1, N] signed gamma heatmap (same as Bucket 3's dense target)
    num_faults_target:[B] long, class index 0-4

    presence_pos_weight compensates for extreme class imbalance -- true
    peaks occupy a tiny fraction of the 1168 samples (a handful of
    Gaussian bumps with sigma=2.5 against a mostly-flat record), so an
    unweighted BCE would trivially predict "no peak everywhere" and still
    get very low loss.

    gamma loss uses RELATIVE error (normalized by target magnitude + eps),
    masked to true peak locations only -- this is what prevents small-gamma
    (soft-fault) peaks from being drowned out by large-gamma (open/short)
    peaks in the loss, which was flagged as a risk back in Bucket 3 and
    deliberately left for training time to handle.
    """
    presence_loss = F.binary_cross_entropy_with_logits(
        presence_logits, presence_target,
        pos_weight=torch.tensor(presence_pos_weight, device=presence_logits.device),
    )

    mask = presence_target
    rel_error = (gamma_map - gamma_target) / (gamma_target.abs() + gamma_eps)
    gamma_loss = (rel_error.pow(2) * mask).sum() / mask.sum().clamp(min=1.0)

    class_loss = F.cross_entropy(num_faults_logits, num_faults_target)

    total = w_presence * presence_loss + w_gamma * gamma_loss + w_class * class_loss
    return total, {"presence": presence_loss.item(), "gamma": gamma_loss.item(), "class": class_loss.item()}