"""
DenseContactNet — ResNet18 encoder + U-Net decoder + spatial output heads.

Outputs per forward pass:
    contact_map  : (B, H, W)  contact probability in [0, 1]
    depth_map    : (B, H, W)  normalised depth    in [0, 1]
    pressure_map : (B, H, W)  normalised pressure in [0, 1]
    loc_x        : (B,)       contact X in mm  (via soft-argmax)
    loc_y        : (B,)       contact Y in mm  (via soft-argmax)
    displacement : (B,)       normalised total depth
    force        : (B,)       normalised total force

H = GRID_H = 10,  W = GRID_W = 27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from dataset import GRID_H, GRID_W, GRID_X, GRID_Y


# ── Decoder building block ────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    Upsample x to match the spatial size of skip, concatenate, then fuse with
    two 3×3 convolutions.

    in_ch   : channels of the upsampled (coarser) feature map
    skip_ch : channels of the skip connection from the encoder
    out_ch  : output channels after fusion
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


# ── Main network ──────────────────────────────────────────────────────────────

class DenseContactNet(nn.Module):
    """
    Encoder:  ResNet18 pretrained on ImageNet, split into 5 stages.
    Decoder:  4 U-Net decoder blocks, ending at 112×112×64.
    Pool:     AdaptiveAvgPool2d → (GRID_H × GRID_W) = (10 × 27).
    Heads:
      - Three 1×1 conv heads on the pooled feature map:
          contact_head  → sigmoid    → contact probability map
          depth_head    → ReLU       → depth map
          pressure_head → ReLU       → pressure map
      - One MLP scalar head off the bottleneck:
          → [normalised_displacement, normalised_force]
    Soft-argmax on the contact map extracts (loc_x, loc_y) in mm.

    Spatial dimensions at each encoder stage (input 224×224):
        enc_stem  : 112×112, 64 ch   (conv1 + bn1 + relu)
        enc_pool  : 56×56,   64 ch   (maxpool)  ← fed into enc1
        enc1      : 56×56,   64 ch   (layer1)
        enc2      : 28×28,  128 ch   (layer2)
        enc3      : 14×14,  256 ch   (layer3)
        enc4      : 7×7,   512 ch   (layer4)  ← bottleneck
    """

    def __init__(self):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        bb = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        self.enc_stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu)  # 224→112, 64ch
        self.enc_pool = bb.maxpool                                  # 112→56,  64ch
        self.enc1     = bb.layer1                                   # 56×56,   64ch
        self.enc2     = bb.layer2                                   # 28×28,  128ch
        self.enc3     = bb.layer3                                   # 14×14,  256ch
        self.enc4     = bb.layer4                                   # 7×7,   512ch

        # ── Decoder ──────────────────────────────────────────────────────────
        # (in_ch from coarser path, skip_ch from encoder, out_ch)
        self.dec4 = DecoderBlock(512, 256, 256)   # 7→14
        self.dec3 = DecoderBlock(256, 128, 128)   # 14→28
        self.dec2 = DecoderBlock(128,  64,  64)   # 28→56
        self.dec1 = DecoderBlock( 64,  64,  64)   # 56→112

        # ── Pool to sensor grid ───────────────────────────────────────────────
        self.grid_pool = nn.AdaptiveAvgPool2d((GRID_H, GRID_W))  # 112→(10, 27)

        # ── Spatial heads (1×1 conv = independent linear projection per cell) ─
        self.contact_head  = nn.Conv2d(64, 1, 1)
        self.depth_head    = nn.Conv2d(64, 1, 1)
        self.pressure_head = nn.Conv2d(64, 1, 1)

        # ── Scalar head off bottleneck ────────────────────────────────────────
        self.scalar_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 2),   # [displacement, force]
            nn.Sigmoid(),        # outputs in (0, 1) matching normalised targets
        )

        # ── Grid coordinate buffers for soft-argmax ───────────────────────────
        self.register_buffer('grid_x', torch.from_numpy(GRID_X))  # (W,)
        self.register_buffer('grid_y', torch.from_numpy(GRID_Y))  # (H,)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Encoder — save skip connections
        s0 = self.enc_stem(x)             # 112×112, 64ch
        s1 = self.enc1(self.enc_pool(s0)) # 56×56,   64ch
        s2 = self.enc2(s1)                # 28×28,  128ch
        s3 = self.enc3(s2)                # 14×14,  256ch
        s4 = self.enc4(s3)                # 7×7,   512ch  (bottleneck)

        # Scalar head off bottleneck
        scalars = self.scalar_head(s4)    # (B, 2)

        # Decoder — each stage upsamples and fuses with skip
        d = self.dec4(s4, s3)             # 14×14,  256ch
        d = self.dec3(d,  s2)             # 28×28,  128ch
        d = self.dec2(d,  s1)             # 56×56,   64ch
        d = self.dec1(d,  s0)             # 112×112,  64ch

        # Pool to sensor grid resolution
        g = self.grid_pool(d)             # (B, 64, 10, 27)

        # Spatial heads
        contact  = torch.sigmoid(self.contact_head(g)).squeeze(1)   # (B, 10, 27)
        depth    = F.relu(self.depth_head(g)).squeeze(1)             # (B, 10, 27)
        pressure = F.relu(self.pressure_head(g)).squeeze(1)          # (B, 10, 27)

        # Extract (loc_x, loc_y) as continuous mm coordinates
        loc_x, loc_y = self._soft_argmax(contact)

        return {
            'contact_map':  contact,
            'depth_map':    depth,
            'pressure_map': pressure,
            'loc_x':        loc_x,
            'loc_y':        loc_y,
            'displacement': scalars[:, 0],
            'force':        scalars[:, 1],
        }

    # ── Soft-argmax ───────────────────────────────────────────────────────────

    def _soft_argmax(
        self, contact_map: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable location extraction.

        Applies a temperature-scaled softmax over all grid cells so that the
        result is a probability-weighted average of the cell coordinates.
        Temperature τ=10 sharpens the distribution around the peak while
        keeping gradients well-behaved.

        Args:
            contact_map : (B, H, W)
        Returns:
            loc_x, loc_y : (B,) in mm
        """
        B, H, W = contact_map.shape
        tau = 10.0

        # Flatten, sharpen, reshape back
        weights = torch.softmax(contact_map.view(B, -1) * tau, dim=-1)
        weights = weights.view(B, H, W)

        # Broadcast grid coords over batch dimension
        gx = self.grid_x.view(1, 1, W).expand(B, H, W)  # (B, H, W)
        gy = self.grid_y.view(1, H, 1).expand(B, H, W)  # (B, H, W)

        loc_x = (weights * gx).sum(dim=(1, 2))
        loc_y = (weights * gy).sum(dim=(1, 2))

        return loc_x, loc_y
