"""Class-conditional 1D U-Net for epsilon-prediction DDPM on IE signals.

Input shape: (B, 1, L) where L is padded to a multiple of 16 (864 for IE).
Conditioning: diffusion timestep (sinusoidal + MLP) + class id (5 + 1 for null).
Conditioning is fused into each residual block via FiLM (scale + shift).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, ch: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * ch)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return x * (1.0 + scale[:, :, None]) + shift[:, :, None]


class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.film = FiLM(cond_dim, out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.film(h, cond)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Down1D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv1d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Up1D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.ConvTranspose1d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class UNet1D(nn.Module):
    """Channel progression: 64 → 128 → 256 → 256. Four resolution levels."""

    def __init__(self, num_classes: int = 5, base_ch: int = 64, cond_dim: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.cond_dim = cond_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.class_emb = nn.Embedding(num_classes + 1, cond_dim)

        chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 4]
        self.stem = nn.Conv1d(1, chs[0], 3, padding=1)

        self.d1 = ResBlock1D(chs[0], chs[0], cond_dim)
        self.d1b = ResBlock1D(chs[0], chs[0], cond_dim)
        self.down1 = Down1D(chs[0])

        self.d2 = ResBlock1D(chs[0], chs[1], cond_dim)
        self.d2b = ResBlock1D(chs[1], chs[1], cond_dim)
        self.down2 = Down1D(chs[1])

        self.d3 = ResBlock1D(chs[1], chs[2], cond_dim)
        self.d3b = ResBlock1D(chs[2], chs[2], cond_dim)
        self.down3 = Down1D(chs[2])

        self.mid1 = ResBlock1D(chs[2], chs[3], cond_dim)
        self.mid2 = ResBlock1D(chs[3], chs[3], cond_dim)

        self.up3 = Up1D(chs[3])
        self.u3 = ResBlock1D(chs[3] + chs[2], chs[2], cond_dim)
        self.u3b = ResBlock1D(chs[2], chs[2], cond_dim)

        self.up2 = Up1D(chs[2])
        self.u2 = ResBlock1D(chs[2] + chs[1], chs[1], cond_dim)
        self.u2b = ResBlock1D(chs[1], chs[1], cond_dim)

        self.up1 = Up1D(chs[1])
        self.u1 = ResBlock1D(chs[1] + chs[0], chs[0], cond_dim)
        self.u1b = ResBlock1D(chs[0], chs[0], cond_dim)

        self.out_norm = nn.GroupNorm(min(8, chs[0]), chs[0])
        self.out_conv = nn.Conv1d(chs[0], 1, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, L), t: (B,) ints in [0, T), y: (B,) class ids (num_classes = null)."""
        t_emb = sinusoidal_time_embedding(t, self.cond_dim)
        t_emb = self.time_mlp(t_emb)
        y_emb = self.class_emb(y)
        cond = t_emb + y_emb

        h0 = self.stem(x)
        h1 = self.d1(h0, cond); h1 = self.d1b(h1, cond)
        h2_in = self.down1(h1)
        h2 = self.d2(h2_in, cond); h2 = self.d2b(h2, cond)
        h3_in = self.down2(h2)
        h3 = self.d3(h3_in, cond); h3 = self.d3b(h3, cond)
        h4_in = self.down3(h3)

        m = self.mid1(h4_in, cond); m = self.mid2(m, cond)

        u3 = self.up3(m)
        u3 = torch.cat([u3, h3], dim=1)
        u3 = self.u3(u3, cond); u3 = self.u3b(u3, cond)
        u2 = self.up2(u3)
        u2 = torch.cat([u2, h2], dim=1)
        u2 = self.u2(u2, cond); u2 = self.u2b(u2, cond)
        u1 = self.up1(u2)
        u1 = torch.cat([u1, h1], dim=1)
        u1 = self.u1(u1, cond); u1 = self.u1b(u1, cond)

        return self.out_conv(F.silu(self.out_norm(u1)))
