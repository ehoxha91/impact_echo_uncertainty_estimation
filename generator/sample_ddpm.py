"""Sample 2000 diffusion-augmented IE signals using EMA weights.

Class composition matches the 2000-sample modal-synth run for a fair A/B:
  75% sound / 25% defect, defect split evenly across 4 kinds.
Output (data/X_train_860_diffusion.npy, data/y_train_diffusion.npy) matches
the training data format: float64 in [0,1], multi-class y in {0..4}.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from generator.ddpm import DDPM
from generator.unet1d import UNet1D


N_TOTAL = 2000
REAL_RATIO = np.array([1307, 111, 112, 112, 110], dtype=float)
REAL_RATIO /= REAL_RATIO.sum()
DDIM_STEPS = 50
CFG_SCALE = 2.0
BATCH = 64

OUT_X = Path("data/X_train_860_diffusion.npy")
OUT_Y = Path("data/y_train_diffusion.npy")
CKPT = "weights/ddpm_ie_best.pth"


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    model = UNet1D(num_classes=5, base_ch=64, cond_dim=256).to(device)
    ck = torch.load(CKPT, map_location=device)
    model.load_state_dict(ck["ema"])
    model.eval()
    print(f"loaded EMA weights from epoch {ck['epoch']} (loss {ck['loss']:.5f})")

    counts = np.round(REAL_RATIO * N_TOTAL).astype(int)
    counts[0] += N_TOTAL - counts.sum()
    print(f"class counts (0..4): {counts.tolist()}")

    y_all = np.concatenate([np.full(int(c), i, dtype=np.int64) for i, c in enumerate(counts)])
    rng = np.random.default_rng(0)
    rng.shuffle(y_all)

    ddpm = DDPM(T=1000, device=device)
    X_out = np.zeros((N_TOTAL, 860), dtype=np.float64)

    for start in range(0, N_TOTAL, BATCH):
        end = min(start + BATCH, N_TOTAL)
        y_b = torch.from_numpy(y_all[start:end]).to(device)
        x_b = ddpm.ddim_sample(model, y_b.shape[0], length=864, y=y_b,
                               steps=DDIM_STEPS, cfg_scale=CFG_SCALE)
        x_b = x_b.squeeze(1).cpu().numpy().astype(np.float64)
        x_b = x_b[:, :860]
        x_b = (x_b + 1.0) * 0.5
        # Per-sample min-max renormalize to [0,1] to match dataloader expectations
        mn = x_b.min(axis=1, keepdims=True)
        mx = x_b.max(axis=1, keepdims=True)
        rng_range = mx - mn
        rng_range[rng_range == 0] = 1.0
        X_out[start:end] = (x_b - mn) / rng_range
        print(f"  {end}/{N_TOTAL} sampled")

    OUT_X.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_X, X_out)
    np.save(OUT_Y, y_all.astype(np.float64))
    print(f"saved {OUT_X} shape={X_out.shape} range=[{X_out.min():.3f},{X_out.max():.3f}]")
    print(f"saved {OUT_Y} shape={y_all.shape} bincount={np.bincount(y_all).tolist()}")


if __name__ == "__main__":
    main()
