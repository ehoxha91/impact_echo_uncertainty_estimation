"""Train the class-conditional 1D DDPM on real DS1 training signals.

Data: data/X_train_860.npy (1752, 860), data/y_train.npy (1752,) with classes
{0 sound, 1 delam, 2 void, 3 honeycomb, 4 overlay_debond}.

Preprocessing: pad 860 -> 864 so length is divisible by 2^3 = 8 (we downsample
three times), and rescale each signal from [0,1] to [-1,1] so the DDPM end
distribution matches N(0, I) after the cosine envelope.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from generator.ddpm import DDPM
from generator.unet1d import UNet1D


def ema_update(ema_model, model, decay: float = 0.999):
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
    for b_ema, b in zip(ema_model.buffers(), model.buffers()):
        b_ema.data.copy_(b.data)


def main():
    X = np.load("data/X_train_860.npy").astype(np.float32)
    y = np.load("data/y_train.npy").astype(np.int64)
    print(f"Loaded X {X.shape}, y {y.shape}, classes {np.bincount(y)}")

    L_pad = 864
    X_pad = np.zeros((X.shape[0], L_pad), dtype=np.float32)
    X_pad[:, :860] = X
    X_pad = X_pad * 2.0 - 1.0
    X_t = torch.from_numpy(X_pad).unsqueeze(1)
    y_t = torch.from_numpy(y)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

    model = UNet1D(num_classes=5, base_ch=64, cond_dim=256).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    ddpm = DDPM(T=1000, device=device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet1D parameters: {n_params:,}")

    epochs = 300
    best_loss = float("inf")
    patience = 40
    stale = 0
    log_every = 10
    Path("weights").mkdir(exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            loss = ddpm.loss(model, xb, yb, class_dropout=0.1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema_update(ema, model, decay=0.999)
            total += loss.item()
            n_batches += 1
        avg = total / max(1, n_batches)

        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:03d}  loss={avg:.5f}")

        if avg < best_loss - 1e-4:
            best_loss = avg
            stale = 0
            torch.save({
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "epoch": epoch,
                "loss": avg,
            }, "weights/ddpm_ie_best.pth")
        else:
            stale += 1
        if stale >= patience:
            print(f"early stop at epoch {epoch} (no improvement for {patience} epochs)")
            break

    print(f"best loss: {best_loss:.5f}")


if __name__ == "__main__":
    main()
