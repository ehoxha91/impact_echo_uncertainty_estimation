"""Generate companion-paper figures for the diffusion-augmentation study.

Saves into /Users/evhoxha/Downloads/computer-aided/els-cas-templates/:
  - diffusion_spectra_per_class.pdf  (real vs. generated mean spectrum per class)
  - diffusion_time_examples.pdf      (time-domain examples, one row per class)
  - diffusion_umap.pdf               (UMAP or PCA scatter of real vs. generated)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq


OUT_DIR = Path("/Users/evhoxha/Downloads/computer-aided/els-cas-templates")
CLASS_NAMES = {0: "Sound", 1: "Delamination", 2: "Void", 3: "Honeycomb", 4: "Overlay debond"}
FS = 500_000


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xr = np.load("data/X_train_860.npy").astype(np.float32)
    yr = np.load("data/y_train.npy").astype(int)
    Xg = np.load("data/X_train_860_diffusion.npy").astype(np.float32)
    yg = np.load("data/y_train_diffusion.npy").astype(int)
    return Xr, yr, Xg, yg


def fig_spectra(Xr, yr, Xg, yg, out: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(14, 2.6), sharey=True)
    freqs = rfftfreq(860, d=1.0 / FS) / 1e3
    win = np.hanning(860)
    for i, (ax, cls) in enumerate(zip(axes, [0, 1, 2, 3, 4])):
        r = Xr[yr == cls]
        g = Xg[yg == cls]
        sr = np.abs(rfft(r * win, axis=1)).mean(axis=0)
        sg = np.abs(rfft(g * win, axis=1)).mean(axis=0)
        sr /= sr.max()
        sg /= sg.max()
        ax.semilogy(freqs, sr + 1e-4, lw=1.0, label="Real DS1")
        ax.semilogy(freqs, sg + 1e-4, lw=1.0, label="DDPM", alpha=0.8)
        ax.set_xlim(0, 40)
        ax.set_xlabel("Frequency (kHz)")
        ax.set_title(f"{CLASS_NAMES[cls]} (n={len(r)}/{len(g)})", fontsize=9)
        if i == 0:
            ax.set_ylabel("Normalized |FFT|")
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_time_examples(Xr, yr, Xg, yg, out: Path) -> None:
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(5, 2, figsize=(9, 8), sharex=True)
    t = np.arange(860) / FS * 1e3
    for i, cls in enumerate([0, 1, 2, 3, 4]):
        r = Xr[yr == cls]
        g = Xg[yg == cls]
        idx_r = rng.integers(0, len(r))
        idx_g = rng.integers(0, len(g))
        axes[i, 0].plot(t, r[idx_r], lw=0.6, color="C0")
        axes[i, 0].set_ylabel(CLASS_NAMES[cls], fontsize=9)
        axes[i, 1].plot(t, g[idx_g], lw=0.6, color="C1")
        for ax in axes[i]:
            ax.set_ylim(-0.05, 1.05)
    axes[0, 0].set_title("Real DS1 recording")
    axes[0, 1].set_title("DDPM sample")
    for ax in axes[-1]:
        ax.set_xlabel("Time (ms)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_embedding(Xr, yr, Xg, yg, out: Path) -> None:
    """PCA scatter (UMAP optional). Keeps companion paper zero-dep on umap-learn."""
    from sklearn.decomposition import PCA
    yr_b = (yr > 0).astype(int)
    yg_b = (yg > 0).astype(int)
    # subsample for clarity
    rng = np.random.default_rng(0)
    def sub(X, y, n):
        idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
        return X[idx], y[idx]
    Xr_s, yr_s = sub(Xr, yr_b, 500)
    Xg_s, yg_s = sub(Xg, yg_b, 500)
    Z = np.vstack([Xr_s, Xg_s])
    labels = np.concatenate([yr_s, yg_s])
    source = np.concatenate([np.zeros(len(Xr_s)), np.ones(len(Xg_s))])
    Z = PCA(n_components=2, random_state=0).fit_transform(Z)

    fig, ax = plt.subplots(figsize=(5, 4))
    for s_val, marker, lbl in [(0, "o", "Real"), (1, "x", "DDPM")]:
        for c_val, color in [(0, "tab:blue"), (1, "tab:red")]:
            m = (source == s_val) & (labels == c_val)
            ax.scatter(Z[m, 0], Z[m, 1], s=8, marker=marker, color=color, alpha=0.5,
                       label=f"{lbl} {'defect' if c_val == 1 else 'sound'}")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    Xr, yr, Xg, yg = load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_spectra(Xr, yr, Xg, yg, OUT_DIR / "diffusion_spectra_per_class.pdf")
    fig_time_examples(Xr, yr, Xg, yg, OUT_DIR / "diffusion_time_examples.pdf")
    fig_embedding(Xr, yr, Xg, yg, OUT_DIR / "diffusion_pca.pdf")
    print("saved diffusion paper figures to", OUT_DIR)


if __name__ == "__main__":
    main()
