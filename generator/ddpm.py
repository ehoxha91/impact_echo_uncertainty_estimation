"""DDPM schedule, epsilon-prediction loss, and DDIM sampler with CFG."""
from __future__ import annotations

import torch
import torch.nn.functional as F


class DDPM:
    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02,
                 device: torch.device | str = "cpu"):
        self.T = T
        self.device = torch.device(device)
        betas = torch.linspace(beta_start, beta_end, T, device=self.device)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.betas = betas
        self.alphas_bar = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = self.sqrt_alphas_bar[t][:, None, None]
        b = self.sqrt_one_minus_alphas_bar[t][:, None, None]
        return a * x0 + b * noise

    def loss(self, model, x0: torch.Tensor, y: torch.Tensor,
             class_dropout: float = 0.1) -> torch.Tensor:
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)

        null_id = model.num_classes  # reserved embedding index
        drop_mask = torch.rand(B, device=self.device) < class_dropout
        y_cond = torch.where(drop_mask, torch.full_like(y, null_id), y)

        eps_pred = model(xt, t, y_cond)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def ddim_sample(self, model, n: int, length: int, y: torch.Tensor,
                    steps: int = 50, cfg_scale: float = 2.0) -> torch.Tensor:
        """Deterministic DDIM (eta=0) with classifier-free guidance."""
        null_id = model.num_classes
        x = torch.randn(n, 1, length, device=self.device)
        ts = torch.linspace(self.T - 1, 0, steps + 1, device=self.device).long()

        for i in range(steps):
            t = ts[i]
            t_next = ts[i + 1]
            t_batch = torch.full((n,), int(t.item()), device=self.device, dtype=torch.long)

            eps_cond = model(x, t_batch, y)
            eps_uncond = model(x, t_batch, torch.full_like(y, null_id))
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

            a_t = self.alphas_bar[t]
            a_next = self.alphas_bar[t_next] if t_next >= 0 else torch.tensor(1.0, device=self.device)
            x0_pred = (x - torch.sqrt(1.0 - a_t) * eps) / torch.sqrt(a_t)
            x0_pred = x0_pred.clamp(-1.0, 1.0)
            x = torch.sqrt(a_next) * x0_pred + torch.sqrt(1.0 - a_next) * eps

        return x
