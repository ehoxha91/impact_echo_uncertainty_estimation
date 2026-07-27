### Algorithm

- Sample training data (i.e., IE signal, or an Image/Audio etc) `x_sample`
- Sample timestep t. This determines the level of noise. (`t = torch.randint(1, timestamp + 1)`)
- Sample the noise (`noise = torch.randn_like(x_sample)`)
- Add noise to the data (IE Signal, Image etc.) (`x_perturbed = perturb_input(x_sample, t, noise)`)
- Input the perturbed signal (`x_perturbed`) to the `model` -> `predicted_noise = model(x_perturbed, t/timesteps)` and predict the noise.
- Compute Loss (`MSE = loss(predicted_noise, noise)`)
- Backpropagate and learn

```python
model.train()
timesteps = 32  # diffusion process timesteps (t=32 removes all the noise)
for epoch in range(n_epochs):
    print(f"Epoch: {epoch}")

    lr_scheduler.step()
    pbar = tqdm(dataloader, mininterval=2)

    for x, _ in pbar:
        optimizer.zero_grad()
        x = x.to(device)

        # perturb data
        noise = torch.randn_like(x)
        t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
        x_perturbed = perturb_input(x, t, noise)

        # use network to predict noise
        predicted_noise = model(x_perturbed, t / timesteps)

        # calculate the MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        loss.backward()
        optimizer.step()
```

## The physics: a Markov noising process

DDPM defines a forward process that gradually destroys a clean sample $x_0$ by adding Gaussian noise over $T$ steps:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\; \sqrt{1 - \beta_t}\, x_{t-1},\; \beta_t\, \mathbf{I}\right)$$

Each step is a tiny **variance-preserving** Gaussian kick governed by a schedule $\beta_t \in (0,1)$. Think of it like Brownian motion: the signal decays multiplicatively while noise variance accumulates.

### Definitions

- $\alpha_t = 1 - \beta_t$  — fraction of signal kept per step
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$  — cumulative signal retention (this is your `ab_t`)

## The closed-form jump

A nice property of Gaussians: you can **skip all intermediate steps** and jump from $x_0$ straight to $x_t$:

$$q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\; \sqrt{\bar{\alpha}_t}\, x_0,\; (1 - \bar{\alpha}_t)\, \mathbf{I}\right)$$

Using the reparameterization trick with $\varepsilon \sim \mathcal{N}(0, \mathbf{I})$:

$$\boxed{\; x_t = \sqrt{\bar{\alpha}_t}\, x_0 \;+\; \sqrt{1 - \bar{\alpha}_t}\, \varepsilon \;}$$

This is the **canonical DDPM equation** (Ho et al. 2020, eq. 4). The square root on $\bar{\alpha}_t$ preserves variance so $\mathrm{Var}(x_t) \approx 1$ for all $t$.

## What the snippet does

```python
ab_t.sqrt()[t] * x  +  (1 - ab_t[t]) * noise
```

It samples $x_t$ directly from $x_0$ using the closed-form jump above — so during training you can pick a random $t$, perturb once, and train the network to predict the noise $\varepsilon$.

## ⚠️ One caveat

The second coefficient is $(1 - \bar{\alpha}_t)$ rather than $\sqrt{1 - \bar{\alpha}_t}$:

| Form           | Noise coefficient             | Variance of $x_t$                                   |
| -------------- | ----------------------------- | --------------------------------------------------- |
| Canonical DDPM | $\sqrt{1 - \bar{\alpha}_t}$   | $\bar{\alpha}_t + (1 - \bar{\alpha}_t) = 1$ ✅       |
| This snippet   | $1 - \bar{\alpha}_t$          | $\bar{\alpha}_t + (1 - \bar{\alpha}_t)^2 \ne 1$ ❌   |

This is the simplification used in the DeepLearning.AI "How Diffusion Models Work" short course. It **breaks the unit-variance property** — the process is no longer variance-preserving. It still works on small toy datasets, but drifts from the probabilistic derivation. For real tasks, use $\sqrt{1 - \bar{\alpha}_t}$.

## Variance Preservation: where it comes from

### The core idea

"Variance-preserving" (VP) means: if your clean data $x_0$ has unit variance, then every noised version $x_t$ also has unit variance. The signal gets replaced by noise, but the total "energy" stays constant.

$$\mathrm{Var}(x_0) = 1 \;\;\Longrightarrow\;\; \mathrm{Var}(x_t) = 1 \quad \forall t$$

### The one-step update

Recall the forward step:

$$x_t = \sqrt{1 - \beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \mathbf{I})$$

The coefficients are **not arbitrary** — they satisfy:

$$\underbrace{(\sqrt{1-\beta_t})^2}_{\text{signal weight}^2} + \underbrace{(\sqrt{\beta_t})^2}_{\text{noise weight}^2} = (1-\beta_t) + \beta_t = 1$$

That "$a^2 + b^2 = 1$" identity is the whole trick.

### The variance algebra

Because $x_{t-1}$ and the fresh noise $\varepsilon_t$ are **independent**, variance of a sum = sum of variances. For any scalars $a, b$:

$$\mathrm{Var}(a\, x_{t-1} + b\, \varepsilon_t) = a^2\, \mathrm{Var}(x_{t-1}) + b^2\, \mathrm{Var}(\varepsilon_t)$$

Plug in $a = \sqrt{1-\beta_t}$, $b = \sqrt{\beta_t}$, and assume $\mathrm{Var}(x_{t-1}) = 1$, $\mathrm{Var}(\varepsilon_t) = 1$:

$$\mathrm{Var}(x_t) = (1-\beta_t)\cdot 1 + \beta_t \cdot 1 = 1 \;\;\checkmark$$

So variance $=1$ propagates by induction: if $\mathrm{Var}(x_0)=1$, then $\mathrm{Var}(x_1)=1$, hence $\mathrm{Var}(x_2)=1$, and so on.

### Why the square roots?

Forgetting the square roots breaks this. Consider a hypothetical "bad" update:

$$x_t \stackrel{?}{=} (1-\beta_t)\, x_{t-1} + \beta_t\, \varepsilon_t$$

Then:

$$\mathrm{Var}(x_t) = (1-\beta_t)^2 + \beta_t^2 \;\ne\; 1$$

For example, with $\beta_t = 0.5$: variance collapses to $0.25 + 0.25 = 0.5$. Iterate this and your signal **shrinks to zero** rather than dissolving into unit-variance noise. That's exactly the issue with the snippet's $(1 - \bar{\alpha}_t)$ coefficient.

### Extending to the closed-form jump

The same $a^2 + b^2 = 1$ identity shows up in the skip-ahead formula:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \varepsilon$$

Check:

$$\underbrace{\bar{\alpha}_t}_{a^2} + \underbrace{(1 - \bar{\alpha}_t)}_{b^2} = 1$$

So $\mathrm{Var}(x_t) = \bar{\alpha}_t \cdot 1 + (1-\bar{\alpha}_t) \cdot 1 = 1$. ✅

### The physical / geometric picture

Think of $x_t$ as a point on a **unit sphere** in high-dimensional space. The forward process rotates the data vector toward a random Gaussian direction while keeping its length fixed:

- At $t=0$: pure signal, $\sqrt{\bar{\alpha}_0} \approx 1$
- At intermediate $t$: mixture, but total length preserved
- At $t=T$: pure noise, $\sqrt{\bar{\alpha}_T} \approx 0$, length still $\approx 1$

It's a smooth interpolation along a great-circle-like path between data and isotropic Gaussian, with **constant radius**.

### Why it matters practically

1. **Stable training.** The network sees inputs of roughly the same scale at every $t$ — no need to rescale activations per timestep. LayerNorm / BatchNorm behave consistently.
2. **Matches the prior.** At $t=T$, $x_T \approx \mathcal{N}(0, \mathbf{I})$, which is the exact distribution you sample from at inference. No scale mismatch between training and sampling.
3. **Connects to SDEs.** Song et al. (2021) show the DDPM forward process is a discretization of the **Variance Preserving SDE**:

$$\mathrm{d}x = -\tfrac{1}{2}\beta(t)\, x\, \mathrm{d}t + \sqrt{\beta(t)}\, \mathrm{d}w$$

The drift $-\tfrac{1}{2}\beta(t)\, x$ pulls toward zero; the diffusion $\sqrt{\beta(t)}\, \mathrm{d}w$ injects noise. These two terms are **balanced** so that the stationary distribution is $\mathcal{N}(0, \mathbf{I})$ — variance neither explodes nor collapses. This contrasts with the **Variance Exploding SDE** (used in score-matching / NCSN), where variance grows unboundedly and the signal is never attenuated.

### Summary

Variance preservation comes from picking forward-process coefficients whose **squares sum to one**, combined with independence of the injected noise. It's the discrete-time echo of an Ornstein–Uhlenbeck-like SDE with a stationary unit-Gaussian distribution — which is exactly the prior you want to sample from at test time.

---

# Implementation walkthrough (`generator/`)

This section explains the key pieces of our implementation step by step. The code lives in four files:

- `ddpm.py` — noise schedule, training loss, DDIM sampler
- `unet1d.py` — 1D U-Net backbone for $\varepsilon$-prediction
- `train_ddpm.py` — training loop on real IE signals
- `sample_ddpm.py` — class-balanced generation using EMA weights

## 1. The noise schedule — `DDPM.__init__` (`ddpm.py:9`)

```python
betas = torch.linspace(beta_start, beta_end, T, device=self.device)
alphas = 1.0 - betas
self.alphas_bar = torch.cumprod(alphas, dim=0)
self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_bar)
```

- Linear $\beta$ schedule from $10^{-4}$ to $0.02$ over $T = 1000$ steps (the canonical Ho et al. 2020 choice).
- Precompute $\sqrt{\bar{\alpha}_t}$ and $\sqrt{1-\bar{\alpha}_t}$ once so `q_sample` becomes two tensor lookups. Note these are the **correct** variance-preserving square roots, unlike the DeepLearning.AI simplification discussed earlier.

## 2. Forward process — `q_sample` (`ddpm.py:21`)

```python
def q_sample(self, x0, t, noise):
    a = self.sqrt_alphas_bar[t][:, None, None]
    b = self.sqrt_one_minus_alphas_bar[t][:, None, None]
    return a * x0 + b * noise
```

Direct implementation of the closed-form jump:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \varepsilon$$

The `[:, None, None]` broadcasting turns per-sample scalars into `(B, 1, 1)` so they multiply correctly against `(B, 1, L)` 1D signals.

## 3. Training loss — `DDPM.loss` (`ddpm.py:26`)

```python
t = torch.randint(0, self.T, (B,), device=self.device, dtype=torch.long)
noise = torch.randn_like(x0)
xt = self.q_sample(x0, t, noise)

null_id = model.num_classes
drop_mask = torch.rand(B, device=self.device) < class_dropout
y_cond = torch.where(drop_mask, torch.full_like(y, null_id), y)

eps_pred = model(xt, t, y_cond)
return F.mse_loss(eps_pred, noise)
```

Three things happen per batch:

1. **Random $t$ per sample** — simple Monte-Carlo estimate of the ELBO over the uniform timestep distribution.
2. **$\varepsilon$-prediction loss** — the network learns to predict the exact Gaussian noise that was injected. This is the simplified objective from Ho et al. 2020 (eq. 14).
3. **Classifier-free guidance (CFG) dropout** — with probability $p = 0.1$ we swap the class label for a reserved `null_id = num_classes`. This trains *one* network to do both conditional and unconditional denoising, enabling guidance at sampling time (see §6).

## 4. The U-Net backbone — `UNet1D` (`unet1d.py:68`)

Channel progression: `64 → 128 → 256 → 256` across four resolution levels. Three key design choices:

### 4a. Sinusoidal time embedding (`unet1d.py:16`)

```python
freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
args = t[:, None].float() * freqs[None, :]
return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
```

The diffusion timestep is injected via the same positional encoding used in Transformers — gives the network a smooth, high-frequency representation of $t \in [0, T)$. Then passed through a 2-layer MLP.

### 4b. FiLM conditioning (`unet1d.py:23`)

```python
def forward(self, x, cond):
    scale, shift = self.proj(cond).chunk(2, dim=-1)
    return x * (1.0 + scale[:, :, None]) + shift[:, :, None]
```

Both timestep embedding and class embedding are fused into a single `cond` vector, which modulates every residual block via **Feature-wise Linear Modulation**: $h \leftarrow h \cdot (1 + \gamma(c)) + \beta(c)$. This is a lightweight alternative to cross-attention and works well for small 1D signals.

### 4c. Standard U-Net with skips

Three downsampling stages (strided conv) + bottleneck + three upsampling stages (transposed conv) with **channel-wise skip concatenation** at each level. Each level has two `ResBlock1D`s (GroupNorm + SiLU + Conv1d + FiLM + residual).

## 5. Training loop — `train_ddpm.py`

### 5a. Preprocessing (`train_ddpm.py:35`)

```python
L_pad = 864
X_pad = np.zeros((X.shape[0], L_pad), dtype=np.float32)
X_pad[:, :860] = X
X_pad = X_pad * 2.0 - 1.0
```

- **Pad 860 → 864** so the length is divisible by $2^3 = 8$ (three downsampling stages).
- **Rescale $[0, 1] \to [-1, 1]$** so the data distribution approximately matches the $\mathcal{N}(0, \mathbf{I})$ prior the forward process converges to. Without this rescaling, the VP property's unit-variance assumption is violated from the start.

### 5b. EMA weights (`train_ddpm.py:23`)

```python
def ema_update(ema_model, model, decay: float = 0.999):
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
```

Every step, maintain an exponential moving average of model weights with decay $0.999$. Sampling uses the EMA copy — this is standard practice in diffusion training and substantially improves sample quality by averaging out optimizer noise.

### 5c. Other training details

- `AdamW(lr=2e-4, weight_decay=1e-5)` — canonical DDPM optimizer
- `clip_grad_norm_(1.0)` — prevents rare large updates from destabilizing training
- Early stopping with `patience=40` epochs tracking best training loss
- Checkpointing both `model` and `ema` state_dicts

## 6. DDIM sampling with CFG — `DDPM.ddim_sample` (`ddpm.py:40`)

This is where generation happens. Two ideas are combined: **DDIM** for fast deterministic sampling, and **CFG** for stronger class conditioning.

### 6a. Skipping the reverse chain with DDIM

```python
ts = torch.linspace(self.T - 1, 0, steps + 1, device=self.device).long()
```

Rather than the full $T = 1000$ reverse steps, we take only $50$ — DDIM (Song et al. 2020) lets you skip timesteps by parameterizing the reverse process deterministically (`eta=0` here).

### 6b. Classifier-free guidance (`ddpm.py:53`)

```python
eps_cond = model(x, t_batch, y)
eps_uncond = model(x, t_batch, torch.full_like(y, null_id))
eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
```

At each step, run the network **twice** — once with the class label, once with the null token — and extrapolate:

$$\hat{\varepsilon} = \varepsilon_{\text{uncond}} + w\cdot (\varepsilon_{\text{cond}} - \varepsilon_{\text{uncond}})$$

With $w = 2.0$ we push generation further in the "this class" direction than the conditional model alone would. This is why we bothered with `class_dropout=0.1` during training.

### 6c. The DDIM update (`ddpm.py:57`)

```python
x0_pred = (x - torch.sqrt(1.0 - a_t) * eps) / torch.sqrt(a_t)
x0_pred = x0_pred.clamp(-1.0, 1.0)
x = torch.sqrt(a_next) * x0_pred + torch.sqrt(1.0 - a_next) * eps
```

Two moves:

1. **Predict $x_0$** by inverting the forward equation with the predicted noise:
   $$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\, \hat{\varepsilon}}{\sqrt{\bar{\alpha}_t}}$$
2. **Re-noise** to the next (smaller) timestep using the same $\hat{\varepsilon}$:
   $$x_{t_\text{next}} = \sqrt{\bar{\alpha}_{t_\text{next}}}\, \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t_\text{next}}}\, \hat{\varepsilon}$$

The `clamp(-1, 1)` is a common stabilization trick — since the training data lives in $[-1, 1]$, we know any "true" $x_0$ must too.

## 7. Post-sampling — `sample_ddpm.py`

### 7a. Class-balanced generation (`sample_ddpm.py:20`)

```python
REAL_RATIO = np.array([1307, 111, 112, 112, 110], dtype=float)
```

We match the real DS1 training-set class proportions: 75% sound / 25% defects split roughly evenly. This mirrors the modal-synthesis baseline for a fair A/B comparison.

### 7b. Output rescaling (`sample_ddpm.py:59`)

```python
x_b = (x_b + 1.0) * 0.5
mn = x_b.min(axis=1, keepdims=True)
mx = x_b.max(axis=1, keepdims=True)
X_out[start:end] = (x_b - mn) / rng_range
```

- First undo the $[-1,1] \to [0,1]$ rescaling from training.
- Then **per-sample min-max normalize** to $[0,1]$ so the outputs look like the real IE signals consumed by the downstream dataloader.

## 8. How the pieces fit together

```
train_ddpm.py
    └── loads X_train_860.npy, y_train.npy
    └── pads 860→864, rescales [0,1]→[-1,1]
    └── for each batch:
          ├── DDPM.loss  (ddpm.py:26)
          │    ├── q_sample(x0, t, noise)         ← forward process (§2)
          │    ├── classifier-free dropout         ← §3
          │    └── UNet1D(xt, t, y_cond)           ← §4
          ├── backward + clip + AdamW step
          └── EMA update                           ← §5b

sample_ddpm.py
    └── loads EMA weights
    └── DDPM.ddim_sample  (ddpm.py:40)
          ├── 50 reverse steps                      ← DDIM (§6a)
          ├── two forward passes per step (CFG)     ← §6b
          └── predict-x0 → re-noise                 ← §6c
    └── rescale back to [0, 1] + min-max normalize
    └── save data/X_train_860_diffusion.npy
```

The whole pipeline is ~200 lines of PyTorch. The theory (forward diffusion, VP, DDIM, CFG) does the heavy lifting; the code is mostly bookkeeping.
