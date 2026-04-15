"""
gaussian_diffusion.py
=====================
Implements the DDPM forward (noising) and reverse (denoising) processes.

Key equations from Ho et al. 2020:
  Forward:  q(x_t | x_0) = N(x_t; sqrt(ᾱ_t)*x_0, (1-ᾱ_t)*I)
  Reverse:  p(x_{t-1}|x_t) ≈ N(μ_θ(x_t,t), β_t*I)
  Loss:     E[||ε - ε_θ(sqrt(ᾱ_t)*x_0 + sqrt(1-ᾱ_t)*ε, t)||²]
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Noise Schedules
# ---------------------------------------------------------------------------

def linear_beta_schedule(timesteps):
    """β increases linearly from 1e-4 → 0.02 (original DDPM)."""
    return torch.linspace(1e-4, 0.02, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule (Improved DDPM, Nichol & Dhariwal 2021).
    Smoother — less aggressive noise at the start and end.
    """
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    f = torch.cos((steps / timesteps + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(0, 0.999).float()


def sigmoid_beta_schedule(timesteps):
    """Sigmoid schedule — smooth S-curve from low → high noise."""
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (0.02 - 1e-4) + 1e-4


SCHEDULES = {
    "linear":  linear_beta_schedule,
    "cosine":  cosine_beta_schedule,
    "sigmoid": sigmoid_beta_schedule,
}


# ---------------------------------------------------------------------------
# GaussianDiffusion
# ---------------------------------------------------------------------------

class GaussianDiffusion:
    """
    Manages the DDPM forward noising and reverse denoising processes.

    The UNet model predicts either:
      "pred_noise"   : ε_θ(x_t, t)   — predict the added noise
      "pred_x_start" : x̂_0(x_t, t)  — predict the clean image directly
    Both are equivalent; one can be derived from the other.
    """

    def __init__(
        self,
        model,
        image_size=32,
        timesteps=1000,
        beta_schedule="cosine",
        objective="pred_noise",
    ):
        self.model      = model
        self.image_size = image_size
        self.timesteps  = timesteps
        self.objective  = objective

        # ── Build noise schedule ──────────────────────────────────────────
        betas = SCHEDULES[beta_schedule](timesteps)          # β_1 … β_T
        alphas = 1.0 - betas                                 # α_t = 1 - β_t
        alphas_cumprod = torch.cumprod(alphas, dim=0)        # ᾱ_t = Π_{s=1}^t α_s
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Forward process coefficients  q(x_t | x_0)
        self.sqrt_alphas_cumprod           = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod).sqrt()

        # Reverse process posterior coefficients  q(x_{t-1} | x_t, x_0)
        # σ²_t  = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        self.posterior_variance = (
            betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )
        # μ_q coefs (Eq. 7 in paper)
        self.posterior_mean_coef1 = (
            betas * alphas_cumprod_prev.sqrt() / (1 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1 - alphas_cumprod_prev) * alphas.sqrt() / (1 - alphas_cumprod)
        )

    # ── Internal helper ────────────────────────────────────────────────────

    def _extract(self, arr, t, x_shape):
        """
        Gather schedule values at timestep indices t, reshape to
        (B, 1, 1, 1) for broadcasting against (B, C, H, W).
        """
        # Move arr to the same device as t
        arr = arr.to(t.device)
        vals = arr[t].float()
        return vals.reshape(x_shape[0], *((1,) * (len(x_shape) - 1)))

    # ── Normalisation ──────────────────────────────────────────────────────

    def normalize(self, x):
        """[0, 1] → [-1, 1]  (model expects this range)"""
        return x * 2 - 1

    def unnormalize(self, x):
        """[-1, 1] → [0, 1]  (for display)"""
        return (x + 1) / 2

    # ── Forward Process ───────────────────────────────────────────────────

    def q_sample(self, x_start, t, noise=None):
        """
        Sample x_t from q(x_t | x_0) — the closed-form forward process.

          x_t = sqrt(ᾱ_t) · x_0  +  sqrt(1 - ᾱ_t) · ε,   ε ~ N(0, I)

        No need to run t Markov steps; this is a single-step reparameterization.

        Args:
            x_start : (B, C, H, W)  clean images in [-1, 1]
            t       : (B,)          timestep indices
            noise   : (B, C, H, W)  optional pre-sampled ε

        Returns:
            x_t     : (B, C, H, W)  noisy images
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        s1 = self._extract(self.sqrt_alphas_cumprod,           t, x_start.shape)
        s2 = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return s1 * x_start + s2 * noise

    # ── Conversion helpers ────────────────────────────────────────────────

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Recover x_0 given x_t and noise ε:
          x_0 = (x_t - sqrt(1 - ᾱ_t) · ε) / sqrt(ᾱ_t)
        """
        s1 = self._extract(self.sqrt_alphas_cumprod,           t, x_t.shape)
        s2 = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - s2 * noise) / s1.clamp(min=1e-8)

    def predict_noise_from_start(self, x_t, t, x_start):
        """
        Recover ε given x_t and x_0:
          ε = (x_t - sqrt(ᾱ_t) · x_0) / sqrt(1 - ᾱ_t)
        """
        s1 = self._extract(self.sqrt_alphas_cumprod,           t, x_t.shape)
        s2 = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - s1 * x_start) / s2.clamp(min=1e-8)

    # ── Training Loss ─────────────────────────────────────────────────────

    def p_losses(self, x_start, model_kwargs=None):
        """
        Compute the DDPM denoising loss (simplified ELBO).

        Steps:
          1. Sample t ~ Uniform{0, T-1}
          2. Sample ε ~ N(0, I)
          3. Build noisy x_t via q_sample
          4. Predict ε (or x_0) with the model
          5. Return MSE between prediction and target

        Args:
            x_start      : (B, C, H, W)  clean training images in [-1, 1]
            model_kwargs : dict, e.g. {"text_emb": tensor}

        Returns:
            loss : scalar MSE
        """
        b, device = x_start.shape[0], x_start.device
        t     = torch.randint(0, self.timesteps, (b,), device=device).long()
        noise = torch.randn_like(x_start)
        x_t   = self.q_sample(x_start, t, noise)

        pred   = self.model(x_t, t, model_kwargs)
        target = noise if self.objective == "pred_noise" else x_start
        return F.mse_loss(pred, target)

    # ── Reverse Process ───────────────────────────────────────────────────

    def p_sample(self, x_t, t_index, model_kwargs=None):
        """
        One reverse denoising step: x_t → x_{t-1}.

        Uses the DDPM posterior (Eq. 6 & 7 in paper):
          x̂_0    = derived from model output
          μ_θ    = coef1 · x̂_0  +  coef2 · x_t
          x_{t-1} = μ_θ  +  sqrt(σ²_t) · z,   z=0 at t=0

        Args:
            x_t      : (B, C, H, W)
            t_index  : int  scalar timestep
            model_kwargs : conditioning dict

        Returns:
            x_{t-1} : (B, C, H, W)
        """
        b, device = x_t.shape[0], x_t.device
        t_batch   = torch.full((b,), t_index, device=device, dtype=torch.long)

        pred = self.model(x_t, t_batch, model_kwargs)

        if self.objective == "pred_noise":
            x0_pred = self.predict_start_from_noise(x_t, t_batch, pred)
        else:
            x0_pred = pred
        x0_pred = x0_pred.clamp(-1, 1)

        c1  = self._extract(self.posterior_mean_coef1, t_batch, x_t.shape)
        c2  = self._extract(self.posterior_mean_coef2, t_batch, x_t.shape)
        mu  = c1 * x0_pred + c2 * x_t

        var   = self._extract(self.posterior_variance, t_batch, x_t.shape)
        noise = torch.randn_like(x_t) if t_index > 0 else torch.zeros_like(x_t)
        return mu + var.sqrt() * noise

    @torch.no_grad()
    def sample(self, batch_size, model_kwargs=None, device="cpu", return_all_timesteps=False):
        """
        Generate images by running the full reverse chain x_T → … → x_0.

        Args:
            batch_size           : int
            model_kwargs         : dict (e.g. {"text_emb": ..., "cfg_scale": 2.0})
            device               : torch device
            return_all_timesteps : if True return every intermediate frame

        Returns:
            (B, C, H, W) in [0,1]  or  (B, T+1, C, H, W) if return_all_timesteps
        """
        shape = (batch_size, 3, self.image_size, self.image_size)
        x = torch.randn(shape, device=device)

        frames = [x]
        for t in tqdm(reversed(range(self.timesteps)), desc="Sampling", total=self.timesteps):
            x = self.p_sample(x, t, model_kwargs)
            frames.append(x)

        if return_all_timesteps:
            return torch.stack([self.unnormalize(f).clamp(0, 1) for f in frames], dim=1)
        return self.unnormalize(x).clamp(0, 1)
