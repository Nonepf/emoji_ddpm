"""
unet.py
=======
A small UNet for DDPM denoising with timestep + text conditioning.

Architecture (dim=d, dim_mults=(1,2,4)):

  x (3,H,W)
      │ init_conv
  (d, H, W)  ──────────────────────────────────────────────── skip_0 ─┐
      │ ResBlock, ResBlock, Downsample                                  │
  (d, H/2, W/2)  ──────────────────────────────────────── skip_1 ─┐   │
      │ ResBlock, ResBlock, Downsample                              │   │
  (2d, H/4, W/4)  ──────────────────────────────────── skip_2 ─┐  │   │
      │ ResBlock, ResBlock  (no downsample at bottleneck)        │  │   │
  (4d, H/4, W/4)  ← mid_block1, mid_block2                      │  │   │
      │ [cat skip_2] ResBlock, ResBlock, Upsample ────────────────┘  │   │
  (2d, H/2, W/2)                                                     │   │
      │ [cat skip_1] ResBlock, ResBlock, Upsample ─────────────────┘   │
  (d, H, W)                                                            │
      │ [cat skip_0] ResBlock, ResBlock ──────────────────────────────┘
  (d, H, W)
      │ final_conv
  (3, H, W)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def groupnorm(channels):
    """GroupNorm that works for any channel count, including very small ones."""
    num_groups = 1
    for g in [8, 4, 2, 1]:
        if channels % g == 0:
            num_groups = g
            break
    return nn.GroupNorm(num_groups, channels)


# ---------------------------------------------------------------------------
# Sinusoidal Timestep Embedding
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """
    Encode scalar timestep t ∈ {0,...,T} as a d-dimensional vector.
    Same idea as Transformer positional encoding — lets the model know
    how noisy the current input is.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """t : (B,) int  →  (B, dim)"""
        device = t.device
        half   = self.dim // 2
        freqs  = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)    # (B, dim)


# ---------------------------------------------------------------------------
# ResNet Block
# ---------------------------------------------------------------------------

class ResnetBlock(nn.Module):
    """
    Two-layer residual block with context conditioning.

    Context (= time embedding + text embedding) is projected and added
    after the first convolution, so every block knows what timestep we're
    at and what text prompt is being used.

    Structure:
      x ──► Conv─GN─SiLU  ──(+ ctx_proj)──► Conv─GN─SiLU ──►(+)──► out
      └──────────────────── 1×1 conv (if ch. change) ──────────────────┘
    """

    def __init__(self, dim, dim_out, context_dim):
        super().__init__()
        self.dim         = dim
        self.dim_out     = dim_out
        self.context_dim = context_dim

        self.block1 = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            groupnorm(dim_out),
            nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, 3, padding=1),
            groupnorm(dim_out),
            nn.SiLU(),
        )
        # Context → dim_out (added channel-wise after block1)
        self.context_proj = nn.Linear(context_dim, dim_out)
        # Residual: 1×1 conv if in/out channels differ, else identity
        self.res_conv = (
            nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        )

    def forward(self, x, context):
        """
        Args:
            x       : (B, dim, H, W)
            context : (B, context_dim)

        Returns:
            out : (B, dim_out, H, W)
        """
        h = self.block1(x)
        h = h + self.context_proj(context)[:, :, None, None]  # broadcast over H,W
        h = self.block2(h)
        return h + self.res_conv(x)


# ---------------------------------------------------------------------------
# Down / Up sampling
# ---------------------------------------------------------------------------

class Downsample(nn.Module):
    """Halve spatial resolution with a strided convolution."""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Double spatial resolution with a transposed convolution."""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------

class Unet(nn.Module):
    """
    UNet denoiser for DDPM with:
      • Sinusoidal timestep embedding
      • Optional text-prompt conditioning (projected CLIP embedding)
      • Classifier-Free Guidance (CFG) via cfg_forward()

    Args:
        dim          : base channel width (smallest feature map channels)
        condition_dim: size of the context vector fed to each ResnetBlock
        dim_mults    : channel multipliers at each resolution level
    """

    def __init__(self, dim, condition_dim, dim_mults=(1, 2, 4)):
        super().__init__()

        # ── Context encoding ──────────────────────────────────────────────
        # Time emb + text emb are each projected to condition_dim//2,
        # then concatenated → condition_dim total context per ResBlock.
        half = condition_dim // 2
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(half),
            nn.Linear(half, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, half),
        )
        # Text embedding: CLIP outputs 512-d; project to half
        self.text_proj = nn.Linear(512, half)
        ctx_dim = condition_dim  # = half + half

        # ── Channel sizes at each level ───────────────────────────────────
        dims   = [dim] + [dim * m for m in dim_mults]   # e.g. [d, d, 2d, 4d]
        in_out = list(zip(dims[:-1], dims[1:]))          # [(d,d),(d,2d),(2d,4d)]

        # ── Initial projection 3 → dim ────────────────────────────────────
        self.init_conv = nn.Conv2d(3, dim, 7, padding=3)

        # ── Encoder ───────────────────────────────────────────────────────
        # Each level: ResBlock → ResBlock → Downsample
        # Last level skips the Downsample (it's the bottleneck input).
        self.downs = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(in_out):
            is_last = (i == len(in_out) - 1)
            self.downs.append(nn.ModuleList([
                ResnetBlock(d_in,  d_out, ctx_dim),
                ResnetBlock(d_out, d_out, ctx_dim),
                nn.Identity() if is_last else Downsample(d_out),
            ]))

        # ── Bottleneck ────────────────────────────────────────────────────
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, ctx_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, ctx_dim)

        # ── Decoder ───────────────────────────────────────────────────────
        # Mirror of encoder.  Skip connections double the input channels.
        # Order per level: cat(skip) → ResBlock → ResBlock → Upsample
        # Last level (shallowest) skips the Upsample.
        self.ups = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(reversed(in_out)):
            is_last = (i == len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                ResnetBlock(d_out * 2, d_out, ctx_dim),  # *2: skip concat
                ResnetBlock(d_out,     d_in,  ctx_dim),
                nn.Identity() if is_last else Upsample(d_in),
            ]))

        # ── Final projection dim → 3 ──────────────────────────────────────
        self.final_conv = nn.Sequential(
            groupnorm(dim),
            nn.SiLU(),
            nn.Conv2d(dim, 3, 1),
        )

    # ── Context builder ───────────────────────────────────────────────────

    def _context(self, time, model_kwargs):
        """
        Build the context vector for ResnetBlocks:
          context = cat([time_emb, text_emb], dim=-1)
        If no text_emb is provided, uses zeros.
        """
        t_emb = self.time_mlp(time)                               # (B, half)
        if model_kwargs and "text_emb" in model_kwargs:
            txt = self.text_proj(model_kwargs["text_emb"])         # (B, half)
        else:
            txt = torch.zeros_like(t_emb)
        return torch.cat([t_emb, txt], dim=-1)                    # (B, ctx_dim)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x, time, model_kwargs=None):
        """
        Standard forward pass (no CFG).

        Args:
            x           : (B, 3, H, W)  noisy image
            time        : (B,)          timestep indices
            model_kwargs: dict; may contain "text_emb" and/or "cfg_scale"

        Returns:
            (B, 3, H, W)  predicted noise or x_0
        """
        if model_kwargs and "cfg_scale" in model_kwargs:
            return self.cfg_forward(x, time, model_kwargs)

        ctx = self._context(time, model_kwargs)
        h   = self.init_conv(x)

        # Encoder — save one skip per level
        skips = []
        for res1, res2, downsample in self.downs:
            h = res1(h, ctx)
            h = res2(h, ctx)
            skips.append(h)
            h = downsample(h)

        # Bottleneck
        h = self.mid_block1(h, ctx)
        h = self.mid_block2(h, ctx)

        # Decoder — pop skips and concat before each ResBlock pair
        for res1, res2, upsample in self.ups:
            skip = skips.pop()
            h    = torch.cat([h, skip], dim=1)  # skip connection
            h    = res1(h, ctx)
            h    = res2(h, ctx)
            h    = upsample(h)

        return self.final_conv(h)

    # ── Classifier-Free Guidance ──────────────────────────────────────────

    def cfg_forward(self, x, time, model_kwargs):
        """
        CFG forward pass (Ho & Salimans 2022).

        During inference, blend two predictions:
          ε_cond   = model(x_t, t, text_emb)        ← with text
          ε_uncond = model(x_t, t, zeros)            ← without text

          ε_guided = (w + 1) · ε_cond  −  w · ε_uncond

        Higher w → more text-faithful, less diverse.
        w = 0 → ordinary conditional generation.

        Args:
            x           : (B, 3, H, W)
            time        : (B,)
            model_kwargs: must contain "text_emb" (B, 512) and "cfg_scale" (float)
        """
        w        = model_kwargs["cfg_scale"]
        text_emb = model_kwargs["text_emb"]

        eps_cond   = self.forward(x, time, {"text_emb": text_emb})
        eps_uncond = self.forward(x, time, {"text_emb": torch.zeros_like(text_emb).to(text_emb.device)})

        return (w + 1) * eps_cond - w * eps_uncond
