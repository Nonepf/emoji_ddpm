# DDPM Emoji Generator

A minimal implementation of **Denoising Diffusion Probabilistic Models** (Ho et al. 2020),
following the CS231n assignment structure.

All the python files are generated with the help of AI.

---

## 1. File Structure

| File | What it does |
|---|---|
| `gaussian_diffusion.py` | Forward noising, reverse denoising, training loss |
| `unet.py` | UNet with ResNet blocks, time & text conditioning, CFG |
| `trainer.py` | Training loop + checkpointing |
| `sample.py` | Inference (optional CLIP text conditioning) |
| `tests.py` | Unit tests for every component |
| `download_openmoji.py` | download emoji data from openmoji.org |

---

## 2. Core Concepts

### 2.1 Forward Process — `q_sample()`
Add noise to a clean image x₀ in one closed-form step:
```
x_t = sqrt(ᾱ_t) · x_0  +  sqrt(1 - ᾱ_t) · ε,    ε ~ N(0, I)
```
`sqrt(ᾱ_t)` shrinks the signal; `sqrt(1-ᾱ_t)` scales the noise.
At t = T the image becomes nearly pure Gaussian noise.

### 2.2 Reverse Process — `p_sample()`
The UNet denoises one step at a time via the DDPM posterior:
```
μ_θ     = coef1 · x̂_0  +  coef2 · x_t
x_{t-1} = μ_θ  +  sqrt(σ²_t) · z,    z ~ N(0, I)  (z = 0 at t = 0)
```

### 2.3 Training Loss — `p_losses()`
MSE between predicted and true noise (or x₀ — equivalent):
```
L = E[|| ε − ε_θ(sqrt(ᾱ_t)·x_0 + sqrt(1−ᾱ_t)·ε, t) ||²]
```

### 2.4 UNet — `unet.py`
```
x (3,H,W) → init_conv → [ResBlock, ResBlock, Downsample] × N
                       → mid_block1, mid_block2
                       → [ResBlock(+skip), ResBlock, Upsample] × N
                       → final_conv → (3,H,W)
```
Every ResBlock receives a **context vector** =
`cat(time_embedding, text_embedding)` so the model knows
the timestep and the text prompt at every layer.

### 2.5 Classifier-Free Guidance — `cfg_forward()`
```
ε_guided = (w + 1) · ε_cond  −  w · ε_uncond
```
Higher `w` → more text-faithful; lower → more diverse.

---

## 3. Quick Start

### 3.1 Environment Setup

- **Python Packages**

```bash
# install PyTorch
pip install torch torchvision pillow

# install CLIP (optional)
pip install git+https://github.com/openai/CLIP.git
```

- **Emoji Dataset**

```bash
python download_openmoji.py
```

this command will download approximatly 4000 emoji png files in `./emoji`

### 3.2 Training

```bash
python trainer.py --data_dir ./emojis --save_dir ./checkpoints

# optional parameters
# --image_size: (default: 32)
# --dim: channels of UNet (default: 32)
# --timesteps: diffusion steps, (default: 1000)
# --steps: total training steps, (default: 50000)
# --batch: batch size, (default: 64)
# --lr: learning rate (default: 1e-3)
```

The train function will backup model weights every 2000 steps, and try to generate samples to visualize training effect.

### 3.3 Generate Images

- **unconditional version**:

```bash
python sample.py --checkpoint ./checkpoints/model_0050000.pt --n 16
```

- **conditional version**:

```bash
python sample.py --checkpoint ./checkpoints/model_0050000.pt \
                 --text "smiling face with sunglasses" --cfg_scale 2.0
```
