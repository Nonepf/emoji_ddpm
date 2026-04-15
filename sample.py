"""
sample.py
=========
Generate emoji images from a trained DDPM checkpoint.

Usage (unconditional):
    python sample.py --checkpoint ./checkpoints/model_0050000.pt

Usage (text-conditioned, requires CLIP):
    python sample.py --checkpoint ./checkpoints/model_0050000.pt \
                     --text "smiling face with sunglasses" --cfg_scale 2.0
"""

import argparse
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from pathlib import Path

from unet import Unet
from gaussian_diffusion import GaussianDiffusion


def load_model(ckpt_path, device, dim=32, condition_dim=64):
    model = Unet(dim=dim, condition_dim=condition_dim, dim_mults=(1, 2, 4))
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def get_text_emb(text, device):
    """Encode a text prompt with CLIP (512-d)."""
    try:
        import clip
        clip_model, _ = clip.load("ViT-B/32", device=device)
        with torch.no_grad():
            tokens = clip.tokenize([text]).to(device)
            return clip_model.encode_text(tokens).float()
    except ImportError:
        raise SystemExit("Install CLIP:  pip install git+https://github.com/openai/CLIP.git")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--n",           type=int,   default=16,   help="Images to generate")
    p.add_argument("--image_size",  type=int,   default=32)
    p.add_argument("--timesteps",   type=int,   default=1000)
    p.add_argument("--text",        type=str,   default="",   help="Optional text prompt")
    p.add_argument("--cfg_scale",   type=float, default=2.0,  help="CFG guidance scale")
    p.add_argument("--out",         type=str,   default="generated.png")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model     = load_model(args.checkpoint, device).to(device)
    diffusion = GaussianDiffusion(
        model, image_size=args.image_size,
        timesteps=args.timesteps, beta_schedule="cosine", objective="pred_noise",
    )

    model_kwargs = None
    if args.text:
        emb = get_text_emb(args.text, device).expand(args.n, -1)
        model_kwargs = {"text_emb": emb, "cfg_scale": args.cfg_scale}
        print(f"Prompt: '{args.text}'  (cfg_scale={args.cfg_scale})")

    print(f"Generating {args.n} images ...")
    imgs = diffusion.sample(args.n, model_kwargs=model_kwargs, device=device)

    save_image(imgs, args.out, nrow=4)
    print(f"Saved → {args.out}")

    grid = make_grid(imgs, nrow=4).permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(grid)
    plt.axis("off")
    title = f"'{args.text}'" if args.text else "unconditional"
    plt.title(f"DDPM samples — {title}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
