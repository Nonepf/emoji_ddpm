"""
trainer.py
==========
Simple training loop for DDPM on a folder of emoji PNG images.

Usage:
    python trainer.py --data_dir ./emojis --save_dir ./checkpoints
"""

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from unet import Unet
from gaussian_diffusion import GaussianDiffusion


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EmojiDataset(Dataset):
    """
    Loads PNG images from a folder tree.
    Returns (B, 3, H, W) tensors in [-1, 1] (normalized for DDPM).

    For text conditioning you would also return a pre-computed CLIP embedding
    per image; here we keep it unconditional for simplicity.
    """

    def __init__(self, folder, image_size=32):
        self.paths = sorted(Path(folder).glob("**/*.png"))
        if not self.paths:
            raise RuntimeError(f"No PNG files found in {folder}")
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),                          # → [0, 1]
            transforms.Normalize([0.5]*3, [0.5]*3),        # → [-1, 1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Convert to RGBA, and then convert to RGB to avoid possible warnings
        img = Image.open(self.paths[idx])
        if img.mode == 'P':
            img = img.convert('RGBA').convert('RGB')
        else:
            img = img.convert('RGB')
        return self.transform(img)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        diffusion,
        dataset,
        device,
        batch_size=64,
        lr=1e-3,
        num_steps=50_000,
        save_dir="./checkpoints",
        sample_every=2_000,
    ):
        self.diffusion    = diffusion
        self.device       = device
        self.num_steps    = num_steps
        self.sample_every = sample_every
        self.save_dir     = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.step         = 0

        self.loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            drop_last=True, num_workers=8, pin_memory=True,
            prefetch_factor=2, persistent_workers=True,
        )
        self.opt = optim.AdamW(diffusion.model.parameters(), lr=lr)

    def train(self):
        self.diffusion.model.to(self.device).train()
        data_iter = self._cycle(self.loader)
        
        with tqdm(total=self.num_steps, initial=self.step, desc="Training") as pbar:
            while self.step < self.num_steps:
                # load data in advance
                imgs = next(data_iter).to(self.device, non_blocking=True)

                loss = self.diffusion.p_losses(imgs)
                
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                
                self.step += 1
                pbar.update(1)
                
                if self.step % 100 == 0:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                if self.step % self.sample_every == 0:
                    self._save_samples()
                    self._save_checkpoint()

        print("Training complete.")

    def _save_samples(self):
        self.diffusion.model.eval()
        with torch.no_grad():
            imgs = self.diffusion.sample(16, device=self.device)
        path = self.save_dir / f"samples_{self.step:07d}.png"
        save_image(imgs, path, nrow=4)
        print(f"  → samples saved to {path}")
        self.diffusion.model.train()

    def _save_checkpoint(self):
        path = self.save_dir / f"model_{self.step:07d}.pt"
        torch.save({
            "step":      self.step,
            "model":     self.diffusion.model.state_dict(),
            "optimizer": self.opt.state_dict(),
        }, path)
        print(f"  → checkpoint saved to {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.diffusion.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["optimizer"])
        self.step = ckpt["step"]
        print(f"Loaded checkpoint — step {self.step}")

    @staticmethod
    def _cycle(loader):
        while True:
            yield from loader


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="./emojis")
    p.add_argument("--save_dir",   default="./checkpoints")
    p.add_argument("--image_size", type=int,   default=32)
    p.add_argument("--dim",        type=int,   default=32,     help="UNet base channels")
    p.add_argument("--timesteps",  type=int,   default=1000)
    p.add_argument("--steps",      type=int,   default=50_000, help="Total training steps")
    p.add_argument("--batch",      type=int,   default=64)
    p.add_argument("--lr",         type=float, default=1e-3)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = Unet(dim=args.dim, condition_dim=64, dim_mults=(1, 2, 4))
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    diffusion = GaussianDiffusion(
        model, image_size=args.image_size,
        timesteps=args.timesteps, beta_schedule="cosine", objective="pred_noise",
    )

    dataset = EmojiDataset(args.data_dir, image_size=args.image_size)
    print(f"Dataset: {len(dataset)} images")

    trainer = Trainer(
        diffusion, dataset, device,
        batch_size=args.batch, lr=args.lr,
        num_steps=args.steps, save_dir=args.save_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
