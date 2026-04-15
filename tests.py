"""
tests.py
========
Unit tests for all DDPM components.
Run with:  python tests.py
"""

import numpy as np
import torch
import torch.nn as nn

from unet import Unet
from gaussian_diffusion import GaussianDiffusion


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-10, np.abs(x) + np.abs(y))))


# ---------------------------------------------------------------------------
# Test 1: q_sample  –  forward noising is self-consistent
# ---------------------------------------------------------------------------

def test_q_sample():
    print("=" * 55)
    print("Test 1: q_sample  (forward noising, self-consistency)")

    torch.manual_seed(0)
    diffusion = GaussianDiffusion(model=None, image_size=4, timesteps=1000, beta_schedule="cosine")
    B = 3
    x0    = torch.randn(B, 3, 4, 4)
    noise = torch.randn(B, 3, 4, 4)
    t     = torch.randint(1, 999, (B,)).long()

    x_t = diffusion.q_sample(x0, t, noise)

    # Recover x0 from x_t and noise — should be exact
    x0_rec = diffusion.predict_start_from_noise(x_t, t, noise)
    err = rel_error(x0_rec.numpy(), x0.numpy())
    ok  = err < 1e-5
    print(f"  Round-trip error (x_t → x_0): {err:.2e}   {'✓ PASS' if ok else '✗ FAIL'}")

    # At t=0, x_t ≈ x_0 (almost no noise added)
    t0   = torch.zeros(B, dtype=torch.long)
    xt0  = diffusion.q_sample(x0, t0, noise)
    err0 = rel_error(xt0.numpy(), x0.numpy())
    ok0  = err0 < 1.0  # cosine schedule adds small but nonzero noise even at t=0
    print(f"  At t=0, noise is minimal (err={err0:.4f}): {'✓ PASS' if ok0 else '✗ FAIL'}")

    # At t=T-1, x_t should differ significantly from x_0
    tT   = torch.full((B,), 999, dtype=torch.long)
    xtT  = diffusion.q_sample(x0, tT, noise)
    err_noisy = rel_error(xtT.numpy(), x0.numpy())
    ok_noisy  = err_noisy > 0.1
    print(f"  At t=999, x_t ≠ x_0 (err={err_noisy:.4f}):  {'✓ PASS' if ok_noisy else '✗ FAIL'}")

    return ok and ok0 and ok_noisy


# ---------------------------------------------------------------------------
# Test 2: predict_noise_from_start / predict_start_from_noise
# ---------------------------------------------------------------------------

def test_predict_conversions():
    print("=" * 55)
    print("Test 2: predict_noise_from_start / predict_start_from_noise")

    diffusion = GaussianDiffusion(model=None, image_size=4, timesteps=1000, beta_schedule="sigmoid")
    B  = 4
    x0 = torch.randn(B, 3, 4, 4)
    eps = torch.randn(B, 3, 4, 4)
    t  = torch.randint(1, 999, (B,)).long()

    x_t = diffusion.q_sample(x0, t, eps)

    pred_eps = diffusion.predict_noise_from_start(x_t, t, x0)
    pred_x0  = diffusion.predict_start_from_noise(x_t, t, eps)

    err_eps = rel_error(pred_eps.numpy(), eps.numpy())
    err_x0  = rel_error(pred_x0.numpy(), x0.numpy())

    ok_eps = err_eps < 1e-5
    ok_x0  = err_x0  < 1e-4
    print(f"  Noise recovery error   : {err_eps:.2e}   {'✓ PASS' if ok_eps else '✗ FAIL'}")
    print(f"  x_start recovery error : {err_x0:.2e}   {'✓ PASS' if ok_x0 else '✗ FAIL'}")
    return ok_eps and ok_x0


# ---------------------------------------------------------------------------
# Test 3: UNet forward pass – shape and basic sanity
# ---------------------------------------------------------------------------

def test_unet_forward():
    print("=" * 55)
    print("Test 3: UNet forward pass (shape check)")

    torch.manual_seed(0)
    unet = Unet(dim=8, condition_dim=8, dim_mults=(1, 2))
    B, H = 2, 8
    x    = torch.randn(B, 3, H, H)
    t    = torch.randint(0, 1000, (B,)).long()

    # No text conditioning
    out = unet(x, t)
    ok_shape = out.shape == (B, 3, H, H)
    print(f"  Output shape {tuple(out.shape)} == (2,3,8,8): {'✓ PASS' if ok_shape else '✗ FAIL'}")

    # With dummy text embedding (512-d)
    kwargs = {"text_emb": torch.randn(B, 512)}
    out_txt = unet(x, t, kwargs)
    ok_txt  = out_txt.shape == (B, 3, H, H)
    print(f"  With text_emb shape   {tuple(out_txt.shape)}: {'✓ PASS' if ok_txt else '✗ FAIL'}")

    # Deterministic given same seed
    torch.manual_seed(42)
    a = unet(x, t)
    torch.manual_seed(42)
    b_ = unet(x, t)
    ok_det = torch.allclose(a, b_)
    print(f"  Deterministic outputs: {'✓ PASS' if ok_det else '✗ FAIL'}")

    return ok_shape and ok_txt and ok_det


# ---------------------------------------------------------------------------
# Test 4: p_losses – training loss is positive and gradients flow
# ---------------------------------------------------------------------------

def test_p_losses():
    print("=" * 55)
    print("Test 4: p_losses  (training loss)")

    torch.manual_seed(0)
    unet = Unet(dim=8, condition_dim=8, dim_mults=(1, 2))
    diffusion = GaussianDiffusion(model=unet, image_size=8, timesteps=100,
                                  beta_schedule="cosine", objective="pred_noise")
    B  = 3
    x0 = torch.randn(B, 3, 8, 8)

    loss = diffusion.p_losses(x0)
    ok_positive = loss.item() > 0
    print(f"  Loss value  : {loss.item():.4f}   {'✓ PASS (>0)' if ok_positive else '✗ FAIL'}")

    # Check gradients flow to model parameters
    loss.backward()
    has_grads = any(p.grad is not None for p in unet.parameters())
    print(f"  Gradients   : {'✓ PASS (flowing)' if has_grads else '✗ FAIL (None)'}")

    # pred_x_start objective also works
    diff2 = GaussianDiffusion(model=unet, image_size=8, timesteps=100,
                               beta_schedule="cosine", objective="pred_x_start")
    loss2 = diff2.p_losses(x0)
    ok2   = loss2.item() > 0
    print(f"  pred_x_start: {loss2.item():.4f}   {'✓ PASS (>0)' if ok2 else '✗ FAIL'}")

    return ok_positive and has_grads and ok2


# ---------------------------------------------------------------------------
# Test 5: p_sample – reverse step produces correct shape and stays finite
# ---------------------------------------------------------------------------

def test_p_sample():
    print("=" * 55)
    print("Test 5: p_sample  (one reverse step)")

    torch.manual_seed(0)
    unet = Unet(dim=8, condition_dim=8, dim_mults=(1, 2))
    diffusion = GaussianDiffusion(model=unet, image_size=8, timesteps=100,
                                  beta_schedule="cosine", objective="pred_noise")
    B   = 2
    x_t = torch.randn(B, 3, 8, 8)

    out = diffusion.p_sample(x_t, t_index=50)
    ok_shape  = out.shape == (B, 3, 8, 8)
    ok_finite = torch.isfinite(out).all().item()
    print(f"  Output shape  {tuple(out.shape)}: {'✓ PASS' if ok_shape else '✗ FAIL'}")
    print(f"  All finite    : {'✓ PASS' if ok_finite else '✗ FAIL'}")

    # At t=0 there should be no stochastic noise (deterministic)
    torch.manual_seed(7)
    a = diffusion.p_sample(x_t, t_index=0)
    torch.manual_seed(99)
    b_ = diffusion.p_sample(x_t, t_index=0)
    ok_det = torch.allclose(a, b_)
    print(f"  t=0 is deterministic: {'✓ PASS' if ok_det else '✗ FAIL'}")

    return ok_shape and ok_finite and ok_det


# ---------------------------------------------------------------------------
# Test 6: Classifier-Free Guidance
# ---------------------------------------------------------------------------

def test_cfg():
    print("=" * 55)
    print("Test 6: Classifier-Free Guidance (cfg_forward)")

    torch.manual_seed(0)
    unet = Unet(dim=8, condition_dim=8, dim_mults=(1, 2))
    B, H = 2, 8
    x    = torch.randn(B, 3, H, H)
    t    = torch.randint(0, 100, (B,)).long()
    txt  = torch.randn(B, 512)

    out_plain = unet(x, t, {"text_emb": txt})
    out_cfg   = unet(x, t, {"text_emb": txt, "cfg_scale": 2.0})
    out_cfg0  = unet(x, t, {"text_emb": txt, "cfg_scale": 0.0})

    ok_shape  = out_cfg.shape == (B, 3, H, H)
    ok_differs = not torch.allclose(out_cfg, out_plain)
    # cfg_scale=0 → same as plain conditional (w=0 ⇒ (0+1)*cond - 0*uncond = cond)
    ok_w0 = torch.allclose(out_cfg0, out_plain, atol=1e-5)

    print(f"  Output shape {tuple(out_cfg.shape)}: {'✓ PASS' if ok_shape else '✗ FAIL'}")
    print(f"  CFG≠plain (w=2.0)   : {'✓ PASS' if ok_differs else '✗ FAIL'}")
    print(f"  CFG==plain (w=0.0)  : {'✓ PASS' if ok_w0 else '✗ FAIL'}")

    return ok_shape and ok_differs and ok_w0


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = [
        test_q_sample(),
        test_predict_conversions(),
        test_unet_forward(),
        test_p_losses(),
        test_p_sample(),
        test_cfg(),
    ]
    print("=" * 55)
    passed = sum(results)
    print(f"\n{passed}/{len(results)} tests passed")
    if passed == len(results):
        print("All tests passed ✓")
    else:
        print("Some tests failed — see above.")
