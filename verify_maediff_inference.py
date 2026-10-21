"""Verify actual checkpoint inference, patch coverage and independent averaging.

This is a functional check on two central IXI slices, not an accuracy evaluation.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from benchmark_predict_5fold import reconstruct_one_model
from src.datamodules.benchmark_dataset import BenchmarkPatients
from src.utils.maediff_patching import sliding_patch_masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Run this check in an allocated GPU job.")
    torch.manual_seed(2026)
    np.random.seed(2026)
    checkpoint = Path(args.checkpoint)
    cfg = OmegaConf.load(checkpoint.parent.parent / ".hydra/config.yaml")
    # The complete checkpoint already contains SparK weights; avoid reloading a
    # separate pretrained file (or requiring the original training environment).
    cfg.model.cfg.pretrained_encoder = False
    cfg.model.cfg.encoder_path = None
    model = instantiate(cfg.model, prefix="")
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["state_dict"], strict=True)
    del payload
    model.cuda().eval()
    assert model.maediff_enabled, "Checkpoint is not a MAEDiff experiment."
    patient = BenchmarkPatients(args.data)[0]
    images = patient["image"][24:26].unsqueeze(1).cuda()
    masks = sliding_patch_masks(
        images, model.maediff_patch_size, model.maediff_patch_stride,
    )
    coverage = sum(masks)
    assert torch.all(coverage > 0), "Uncovered pixels."
    assert len(masks) == 16, "Expected 16 windows for 96/48/16 geometry."
    print("Patient:", patient["patient_id"], flush=True)
    print("Windows:", len(masks), "coverage min/max:",
          coverage.min().item(), coverage.max().item(), flush=True)

    # Observe real diffusion calls and independently reconstruct their weighted
    # sum. This detects a skipped sliding path or incorrect overlap normalization.
    observed = []
    def record(module, inputs, output):
        reconstruction = output[1].detach()
        assert reconstruction.shape == images.shape
        assert torch.isfinite(reconstruction).all(), "Non-finite raw prediction."
        observed.append(reconstruction.cpu())

    handle = model.diffusion.register_forward_hook(record)
    try:
        with torch.no_grad():
            output = reconstruct_one_model(model, images)
    finally:
        handle.remove()
    timesteps = (
        list(model.cfg.get("step_ensemble", [250, 500, 750]))
        if model.cfg.get("noise_ensemble", False)
        else [model.test_timesteps]
    )
    assert len(observed) == len(masks) * len(timesteps), (
        f"Unexpected diffusion call count: {len(observed)}"
    )
    masks_cpu = [mask.cpu() for mask in masks]
    independently_averaged = []
    for level in range(len(timesteps)):
        total = torch.zeros_like(images.cpu())
        for window, mask in enumerate(masks_cpu):
            total += observed[level * len(masks) + window] * mask
        independently_averaged.append(total / coverage.cpu())
    expected = torch.stack(independently_averaged).mean(0).clamp(0, 1)
    torch.testing.assert_close(output.cpu(), expected, rtol=1e-4, atol=1e-5)
    assert torch.isfinite(output).all()
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination / "inference_check.npz",
        image=images.cpu().numpy(),
        reconstruction=output.cpu().numpy(),
        residual=(images - output).abs().cpu().numpy(),
        coverage=coverage.cpu().numpy(),
    )
    print(f"PASS: {len(observed)} diffusion calls; finite outputs; "
          "overlap and timestep averaging verified.", flush=True)
    print("Saved:", destination / "inference_check.npz", flush=True)


if __name__ == "__main__":
    main()
