"""Patch-mask helpers for the cDDPM-SparK + MAEDiff experiment."""

import torch
import torch.nn.functional as F


def random_patch_mask(images, patch_size):
    """Sample one valid p x p target patch per image and return its binary mask."""
    batch, _, height, width = images.shape
    patch_h, patch_w = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
    if patch_h > height or patch_w > width:
        raise ValueError(f"patch_size={patch_size} exceeds image shape {(height, width)}")
    top = torch.randint(0, height - patch_h + 1, (batch,), device=images.device)
    left = torch.randint(0, width - patch_w + 1, (batch,), device=images.device)
    mask = torch.zeros((batch, 1, height, width), device=images.device, dtype=images.dtype)
    for index in range(batch):
        # Convert CUDA scalars to Python indices before slicing.
        y0, x0 = top[index].item(), left[index].item()
        mask[index, :, y0:y0 + patch_h, x0:x0 + patch_w] = 1
    return mask


def dilate_mask(mask, margin):
    """Expand the MAE exclusion zone without changing the diffusion target patch."""
    if margin <= 0:
        return mask
    kernel = 2 * margin + 1
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=margin).clamp_max_(1)


def sliding_patch_masks(images, patch_size, stride):
    """Return masks covering an image; append the last location to cover its border."""
    _, _, height, width = images.shape
    patch_h, patch_w = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
    if patch_h > height or patch_w > width:
        raise ValueError(f"patch_size={patch_size} exceeds image shape {(height, width)}")

    def positions(length, patch, step):
        values = list(range(0, length - patch + 1, step))
        if values[-1] != length - patch:
            values.append(length - patch)
        return values

    masks = []
    for top in positions(height, patch_h, stride):
        for left in positions(width, patch_w, stride):
            mask = torch.zeros_like(images[:, :1])
            mask[:, :, top:top + patch_h, left:left + patch_w] = 1
            masks.append(mask)
    return masks
