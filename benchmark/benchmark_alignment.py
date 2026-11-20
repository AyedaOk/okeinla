"""Source-derived registration for the benchmark."""

from dataclasses import dataclass
import math

import cv2
import numpy as np


MAX_RGB_SHIFT = 128
MAX_RGB_ALIGNMENT_ERROR = 0.08
MIN_RGB_ALIGNMENT_OVERLAP = 0.95
ALIGNMENT_PROXY_SIGMA = 2.0


@dataclass(frozen=True)
class RgbAlignment:
    dy: int
    dx: int
    gain: float
    error: float
    overlap: float


def overlap_slices(shape, dy, dx):
    height, width = shape
    y0, y1 = max(0, -dy), min(height, height - dy)
    x0, x1 = max(0, -dx), min(width, width - dx)
    if y1 <= y0 or x1 <= x0:
        raise RuntimeError(f"shift has no overlap: dy={dy}, dx={dx}")
    return ((slice(y0, y1), slice(x0, x1)),
            (slice(y0 + dy, y1 + dy), slice(x0 + dx, x1 + dx)))


def _proxy(rgb):
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise RuntimeError(f"expected CHW RGB, got {rgb.shape}")
    luminance = np.einsum("c,chw->hw", [0.2126, 0.7152, 0.0722], rgb)
    # Low-pass before subsampling so Bayer noise cannot select a one-pixel shift.
    luminance = cv2.GaussianBlur(luminance, (0, 0), ALIGNMENT_PROXY_SIGMA)
    return np.sqrt(np.clip(luminance, 0.0, 1.0)).astype(np.float32)


def _candidate_error(gt_proxy, target_proxy, dy, dx):
    target_slice, gt_slice = overlap_slices(gt_proxy.shape, dy, dx)
    gt = gt_proxy[gt_slice][::4, ::4]
    target = target_proxy[target_slice][::4, ::4]
    mask = (gt > 0.03) & (gt < 0.98) & (target > 0.03) & (target < 0.98)
    if mask.mean() < 0.2:
        return float("inf")
    gt_values = gt[mask].astype(np.float64)
    target_values = target[mask].astype(np.float64)
    gain = np.dot(target_values, gt_values) / (np.dot(target_values, target_values) + 1e-12)
    gain = float(np.clip(gain, 0.25, 4.0))
    return float(np.mean(np.abs(gt_values - np.clip(target_values * gain, 0.0, 1.0))))


def _block_means(values, block_size=16):
    height = values.shape[-2] // block_size * block_size
    width = values.shape[-1] // block_size * block_size
    trimmed = values[..., :height, :width]
    shape = (*trimmed.shape[:-2], height // block_size, block_size, width // block_size, block_size)
    return trimmed.reshape(shape).mean(axis=(-3, -1))


def _exposure_gain(gt, target, target_slice, gt_slice):
    gt_values = _block_means(gt[(slice(None), *gt_slice)])
    target_values = _block_means(target[(slice(None), *target_slice)])
    mask = ((gt_values > 0.01) & (gt_values < 0.90)
            & (target_values > 0.01) & (target_values < 0.90))
    if mask.mean() < 0.1:
        raise RuntimeError("too few valid RGB pixels for exposure matching")
    gain = float(np.median(gt_values[mask] / target_values[mask]))
    if not np.isfinite(gain) or gain < 0.25 or gain > 4.0:
        raise RuntimeError(f"implausible RGB exposure gain: {gain}")
    return gain


def estimate_rgb_alignment(gt, target):
    if gt.shape != target.shape or gt.ndim != 3 or gt.shape[0] != 3:
        raise RuntimeError(f"RGB alignment shape mismatch: GT {gt.shape}, target {target.shape}")
    if not np.isfinite(gt).all() or not np.isfinite(target).all():
        raise RuntimeError("alignment input contains NaN or infinity")
    gt_proxy, target_proxy = _proxy(gt), _proxy(target)
    scale = 0.0625
    size = (max(1, round(gt_proxy.shape[1] * scale)),
            max(1, round(gt_proxy.shape[0] * scale)))
    small_gt = cv2.resize(gt_proxy, size, interpolation=cv2.INTER_AREA)
    small_target = cv2.resize(target_proxy, size, interpolation=cv2.INTER_AREA)
    (phase_dx, phase_dy), _ = cv2.phaseCorrelate(small_target, small_gt)
    coarse = (round(phase_dy / scale), round(phase_dx / scale))
    # Bound the coarse search for small synthetic images as well as full RAWs.
    max_y = min(math.ceil(MAX_RGB_SHIFT * scale), small_gt.shape[0] - 1)
    max_x = min(math.ceil(MAX_RGB_SHIFT * scale), small_gt.shape[1] - 1)
    coarse_grid = min(
        ((_candidate_error(small_gt, small_target, dy, dx), dy, dx)
         for dy in range(-max_y, max_y + 1)
         for dx in range(-max_x, max_x + 1)),
        key=lambda item: item[0],
    )
    seeds = {(0, 0), coarse, (-coarse[0], -coarse[1]),
             (round(coarse_grid[1] / scale), round(coarse_grid[2] / scale))}
    candidates = {
        (dy, dx)
        for seed_y, seed_x in seeds
        if abs(seed_y) <= MAX_RGB_SHIFT and abs(seed_x) <= MAX_RGB_SHIFT
        for dy in range(seed_y - 8, seed_y + 9)
        for dx in range(seed_x - 8, seed_x + 9)
        if abs(dy) <= min(MAX_RGB_SHIFT, gt.shape[1] - 1)
        and abs(dx) <= min(MAX_RGB_SHIFT, gt.shape[2] - 1)
    }
    (dy, dx), error = min(
        ((shift, _candidate_error(gt_proxy, target_proxy, *shift)) for shift in candidates),
        key=lambda item: item[1],
    )
    if not np.isfinite(error):
        raise RuntimeError("RGB alignment objective is not finite")
    target_slice, gt_slice = overlap_slices(gt_proxy.shape, dy, dx)
    pixels = ((target_slice[0].stop - target_slice[0].start)
              * (target_slice[1].stop - target_slice[1].start))
    overlap = pixels / float(gt_proxy.size)
    gain = _exposure_gain(gt, target, target_slice, gt_slice)
    if overlap < MIN_RGB_ALIGNMENT_OVERLAP:
        raise RuntimeError(f"RGB alignment overlap {overlap:.4f} is below threshold")
    if error > MAX_RGB_ALIGNMENT_ERROR:
        raise RuntimeError(f"RGB alignment error {error:.6f} exceeds threshold")
    return RgbAlignment(dy, dx, gain, error, overlap)


def apply_rgb_alignment(gt, candidate, alignment):
    if gt.shape != candidate.shape:
        raise RuntimeError(f"RGB candidate shape mismatch: {gt.shape} versus {candidate.shape}")
    if not np.isfinite(gt).all() or not np.isfinite(candidate).all():
        raise RuntimeError("RGB alignment input contains NaN or infinity")
    target_slice, gt_slice = overlap_slices(gt.shape[-2:], alignment.dy, alignment.dx)
    aligned_gt = np.ascontiguousarray(np.clip(gt[(slice(None), *gt_slice)], 0, 1))
    aligned = np.ascontiguousarray(
        np.clip(candidate[(slice(None), *target_slice)] * alignment.gain, 0, 1))
    return aligned_gt, aligned
