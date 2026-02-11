#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage.restoration import inpaint


# 8-neighbor kernel (exclude center)
KERNEL_8N = np.array(
    [[1, 1, 1],
     [1, 0, 1],
     [1, 1, 1]],
    dtype=float
)


def fill_and_smooth_one(a2d, max_iters=100000, do_inpaint=True):
    """Fill NaNs in one 2D field using neighbor averaging + optional inpainting."""

    a = a2d.astype(float).copy()

    # Record original NaN locations
    nan_initial = np.isnan(a)
    nan_mask = nan_initial.copy()
    iters = 0

    while nan_mask.any() and iters < max_iters:
        iters += 1

        # 1 if known, 0 if NaN
        known = (~nan_mask).astype(float)

        # Replace NaNs by 0 for convolution
        vals = np.nan_to_num(a, nan=0.0)

        # Count known neighbors
        cnt = convolve2d(known, KERNEL_8N, mode="same", boundary="symm")

        # Sum of neighboring values
        s = convolve2d(vals, KERNEL_8N, mode="same", boundary="symm")

        # Only fill NaNs with at least one known neighbor
        fillable = nan_mask & (cnt > 0)
        if not fillable.any():
            break

        a[fillable] = s[fillable] / cnt[fillable]
        nan_mask = np.isnan(a)

    out_filled = a
    remaining = int(np.isnan(out_filled).sum())

    if do_inpaint and nan_initial.any():
        # Smooth only originally-missing regions
        smooth = inpaint.inpaint_biharmonic(out_filled, nan_initial)
        out_smooth = out_filled.copy()
        out_smooth[nan_initial] = smooth[nan_initial]
    else:
        out_smooth = out_filled

    return out_smooth, out_filled, iters, remaining


def run_batch_fill_and_smooth(inp, max_iters=100000, do_inpaint=True):
    """Batch wrapper for (H,W), (N,H,W), or (N,1,H,W) inputs."""

    inp = np.asarray(inp)
    single = False

    if inp.ndim == 2:
        single = True
        inp2 = inp[None, ...]
    elif inp.ndim == 4:
        inp2 = np.squeeze(inp, axis=1)
    elif inp.ndim == 3:
        inp2 = inp
    else:
        raise ValueError(f"Unsupported input shape {inp.shape}")

    N, H, W = inp2.shape
    out_smooth_all = np.empty((N, H, W))
    out_filled_all = np.empty((N, H, W))
    stats = []

    for i in range(N):
        print(f"Processing {i+1}/{N}")
        out_smooth, out_filled, iters, remaining = fill_and_smooth_one(
            inp2[i], max_iters=max_iters, do_inpaint=do_inpaint
        )
        out_smooth_all[i] = out_smooth
        out_filled_all[i] = out_filled
        stats.append(dict(index=i, iters=iters, remaining_nans=remaining))

    if single:
        return out_smooth_all[0], out_filled_all[0], stats[0]

    return out_smooth_all, out_filled_all, stats


def main():
    inte = np.load("input.npy")

    out_smooth_all, out_filled_all, stats = run_batch_fill_and_smooth(
        inte, max_iters=100000, do_inpaint=True
    )

    idx = 1
    vmin, vmax = np.nanmin(inte[idx]), np.nanmax(inte[idx])

    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.15)

    ax0, ax1, ax2 = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    ax0.imshow(inte[idx], vmin=vmin, vmax=vmax)
    ax0.set_title("Input")
    ax0.axis("off")

    ax1.imshow(out_filled_all[idx], vmin=vmin, vmax=vmax)
    ax1.set_title("Filled")
    ax1.axis("off")

    im = ax2.imshow(out_smooth_all[idx], vmin=vmin, vmax=vmax)
    ax2.set_title("Filled + Smoothed")
    ax2.axis("off")

    fig.colorbar(im, cax=cax)
    plt.show()

    # np.save("filled_smoothed.npy", out_smooth_all)


if __name__ == "__main__":
    main()