import sys
from pathlib import Path
sys.path.insert(0, (Path.home() / 'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home() / 'repos/XAI/src/SAM').as_posix())

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from tensordict import PersistentTensorDict

from peepholelib.datasets.parsedDataset import ParsedDataset
from peepholelib.datasets.functional.transforms import means, stds

from configs.common import *
from configs.sam import top_k_masks, concepts_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_display_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    img = img_tensor.detach().cpu()

    if img.dtype == torch.uint8:
        return img.permute(1, 2, 0).numpy()

    img = img.float()
    vmin = float(img.min())
    vmax = float(img.max())

    if 0.0 <= vmin and vmax > 1.0:
        img = img.clamp(0.0, 255.0)
        return img.permute(1, 2, 0).numpy().astype(np.uint8)

    if 0.0 <= vmin and vmax <= 1.0:
        return (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

    mean = means['ImageNet'].squeeze().to(img.dtype)
    std = stds['ImageNet'].squeeze().to(img.dtype)
    img = img * std[:, None, None] + mean[:, None, None]
    img = img.clamp(0.0, 1.0)
    return (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def overlay_masks(ax, img_np, masks, scores, n_valid, alpha=0.4):
    """Draw the image and overlay each valid mask with a distinct color."""
    ax.imshow(img_np)
    cmap = plt.get_cmap('tab10')
    legend_patches = []
    for i in range(n_valid):
        color = np.array(cmap(i % 10)[:3])
        overlay = np.zeros((*img_np.shape[:2], 4), dtype=np.float32)
        mask = masks[i].numpy()
        overlay[mask] = [*color, alpha]
        ax.imshow(overlay)
        legend_patches.append(
            mpatches.Patch(color=color, label=f'mask {i}  score={scores[i]:.3f}')
        )
    ax.legend(handles=legend_patches, loc='upper right', fontsize=6, framealpha=0.7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    loader_key   = 'ImageNet-val'           # e.g. 'ImageNet-train'
    n_images     = 5                     # how many images to visualize
    start_idx    = 0                     # first index to visualize
    save_fig     = True                 # set True to save instead of show
    out_fig_path = Path('concepts_visualization.png')

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    concepts_file = concepts_path / ('concepts.' + loader_key)
    if not concepts_file.exists():
        raise FileNotFoundError(
            f'Concepts file not found: {concepts_file}\n'
            f'Run xp_sam_concepts.py first.'
        )

    dataset = ParsedDataset(path=ds_path)

    with dataset as ds:
        ds.load_only(loaders=[loader_key], verbose=False)
        ptd_img = ds._dss[loader_key]

        ptd_con = PersistentTensorDict.from_h5(concepts_file, mode='r')
        print(ptd_con['masks'].sum())

        n_total  = len(ptd_img)
        indices  = list(range(start_idx, min(start_idx + n_images, n_total)))
        n_show   = len(indices)

        fig, axes = plt.subplots(
            n_show, 2,
            figsize=(10, 4 * n_show),
            squeeze=False,
        )
        fig.suptitle(f'Dataset: {loader_key}  |  SAM2 concepts (top {top_k_masks})', fontsize=12)

        for row, idx in enumerate(indices):
            img_t  = ptd_img['image'][idx]        # (C, H, W)
            label  = int(ptd_img['label'][idx])
            img_np = to_display_uint8(img_t)

            masks   = ptd_con['masks'][idx]        # (top_k, H, W)  bool
            scores  = ptd_con['scores'][idx]       # (top_k,)
            n_valid = int(ptd_con['n_masks'][idx])

            # left: raw image
            axes[row, 0].imshow(img_np)
            axes[row, 0].set_title(f'idx={idx}  label={label}', fontsize=9)
            axes[row, 0].axis('off')

            # right: image + masks
            overlay_masks(axes[row, 1], img_np, masks, scores, n_valid)
            axes[row, 1].set_title(f'{n_valid} masks', fontsize=9)
            axes[row, 1].axis('off')

        ptd_con.close()

    plt.tight_layout()
    if save_fig:
        plt.savefig(out_fig_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {out_fig_path}')
    else:
        plt.show()
