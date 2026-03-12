import sys
from pathlib import Path
sys.path.insert(0, (Path.home() / 'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home() / 'repos/XAI/src/SAM').as_posix())

# python stuff
from math import ceil
from tqdm import tqdm

# numpy
import numpy as np
from PIL import Image as PILImage

# torch stuff
import torch
from torch.utils.data import DataLoader
from cuda_selector import auto_cuda

# tensordict
from tensordict import PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT

# peepholelib
from peepholelib.datasets.parsedDataset import ParsedDataset
from peepholelib.datasets.functional.transforms import means, stds

# configs
from configs.common import *
from configs.sam import (
    sam2_checkpoint, sam2_model_cfg,
    amg_kwargs, top_k_masks, concepts_path, sam_input_size,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_sam_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert an image tensor (C, H, W) to uint8 (H, W, 3) for SAM2.
    Supports three common formats:
      - uint8 in [0, 255]
      - float in [0, 1]
      - ImageNet-normalized float
    """
    img = img_tensor.detach().cpu()

    if img.dtype == torch.uint8:
        return img.permute(1, 2, 0).numpy()

    img = img.float()
    vmin = float(img.min())
    vmax = float(img.max())

    # Already in [0, 255] float
    if 0.0 <= vmin and vmax > 1.0:
        img = img.clamp(0.0, 255.0)
        return img.permute(1, 2, 0).numpy().astype(np.uint8)

    # In [0, 1] float
    if 0.0 <= vmin and vmax <= 1.0:
        return (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

    # Fallback: assume ImageNet-normalized float
    mean = means['ImageNet'].squeeze().to(img.dtype)
    std = stds['ImageNet'].squeeze().to(img.dtype)
    img = img * std[:, None, None] + mean[:, None, None]
    img = img.clamp(0.0, 1.0)
    return (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def masks_to_tensor(raw_masks: list, top_k: int, h: int, w: int, sam_h: int, sam_w: int):
    """
    Convert a list of SAM2 mask-dicts to fixed-size tensors.
    If sam_h/sam_w differ from h/w the masks are downscaled with nearest-neighbour.

    Returns:
        masks  : BoolTensor  (top_k, H, W)
        scores : FloatTensor (top_k,)   – stability_score
        areas  : LongTensor  (top_k,)   – mask area in pixels
        n_valid: int  – number of actual masks (rest are zero-padded)
    """
    # sort by stability_score descending
    raw_masks = sorted(raw_masks, key=lambda m: m['stability_score'], reverse=True)

    masks_out  = torch.zeros(top_k, h, w, dtype=torch.bool)
    scores_out = torch.zeros(top_k, dtype=torch.float32)
    areas_out  = torch.zeros(top_k, dtype=torch.long)

    need_resize = (sam_h != h) or (sam_w != w)

    n_valid = min(len(raw_masks), top_k)
    for i in range(n_valid):
        m = raw_masks[i]
        seg = m['segmentation']   # bool ndarray (sam_h, sam_w)
        if need_resize:
            seg = np.array(
                PILImage.fromarray(seg).resize((w, h), PILImage.NEAREST)
            )
        masks_out[i]  = torch.from_numpy(seg)
        scores_out[i] = float(m['stability_score'])
        areas_out[i]  = int(m['area'])

    return masks_out, scores_out, areas_out, n_valid

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    use_cuda = torch.cuda.is_available()
    device   = torch.device(auto_cuda('memory')) if use_cuda else torch.device('cpu')
    print(f'Using {device}')

    # ------------------------------------------------------------------
    # SAM2 model  (import here so the script fails early if not installed)
    # ------------------------------------------------------------------
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError:
        raise ImportError(
            'sam2 not found. Install it with:\n'
            '  pip install git+https://github.com/facebookresearch/sam2.git\n'
            'and download the checkpoint from:\n'
            '  https://github.com/facebookresearch/sam2?tab=readme-ov-file#model-description'
        )

    sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=device)
    mask_gen   = SAM2AutomaticMaskGenerator(sam2_model, **amg_kwargs)
    print('SAM2 model loaded.')

    # ------------------------------------------------------------------
    # Load parsed ImageNet dataset
    # ------------------------------------------------------------------
    concepts_path.mkdir(parents=True, exist_ok=True)

    dataset = ParsedDataset(path=ds_path)
    loaders = ['ImageNet-val']
    overwrite_existing = True

    with dataset as ds:
        ds.load_only(loaders=loaders, verbose=verbose)

        for loader_key in loaders:
            out_path = concepts_path / ('concepts.' + loader_key)

            ptd_src = ds._dss[loader_key]
            n_samples = len(ptd_src)

            # image spatial dimensions (C, H, W stored in the PTD)
            sample_img = ptd_src['image'][0]   # (C, H, W)
            _, H, W = sample_img.shape

            print(f'\n---- {loader_key}  |  {n_samples} images  |  top_k={top_k_masks}  ({H}x{W})\n')

            # ----------------------------------------------------------
            # Create / open output PTD
            # ----------------------------------------------------------
            if out_path.exists():
                if overwrite_existing:
                    print(f'  Output file {out_path} already exists – overwriting.')
                    out_path.unlink()
                else:
                    print(f'  Output file {out_path} already exists – skipping.')
                    continue

            ptd_dst = PersistentTensorDict(
                filename   = out_path,
                batch_size = [n_samples],
                mode       = 'w',
            )

            # pre-allocate
            ptd_dst['masks']   = MMT.empty(shape=torch.Size([n_samples, top_k_masks, H, W]), dtype=torch.bool)
            ptd_dst['scores']  = MMT.empty(shape=torch.Size([n_samples, top_k_masks]),       dtype=torch.float32)
            ptd_dst['areas']   = MMT.empty(shape=torch.Size([n_samples, top_k_masks]),       dtype=torch.long)
            ptd_dst['n_masks'] = MMT.empty(shape=torch.Size([n_samples]),                    dtype=torch.long)
            ptd_dst['label']   = MMT.empty(shape=torch.Size([n_samples]),                    dtype=ptd_src['label'][0].dtype)

            # close 'w' and reopen 'r+' for multi-worker-safe writes
            ptd_dst.close()
            ptd_dst = PersistentTensorDict.from_h5(out_path, mode='r+')

            # ----------------------------------------------------------
            # Extract concepts image by image
            # ----------------------------------------------------------
            dl = DataLoader(
                ptd_src,
                batch_size  = 1,
                shuffle     = False,
                num_workers = 0,          # SAM2 is already GPU-parallel
                collate_fn  = lambda x: x,
            )

            for idx, batch in enumerate(tqdm(dl, total=n_samples, desc=loader_key)):
                img_t = batch['image'][0]    # (C, H, W)
                label = batch['label'][0]

                # convert to uint8 (H, W, 3) for SAM2
                img_np = to_sam_uint8(img_t)

                # upscale to sam_input_size so SAM2 gets a large enough image
                img_sam = np.array(
                    PILImage.fromarray(img_np).resize(
                        (sam_input_size, sam_input_size), PILImage.BILINEAR
                    )
                )

                # SAM2 inference (no_grad handled internally)
                raw_masks = mask_gen.generate(img_sam)
                if idx < 3:
                    print(
                        f'  debug idx={idx}: in_dtype={img_t.dtype} '
                        f'in_range=({float(img_t.min()):.3f},{float(img_t.max()):.3f}) '
                        f'raw_masks={len(raw_masks)}'
                    )

                masks_t, scores_t, areas_t, n_valid = masks_to_tensor(
                    raw_masks, top_k_masks, H, W, sam_input_size, sam_input_size
                )

                ptd_dst['masks'][idx]   = masks_t
                ptd_dst['scores'][idx]  = scores_t
                ptd_dst['areas'][idx]   = areas_t
                ptd_dst['n_masks'][idx] = n_valid
                ptd_dst['label'][idx]   = label

            ptd_dst.close()
            print(f'  Saved concepts to {out_path}')

    print('\nDone.')
