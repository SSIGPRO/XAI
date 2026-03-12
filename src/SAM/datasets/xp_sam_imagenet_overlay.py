import argparse
import itertools
from pathlib import Path
import sys
sys.path.insert(0, (Path.home() / "repos/peepholelib").as_posix())
sys.path.insert(0, (Path.home() / "repos/XAI/src/SAM").as_posix())

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

from configs.imagenet import imagenet_path
from configs.sam import (
    sam2_checkpoint, sam2_model_cfg, amg_kwargs, sam_input_size,
    gdino_checkpoint, gdino_model_cfg, gdino_box_thresh, gdino_text_thresh,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SAM2 on one ImageNet image and save mask overlay."
    )
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--index", type=int, default=0, help="Image index in selected split.")
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help=(
            "Text prompt for Grounded SAM2, e.g. 'man . fish'. "
            "Separate multiple concepts with ' . '. "
            "If omitted, falls back to automatic mask generation."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("src/SAM/datasets/imagenet_sam_overlay.png"),
        help="Output path for overlay image.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.4, help="Mask opacity in [0, 1]."
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=Path("src/SAM/datasets/imagenet_sam_overlay_plot.png"),
        help="Output path for side-by-side visualization plot.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window in addition to saving it.",
    )
    return parser.parse_args()


def get_image_at_index(root: Path, index: int) -> Path:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    it = (p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in exts)
    result = next(itertools.islice(it, index, None), None)
    if result is None:
        raise IndexError(f"index={index} out of range under {root}")
    return result


def overlay_masks(image_rgb: np.ndarray, masks: list, alpha: float, labels: list = None):
    """Blend colored masks onto image_rgb. masks is a list of bool (H,W) arrays."""
    out = image_rgb.astype(np.float32) / 255.0
    cmap = plt.get_cmap("tab20")
    legend_patches = []
    for i, seg in enumerate(masks):
        color = np.array(cmap(i % 20)[:3], dtype=np.float32)
        out[seg] = (1.0 - alpha) * out[seg] + alpha * color
        label = labels[i] if labels else f"mask {i}"
        legend_patches.append(
            plt.matplotlib.patches.Patch(color=color, label=label)
        )
    out = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
    return out, legend_patches


def save_visualization_plot(
    image_np: np.ndarray,
    overlaid: np.ndarray,
    legend_patches: list,
    image_path: Path,
    n_masks: int,
    plot_out: Path,
    show_plot: bool,
    mode: str,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(overlaid)
    axes[1].set_title(f"SAM2 [{mode}]  –  {n_masks} masks")
    axes[1].axis("off")
    if legend_patches:
        axes[1].legend(handles=legend_patches, loc="upper right", fontsize=7, framealpha=0.7)

    fig.suptitle(f"{image_path.name}", fontsize=11)
    plt.tight_layout()

    plot_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_out, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Auto mask mode
# ---------------------------------------------------------------------------

def run_auto(image_sam: np.ndarray, image_np: np.ndarray, device):
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=device)
    mask_gen = SAM2AutomaticMaskGenerator(sam2_model, **amg_kwargs)

    raw = mask_gen.generate(image_sam)
    raw = sorted(raw, key=lambda m: m["stability_score"], reverse=True)

    H, W = image_np.shape[:2]
    masks, labels = [], []
    for i, m in enumerate(raw):
        seg = Image.fromarray(m["segmentation"]).resize((W, H), Image.NEAREST)
        masks.append(np.array(seg, dtype=bool))
        labels.append(f"mask {i}  stab={m['stability_score']:.2f}")

    return masks, labels


# ---------------------------------------------------------------------------
# Grounded SAM2 mode
# ---------------------------------------------------------------------------

def run_grounded(image_np: np.ndarray, text: str, device):
    try:
        from groundingdino.util.inference import load_model, predict
        import groundingdino.datasets.transforms as T
    except ImportError as exc:
        raise ImportError(
            "groundingdino not found. Install with:\n"
            "  pip install groundingdino-py\n"
            "and download the checkpoint."
        ) from exc

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # --- Grounding DINO: image → boxes ---
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    pil_img = Image.fromarray(image_np)
    img_transformed, _ = transform(pil_img, None)

    dino = load_model(gdino_model_cfg, gdino_checkpoint)
    dino = dino.to(device).eval()

    with torch.no_grad():
        boxes_cxcywh, logits, phrases = predict(
            model=dino,
            image=img_transformed,
            caption=text,
            box_threshold=gdino_box_thresh,
            text_threshold=gdino_text_thresh,
            device=device,
        )

    if boxes_cxcywh.numel() == 0:
        print(f"Grounding DINO found no boxes for prompt: '{text}'")
        return [], []

    # convert normalized cxcywh → pixel xyxy
    H, W = image_np.shape[:2]
    cx, cy, bw, bh = boxes_cxcywh.unbind(-1)
    boxes_xyxy = torch.stack([
        (cx - bw / 2) * W,
        (cy - bh / 2) * H,
        (cx + bw / 2) * W,
        (cy + bh / 2) * H,
    ], dim=-1).cpu().numpy()

    # --- SAM2: boxes → masks ---
    sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image_np)

    masks, labels = [], []
    for box, phrase, logit in zip(boxes_xyxy, phrases, logits.tolist()):
        with torch.inference_mode():
            m, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box[None],          # (1, 4)
                multimask_output=False,
            )
        masks.append(m[0].astype(bool))
        labels.append(f"{phrase}  {logit:.2f}")

    return masks, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.index < 0:
        raise IndexError(f"index must be >= 0, got {args.index}")

    split_dir = Path(imagenet_path) / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    image_path = get_image_at_index(split_dir, args.index)
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Image: {image_path.name}  ({image_np.shape[1]}x{image_np.shape[0]})")

    if args.text:
        print(f"Mode: Grounded SAM2  |  Prompt: '{args.text}'")
        masks, labels = run_grounded(image_np, args.text, device)
        mode = f"grounded: {args.text}"
    else:
        print("Mode: Auto mask generation")
        image_sam = np.array(
            image.resize((sam_input_size, sam_input_size), Image.BILINEAR), dtype=np.uint8
        )
        masks, labels = run_auto(image_sam, image_np, device)
        mode = "auto"

    if not masks:
        print("No masks detected.")
        return

    overlaid, legend_patches = overlay_masks(image_np, masks, args.alpha, labels)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlaid).save(args.out)
    save_visualization_plot(
        image_np=image_np,
        overlaid=overlaid,
        legend_patches=legend_patches,
        image_path=image_path,
        n_masks=len(masks),
        plot_out=args.plot_out,
        show_plot=args.show,
        mode=mode,
    )

    print(f"Masks : {len(masks)}")
    print(f"Overlay → {args.out}")
    print(f"Plot   → {args.plot_out}")


if __name__ == "__main__":
    main()
