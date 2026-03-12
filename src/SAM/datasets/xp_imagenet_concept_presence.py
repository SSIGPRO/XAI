import argparse
import csv
import json
import itertools
from pathlib import Path
import sys

sys.path.insert(0, (Path.home() / "repos/peepholelib").as_posix())
sys.path.insert(0, (Path.home() / "repos/XAI/src/SAM").as_posix())

import numpy as np
from PIL import Image
import torch

from configs.imagenet import imagenet_path
from configs.sam import (
    gdino_checkpoint,
    gdino_model_cfg,
    gdino_box_thresh,
    gdino_text_thresh,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Scan ImageNet images and detect a list of concepts with GroundingDINO. "
            "Outputs per-image, per-concept confidence."
        )
    )
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument(
        "--concepts",
        type=str,
        required=True,
        help="Comma-separated concepts, e.g. 'dog,person,bicycle'",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start image index in the selected split.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=100,
        help="How many images to process from start-index.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Stride over the sorted image list.",
    )
    parser.add_argument(
        "--presence-thresh",
        type=float,
        default=0.35,
        help="Concept is marked present if best confidence >= this threshold.",
    )
    parser.add_argument(
        "--box-thresh",
        type=float,
        default=gdino_box_thresh,
        help="GroundingDINO box threshold.",
    )
    parser.add_argument(
        "--text-thresh",
        type=float,
        default=gdino_text_thresh,
        help="GroundingDINO text threshold.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("src/SAM/datasets/imagenet_concept_presence.csv"),
        help="Output CSV with per-image, per-concept scores.",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=Path("src/SAM/datasets/imagenet_concept_presence_summary.json"),
        help="Output JSON summary over processed images.",
    )
    return parser.parse_args()


def list_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(images)


def resolve_gdino_model_cfg(config_path: str) -> str:
    cfg = Path(config_path)
    if cfg.exists():
        return cfg.as_posix()

    try:
        import groundingdino  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError(
            "groundingdino package not found. Install with: pip install groundingdino-py"
        ) from exc

    pkg_root = Path(groundingdino.__file__).resolve().parent
    fallback = pkg_root / "config" / "GroundingDINO_SwinT_OGC.py"
    if fallback.exists():
        return fallback.as_posix()

    raise FileNotFoundError(
        "GroundingDINO config not found. Checked:\n"
        f"1) {config_path}\n"
        f"2) {fallback}"
    )


def load_grounding_dino(device):
    try:
        from groundingdino.util.inference import load_model, predict
        import groundingdino.datasets.transforms as T
    except ImportError as exc:
        raise ImportError(
            "groundingdino not found. Install with:\n"
            "  pip install groundingdino-py"
        ) from exc

    cfg = resolve_gdino_model_cfg(gdino_model_cfg)
    model = load_model(cfg, gdino_checkpoint)
    model = model.to(device).eval()

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return model, transform, predict


def score_concept(
    model,
    predict_fn,
    image_transformed,
    concept: str,
    device,
    box_thresh: float,
    text_thresh: float,
    presence_thresh: float,
):
    with torch.no_grad():
        boxes, logits, phrases = predict_fn(
            model=model,
            image=image_transformed,
            caption=concept,
            box_threshold=box_thresh,
            text_threshold=text_thresh,
            device=device,
        )

    if boxes.numel() == 0:
        return {
            "present": False,
            "confidence": 0.0,
            "n_boxes": 0,
            "best_phrase": "",
        }

    scores = logits.tolist()
    best_idx = int(np.argmax(scores))
    best_conf = float(scores[best_idx])
    return {
        "present": best_conf >= presence_thresh,
        "confidence": best_conf,
        "n_boxes": int(boxes.shape[0]),
        "best_phrase": str(phrases[best_idx]),
    }


def main():
    args = parse_args()
    concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
    if not concepts:
        raise ValueError("No valid concepts provided. Example: --concepts 'dog,person,bicycle'")

    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if args.max_images <= 0:
        raise ValueError("--max-images must be > 0")
    if args.step <= 0:
        raise ValueError("--step must be > 0")

    split_dir = Path(imagenet_path) / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    image_files = list_images(split_dir)
    if not image_files:
        raise FileNotFoundError(f"No images found under: {split_dir}")

    selected = list(
        itertools.islice(
            image_files,
            args.start_index,
            min(args.start_index + args.max_images * args.step, len(image_files)),
            args.step,
        )
    )
    if not selected:
        raise IndexError(
            f"No images selected with start-index={args.start_index}, "
            f"max-images={args.max_images}, step={args.step}."
        )

    if len(selected) > args.max_images:
        selected = selected[: args.max_images]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Split: {args.split} | Total images in split: {len(image_files)}")
    print(f"Processing: {len(selected)} images | Concepts: {concepts}")

    model, transform, predict_fn = load_grounding_dino(device)

    rows = []
    summary_counts = {c: 0 for c in concepts}
    summary_conf_sum = {c: 0.0 for c in concepts}

    for i, image_path in enumerate(selected, start=1):
        pil_img = Image.open(image_path).convert("RGB")
        img_transformed, _ = transform(pil_img, None)

        for concept in concepts:
            result = score_concept(
                model=model,
                predict_fn=predict_fn,
                image_transformed=img_transformed,
                concept=concept,
                device=device,
                box_thresh=args.box_thresh,
                text_thresh=args.text_thresh,
                presence_thresh=args.presence_thresh,
            )

            rows.append(
                {
                    "image_path": image_path.as_posix(),
                    "image_name": image_path.name,
                    "concept": concept,
                    "present": int(result["present"]),
                    "confidence": result["confidence"],
                    "n_boxes": result["n_boxes"],
                    "best_phrase": result["best_phrase"],
                }
            )
            summary_counts[concept] += int(result["present"])
            summary_conf_sum[concept] += result["confidence"]

        if i % 10 == 0 or i == len(selected):
            print(f"Processed {i}/{len(selected)} images")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "image_name",
                "concept",
                "present",
                "confidence",
                "n_boxes",
                "best_phrase",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "split": args.split,
        "num_images": len(selected),
        "concepts": concepts,
        "presence_threshold": args.presence_thresh,
        "box_threshold": args.box_thresh,
        "text_threshold": args.text_thresh,
        "per_concept": {
            c: {
                "present_count": summary_counts[c],
                "present_ratio": summary_counts[c] / len(selected),
                "mean_confidence": summary_conf_sum[c] / len(selected),
            }
            for c in concepts
        },
    }

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.out_summary.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved CSV     : {args.out_csv}")
    print(f"Saved summary : {args.out_summary}")


if __name__ == "__main__":
    main()
