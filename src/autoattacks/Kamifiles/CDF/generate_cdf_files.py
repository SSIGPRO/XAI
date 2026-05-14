import torch
import numpy as np
from tensordict import PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT
import sys
from pathlib import Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
from peepholelib.utils.cdf_scores import compute_cdf_scores

# ── Paths ──────────────────────────────────────────────────────────────
ds_base = "/srv/newpenny/XAI/generated_data/Kami_attacks/datasets/CIFAR100"
ph_base = "/srv/newpenny/XAI/generated_data/Kami_attacks/CIFAR100_WRN28-standard/peepholes/peepholes"

# ── 1. Load train labels ───────────────────────────────────────────────
train_td     = PersistentTensorDict.from_h5(f"{ds_base}/dss.CIFAR100-train")
train_labels = train_td['label'].numpy()
print("Train labels loaded:", train_labels.shape)

# ── 2. Splits to process ───────────────────────────────────────────────
splits = [
    "CIFAR100-train-WRN28-standard",
    "CIFAR100-val-WRN28-standard",
    "CIFAR100-test-WRN28-standard",
    "CIFAR100-test-APGD-ce-WRN28-standard",
    "CIFAR100-test-APGD-t-WRN28-standard",
]

# ── 3. Load train scores (reference) ──────────────────────────────────
train_ph = PersistentTensorDict.from_h5(f"{ph_base}_cdc.svd.CIFAR100-train-WRN28-standard")
layers   = list(train_ph.keys())
print(f"Found {len(layers)} layers: {layers}")

# ── 4. Process each split ──────────────────────────────────────────────
for split in splits:
    print(f"\nProcessing {split} ...")
    query_ph = PersistentTensorDict.from_h5(f"{ph_base}_cdc.svd.{split}")
    out_path = f"{ph_base}_cdf.svd.{split}"

    N_query  = query_ph[layers[0]].shape[0]
    n_classes = query_ph[layers[0]].shape[1]
    print(f"  N_query={N_query}, n_classes={n_classes}")

    # create output file
    out_td = PersistentTensorDict(filename=out_path, batch_size=[N_query], mode='w')
    for layer in layers:
        out_td[layer] = MMT.empty(shape=(N_query, n_classes), dtype=torch.float32)
    out_td.close()

    # fill output file
    out_td = PersistentTensorDict.from_h5(out_path, mode='r+')
    for layer in layers:
        print(f"  Layer {layer} ...", end=' ', flush=True)
        train_S = train_ph[layer].numpy()
        query_S = query_ph[layer].numpy()

        cdf_out = compute_cdf_scores(
            train_scores  = train_S,
            train_labels  = train_labels,
            query_scores  = query_S,
            n_classes     = n_classes,
        )

        out_td[layer] = torch.tensor(cdf_out)
        print("done")

    out_td.close()
    print(f"  Saved to {out_path}")

print("\nAll done!")