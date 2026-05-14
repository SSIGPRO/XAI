import torch
import torch.nn.functional as F
from tensordict import PersistentTensorDict
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
ds_base  = "/srv/newpenny/XAI/generated_data/Kami_attacks/datasets/CIFAR100"
cv_base  = "/srv/newpenny/XAI/generated_data/Kami_attacks/CIFAR100_WRN28-robust/corevectors"
out_base = "/srv/newpenny/XAI/generated_data/Kami_attacks/CIFAR100_WRN28-robust/protoscores"
Path(out_base).mkdir(parents=True, exist_ok=True)

# ── Load proto maps ────────────────────────────────────────────────────
print("Loading proto maps...")
proto_maps = torch.load(f"{cv_base}/proto_maps.svd.WRN28-robust.pt")
layers = list(proto_maps.keys())

# ── Test sets to process ───────────────────────────────────────────────
test_sets = [
    (
        "CIFAR100-test-WRN28-robust",          # corevector file name
        "CIFAR100-test.WRN28-robust",          # dataset file name
        "clean"                                  # tag
    ),
    (
        "CIFAR100-test-APGD-ce-WRN28-robust",
        "CIFAR100-test.APGD-ce-WRN28-robust",
        "apgd-ce"
    ),
    (
        "CIFAR100-test-APGD-t-WRN28-robust",
        "CIFAR100-test.APGD-t-WRN28-robust",
        "apgd-t"
    ),
]

# ── Process each test set ──────────────────────────────────────────────
for cv_name, ds_name, tag in test_sets:
    print(f"\nProcessing {tag}...")

    test_cv = PersistentTensorDict.from_h5(f"{cv_base}/corevectors.svd.{cv_name}")
    test_ds = PersistentTensorDict.from_h5(f"{ds_base}/dss.{ds_name}")

    # read predictions all at once
    preds = test_ds['pred'][:].long()   # (10000,)
    N     = len(preds)
    print(f"  Samples: {N}")

    protoscores = {}

    for idx, layer in enumerate(layers):
        print(f"  Layer {idx+1}/{len(layers)}: {layer}")

        # read entire layer at once
        all_vecs = test_cv[layer][:].float()            # (10000, 500)
        all_vecs = F.normalize(all_vecs, dim=1)         # L2 normalize

        # for each sample get proto of its predicted class
        proto_for_samples = proto_maps[layer][preds]    # (10000, 500)

        # cosine similarity = dot product (both already normalized)
        scores = (all_vecs * proto_for_samples).sum(dim=1)  # (10000,)
        protoscores[layer] = scores

    # save
    save_path = f"{out_base}/protoscores.{cv_name}.pt"
    torch.save(protoscores, save_path)
    print(f"  Saved to {save_path}")

print("\nDone!")