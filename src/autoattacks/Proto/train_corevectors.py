import torch
import torch.nn.functional as F
from tensordict import PersistentTensorDict
from pathlib import Path

ds_base = "/srv/newpenny/XAI/generated_data/Kami_attacks/datasets/CIFAR100"
cv_base = "/srv/newpenny/XAI/generated_data/Kami_attacks/CIFAR100_WRN28-robust/corevectors"

train_ds = PersistentTensorDict.from_h5(f"{ds_base}/dss.CIFAR100-train.WRN28-robust")
train_cv = PersistentTensorDict.from_h5(f"{cv_base}/corevectors.svd.CIFAR100-train-WRN28-robust")

layers    = list(train_cv.keys())
n_classes = 100
cv_dim    = 500 
threshold = 0.8

# ── Read ALL predictions/confidence at once ────────────────────────────
print("Loading predictions...")
results     = train_ds['result'][:].bool()                          # (40000,)
preds       = train_ds['pred'][:].long()                            # (40000,)
confidences = torch.softmax(train_ds['output'][:].float(), dim=1).max(dim=1).values  # (40000,)

# ── Filter good samples ────────────────────────────────────────────────
good_mask = results & (confidences > threshold)                     # (40000,) boolean
good_preds = preds[good_mask]                                       # only good samples
print(f"Good samples: {good_mask.sum().item()} / 40000")

# ── Build proto maps layer by layer ───────────────────────────────────
proto_maps = {}

for idx, layer in enumerate(layers):
    print(f"Processing layer {idx+1}/{len(layers)}: {layer}")

    # read entire layer at once — one disk read per layer
    all_vecs = train_cv[layer][:].float()                           # (40000, 500)
    good_vecs = all_vecs[good_mask]                                 # (N_good, 500)

    # average per class using scatter
    cv_dim = good_vecs.shape[1]
    proto = torch.zeros(n_classes, cv_dim)
    counts = torch.zeros(n_classes)

    for c in range(n_classes):
        class_mask = (good_preds == c)
        if class_mask.sum() > 0:
            proto[c] = good_vecs[class_mask].mean(dim=0)
            counts[c] = class_mask.sum()

    proto_maps[layer] = F.normalize(proto, dim=1)                   # (100, 500)
    print(f"  min count: {counts.min().item():.0f} | max count: {counts.max().item():.0f}")

# ── Save ───────────────────────────────────────────────────────────────
save_path = Path(cv_base) / "proto_maps.svd.WRN28-robust.pt"
torch.save(proto_maps, save_path)
print(f"Saved to {save_path}")