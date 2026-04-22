import numpy as np
import matplotlib.pyplot as plt
from tensordict import PersistentTensorDict
import sys
from pathlib import Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
from peepholelib.utils.cdf_scores import build_typicality_matrix

# ── Paths ──────────────────────────────────────────────────────────────
ds_base = "/srv/newpenny/XAI/generated_data/Kami_attacks/datasets/CIFAR100"
ph_base = "/srv/newpenny/XAI/generated_data/Kami_attacks/CIFAR100_WRN28-robust/peepholes/peepholes"

# ── 1. Load test labels ────────────────────────────────────────────────
test_td     = PersistentTensorDict.from_h5(f"{ds_base}/dss.CIFAR100-test")
test_labels = test_td['label'].numpy()

# ── 2. Load CDF scores ─────────────────────────────────────────────────
clean_ph   = PersistentTensorDict.from_h5(f"{ph_base}_cdf.svd.CIFAR100-test-WRN28-robust")
apgd_ce_ph = PersistentTensorDict.from_h5(f"{ph_base}_cdf.svd.CIFAR100-test-APGD-ce-WRN28-robust")
apgd_t_ph  = PersistentTensorDict.from_h5(f"{ph_base}_cdf.svd.CIFAR100-test-APGD-t-WRN28-robust")
layers     = list(clean_ph.keys())

# ── 3. Pick a sample ───────────────────────────────────────────────────
sample_idx = 2
true_class = test_labels[sample_idx]
print(f"Sample {sample_idx} | true class: {true_class}")

# ── 4. Build typicality matrices ───────────────────────────────────────
clean_mat   = build_typicality_matrix(clean_ph,   layers, sample_idx)
apgd_ce_mat = build_typicality_matrix(apgd_ce_ph, layers, sample_idx)
apgd_t_mat  = build_typicality_matrix(apgd_t_ph,  layers, sample_idx)

# ── 5. Plot ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(32, 10))

for ax, mat, title in [
    (axes[0], clean_mat,   f'Clean   | sample {sample_idx} | true class {true_class}'),
    (axes[1], apgd_ce_mat, f'APGD-ce | sample {sample_idx} | true class {true_class}'),
    (axes[2], apgd_t_mat,  f'APGD-t  | sample {sample_idx} | true class {true_class}'),
]:
    im = ax.imshow(mat, aspect='auto', vmin=0, vmax=1, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('Class')
    ax.set_ylabel('Layer')
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=7)
    ax.axvline(x=true_class, color='red', linewidth=1.5, linestyle='--', label='true class')
    ax.legend(fontsize=8)
    plt.colorbar(im, ax=ax, label='CDF score (typicality)')

plt.suptitle(f'Typicality matrix ({len(layers)} layers x 100 classes) — sample {sample_idx}', fontsize=13)
plt.tight_layout()
plt.savefig(f'typicality_matrix_sample{sample_idx}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved typicality_matrix_sample{sample_idx}.png")