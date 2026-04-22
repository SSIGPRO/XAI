import numpy as np
import matplotlib.pyplot as plt
from tensordict import PersistentTensorDict
from scipy.stats import gaussian_kde
import sys
from pathlib import Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# ── Paths ──────────────────────────────────────────────────────────────
ds_base = "/srv/newpenny/XAI/generated_data/Kami_attacks/datasets/CIFAR100"
ph_base = "/srv/newpenny/XAI/generated_data/Kami_attacks/CIFAR100_WRN28-robust/peepholes/peepholes"

# ── 1. Load test labels ────────────────────────────────────────────────
test_td     = PersistentTensorDict.from_h5(f"{ds_base}/dss.CIFAR100-test")
test_labels = test_td['label'].numpy()
print("Test labels loaded:", test_labels.shape)

# ── 2. Load CDF scores ─────────────────────────────────────────────────
layer = 'logits' #logits for robust, fc for standard
clean_ph   = PersistentTensorDict.from_h5(f"{ph_base}_cdf.svd.CIFAR100-test-WRN28-robust")
apgd_ce_ph = PersistentTensorDict.from_h5(f"{ph_base}_cdf.svd.CIFAR100-test-APGD-ce-WRN28-robust")
apgd_t_ph  = PersistentTensorDict.from_h5(f"{ph_base}_cdf.svd.CIFAR100-test-APGD-t-WRN28-robust")

clean_S   = clean_ph[layer].numpy()
apgd_ce_S = apgd_ce_ph[layer].numpy()
apgd_t_S  = apgd_t_ph[layer].numpy()

# ── 3. Plot classwise KDE ──────────────────────────────────────────────
n_classes_to_plot = 20
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
axes  = axes.flatten()
x_grid = np.linspace(0, 1, 300)

for c in range(n_classes_to_plot):
    ax  = axes[c]
    idx = np.where(test_labels == c)[0]

    for scores, label, color in [
        (clean_S,   'clean',   'steelblue'),
        (apgd_ce_S, 'APGD-ce', 'tomato'),
        (apgd_t_S,  'APGD-t',  'seagreen'),
    ]:
        vals = scores[idx, c]
        if len(vals) > 1 and np.std(vals) > 1e-6:
            kde = gaussian_kde(vals)
            ax.plot(x_grid, kde(x_grid), label=label, color=color)
            ax.fill_between(x_grid, kde(x_grid), alpha=0.15, color=color)

    ax.set_title(f'Class {c}')
    ax.set_xlim(0, 1)
    ax.set_xlabel('CDF score')
    ax.set_ylabel('Density')
    if c == 0:
        ax.legend(fontsize=8)

plt.suptitle('Class-wise CDF scores (logits layer): clean vs adversarial', fontsize=14)
plt.tight_layout()
plt.savefig('cdf_classwise_logits.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved cdf_classwise_logits.png")