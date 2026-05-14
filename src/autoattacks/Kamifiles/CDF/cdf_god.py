import numpy as np
import matplotlib.pyplot as plt
from tensordict import PersistentTensorDict
from scipy.stats import gaussian_kde
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
ds_base = "/srv/newpenny/XAI/generated_data/Kami_attacks/datasets/CIFAR100"
ph_base = "/srv/newpenny/XAI/generated_data/Kami_attacks/CIFAR100_WRN28-standard/peepholes/peepholes"

# ── 1. Load test labels ────────────────────────────────────────────────
test_td     = PersistentTensorDict.from_h5(f"{ds_base}/dss.CIFAR100-test")
test_labels = test_td["label"].numpy()
print("Test labels loaded:", test_labels.shape)

# ── 2. Load CDF scores ─────────────────────────────────────────────────
clean_ph   = PersistentTensorDict.from_h5(f"{ph_base}_cdf.svd.CIFAR100-test-WRN28-standard")
apgd_ce_ph = PersistentTensorDict.from_h5(f"{ph_base}_cdf.svd.CIFAR100-test-APGD-ce-WRN28-standard")
apgd_t_ph  = PersistentTensorDict.from_h5(f"{ph_base}_cdf.svd.CIFAR100-test-APGD-t-WRN28-standard")

# ── 3. Get layers safely ───────────────────────────────────────────────
all_keys = list(clean_ph.keys())

# OPTION A: all conv layers only (recommended)
layer_keys = [k for k in all_keys if "conv" in k]

# OPTION B: everything (danger zone)
# layer_keys = all_keys

print(f"Found {len(layer_keys)} layers")

# ── 4. Plot config ──────────────────────────────────────────────────────
n_classes_to_plot = 20
x_grid = np.linspace(0, 1, 300)

output_dir = Path("cdf_layer_plots")
output_dir.mkdir(exist_ok=True)

# ── 5. Loop over layers ────────────────────────────────────────────────
for layer in layer_keys:
    print(f"Processing layer: {layer}")

    try:
        clean_S   = clean_ph[layer].numpy()
        apgd_ce_S = apgd_ce_ph[layer].numpy()
        apgd_t_S  = apgd_t_ph[layer].numpy()
    except KeyError:
        print(f"Skipping missing layer: {layer}")
        continue

    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()

    for c in range(n_classes_to_plot):
        ax = axes[c]
        idx = np.where(test_labels == c)[0]

        for scores, label, color in [
            (clean_S,   "clean",   "steelblue"),
            (apgd_ce_S, "APGD-ce", "tomato"),
            (apgd_t_S,  "APGD-t",  "seagreen"),
        ]:
            vals = scores[idx, c]

            if len(vals) > 1 and np.std(vals) > 1e-6:
                try:
                    kde = gaussian_kde(vals)
                    density = kde(x_grid)
                    ax.plot(x_grid, density, color=color)
                    ax.fill_between(x_grid, density, alpha=0.15, color=color)
                except Exception:
                    pass

        ax.set_title(f"Class {c}")
        ax.set_xlim(0, 1)
        ax.set_xlabel("CDF score")
        ax.set_ylabel("Density")

    plt.suptitle(f"CDF KDEs - {layer}", fontsize=14)
    plt.tight_layout()

    safe_name = layer.replace(".", "_")
    save_path = output_dir / f"cdf_{safe_name}.png"

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

print("Done. All layer plots saved.")