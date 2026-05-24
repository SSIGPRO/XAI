import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
out_base  = "/srv/newpenny/XAI/generated_data/Kami_attacks/CIFAR100_WRN28-standard/protoscores"
plot_base = "/srv/newpenny/XAI/generated_data/Kami_attacks/CIFAR100_WRN28-standard/protoscores/plots"
Path(plot_base).mkdir(parents=True, exist_ok=True)

# ── Load scores ────────────────────────────────────────────────────────
clean   = torch.load(f"{out_base}/protoscores.CIFAR100-test-WRN28-standard.pt")
apgd_ce = torch.load(f"{out_base}/protoscores.CIFAR100-test-APGD-ce-WRN28-standard.pt")
apgd_t  = torch.load(f"{out_base}/protoscores.CIFAR100-test-APGD-t-WRN28-standard.pt")

layers = list(clean.keys())

# ── Plot CDF for all layers ────────────────────────────────────────────
n_layers = len(layers)
n_cols   = 5
n_rows   = (n_layers + n_cols - 1) // n_cols  # ceil division

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
axes = axes.flatten()

for idx, layer in enumerate(layers):
    ax = axes[idx]

    c  = clean[layer].numpy()
    ce = apgd_ce[layer].numpy()
    t  = apgd_t[layer].numpy()

    # build CDF for each
    for scores, label, color in [
        (c,  'clean',   'blue'),
        (ce, 'apgd-ce', 'orange'),
        (t,  'apgd-t',  'red'),
    ]:
        sorted_scores = np.sort(scores)
        cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        ax.plot(sorted_scores, cdf, label=label, color=color)

    ax.set_title(layer, fontsize=7)
    ax.set_xlabel('protoscore', fontsize=6)
    ax.set_ylabel('CDF', fontsize=6)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

# hide unused subplots
for idx in range(n_layers, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Protoscore - WRN28-standard', fontsize=13)
plt.tight_layout()
plt.subplots_adjust(top=0.96)
save_path = f"{plot_base}/protoscore - standard.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved to {save_path}")