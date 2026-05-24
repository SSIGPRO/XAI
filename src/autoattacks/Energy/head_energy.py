import torch
import numpy as np
import matplotlib.pyplot as plt
from tensordict import PersistentTensorDict
from pathlib import Path
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score

# ── Paths ──────────────────────────────────────────────────────────────
ds_base   = "/srv/newpenny/XAI/generated_data/Kami_attacks/datasets/CIFAR100"
cv_base   = "/srv/newpenny/XAI/generated_data/Kami_attacks/CIFAR100_WRN28-robust/corevectors"
plot_base = "/srv/newpenny/XAI/generated_data/Kami_attacks/CIFAR100_WRN28-robust/tail_energy"
Path(plot_base).mkdir(parents=True, exist_ok=True)

# ── Load dataset for filtering ─────────────────────────────────────────
print("Loading test dataset for filtering...")
test_ds   = PersistentTensorDict.from_h5(f"{ds_base}/dss.CIFAR100-test.WRN28-robust")
good_mask = test_ds['result'][:].bool()
print(f"Correctly classified samples: {good_mask.sum().item()} / 10000")

# ── Load corevectors ───────────────────────────────────────────────────
print("Loading corevectors...")
clean_cv   = PersistentTensorDict.from_h5(f"{cv_base}/corevectors.svd.CIFAR100-test-WRN28-robust")
apgd_ce_cv = PersistentTensorDict.from_h5(f"{cv_base}/corevectors.svd.CIFAR100-test-APGD-ce-WRN28-robust")
apgd_t_cv  = PersistentTensorDict.from_h5(f"{cv_base}/corevectors.svd.CIFAR100-test-APGD-t-WRN28-robust")

layers = list(clean_cv.keys())
head   = 100

# ── Plot KDE for all layers ────────────────────────────────────────────
n_layers = len(layers)
n_cols   = 5
n_rows   = (n_layers + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
axes = axes.flatten()

for idx, layer in enumerate(layers):
    print(f"Layer {idx+1}/{n_layers}: {layer}")
    ax = axes[idx]

    c  = clean_cv[layer][:].float()[good_mask]
    ce = apgd_ce_cv[layer][:].float()[good_mask]
    t  = apgd_t_cv[layer][:].float()[good_mask]

    # head energy — first 100 components
    c_energy  = (c[:,  :head] ** 2).sum(dim=1).numpy()
    ce_energy = (ce[:, :head] ** 2).sum(dim=1).numpy()
    t_energy  = (t[:,  :head] ** 2).sum(dim=1).numpy()

    # compute AUC
    labels_ce = np.concatenate([np.zeros(len(c_energy)), np.ones(len(ce_energy))])
    labels_t  = np.concatenate([np.zeros(len(c_energy)), np.ones(len(t_energy))])
    auc_ce = roc_auc_score(labels_ce, np.concatenate([c_energy, ce_energy]))
    auc_t  = roc_auc_score(labels_t,  np.concatenate([c_energy, t_energy]))

    # plot KDE
    x_min  = min(c_energy.min(), ce_energy.min(), t_energy.min())
    x_max  = max(c_energy.max(), ce_energy.max(), t_energy.max())
    x_grid = np.linspace(x_min, x_max, 300)

    for energy, label, color in [
        (c_energy,  'clean',                        'steelblue'),
        (ce_energy, f'apgd-ce (AUC={auc_ce:.2f})', 'tomato'),
        (t_energy,  f'apgd-t  (AUC={auc_t:.2f})',  'seagreen'),
    ]:
        if np.std(energy) > 1e-6:
            kde = gaussian_kde(energy)
            ax.plot(x_grid, kde(x_grid), label=label, color=color)
            ax.fill_between(x_grid, kde(x_grid), alpha=0.15, color=color)

    ax.set_title(layer, fontsize=7)
    ax.set_xlabel('head energy', fontsize=6)
    ax.set_ylabel('density', fontsize=6)
    ax.legend(fontsize=5)
    ax.grid(True, alpha=0.3)

for idx in range(n_layers, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle(f'Head Energy - WRN28-robust', fontsize=13)
plt.tight_layout()
plt.subplots_adjust(top=0.96)
save_path = f"{plot_base}/head_energy_kde_auc_robust_filtered.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved to {save_path}")