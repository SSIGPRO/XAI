import numpy as np
import matplotlib.pyplot as plt
import torch
from tensordict import PersistentTensorDict
from pathlib import Path
from sklearn.metrics import roc_auc_score

# ── Paths ──────────────────────────────────────────────────────────────
ds_base   = "/srv/newpenny/XAI/generated_data/Kami_attacks/datasets/CIFAR100"
cv_base   = "/srv/newpenny/XAI/generated_data/Kami_attacks/CIFAR100_WRN28-standard/corevectors"
svd_base  = "/srv/newpenny/XAI/generated_data/Kami_attacks/CIFAR100_WRN28-standard/dim_reduction/svd"
plot_base = "/srv/newpenny/XAI/generated_data/Kami_attacks/CIFAR100_WRN28-standard/random_directions"
Path(plot_base).mkdir(parents=True, exist_ok=True)

# ── Load & filter ──────────────────────────────────────────────────────
print("Loading...")
test_ds    = PersistentTensorDict.from_h5(f"{ds_base}/dss.CIFAR100-test.WRN28-standard")
good_mask  = test_ds['result'][:].bool()
clean_cv   = PersistentTensorDict.from_h5(f"{cv_base}/corevectors.svd.CIFAR100-test-WRN28-standard")
apgd_ce_cv = PersistentTensorDict.from_h5(f"{cv_base}/corevectors.svd.CIFAR100-test-APGD-ce-WRN28-standard")
apgd_t_cv  = PersistentTensorDict.from_h5(f"{cv_base}/corevectors.svd.CIFAR100-test-APGD-t-WRN28-standard")

layers = list(clean_cv.keys())

# ── Load true singular values from SVD files ───────────────────────────
print("Loading singular values...")
sv = {}
for layer in layers:
    svd = torch.load(f"{svd_base}/{layer}", weights_only=False)
    sv[layer] = svd['s'].numpy()

n_dirs = 100
n_runs = 100
rng    = np.random.default_rng(42)

# ── Helpers ────────────────────────────────────────────────────────────
def energy(mat, indices):
    return (mat[:, indices] ** 2).sum(axis=1)

def auc_score(c_e, adv_e):
    labels = np.concatenate([np.zeros(len(c_e)), np.ones(len(adv_e))])
    return roc_auc_score(labels, np.concatenate([c_e, adv_e]))

def sv_sum(layer, indices):
    """Sum of true singular values for selected directions."""
    return float(sv[layer][indices].sum())

n_layers = len(layers)
n_cols   = 5
n_rows   = (n_layers + n_cols - 1) // n_cols

# ══════════════════════════════════════════════════════════════════════
# Scatter: AUC vs sv_sum
# ══════════════════════════════════════════════════════════════════════
print("\n── Scatter plot ──")
all_results = {layer: [] for layer in layers}

for layer in layers:
    c  = clean_cv[layer][:].float()[good_mask].numpy()
    ce = apgd_ce_cv[layer][:].float()[good_mask].numpy()
    t  = apgd_t_cv[layer][:].float()[good_mask].numpy()
    D  = c.shape[1]

    # tail
    sel = np.arange(D - n_dirs, D)
    c_e, ce_e, t_e = energy(c, sel), energy(ce, sel), energy(t, sel)
    all_results[layer].append(dict(
        xval   = sv_sum(layer, sel),
        auc_ce = auc_score(c_e, ce_e),
        auc_t  = auc_score(c_e, t_e),
        kind   = 'tail',
    ))

    # head
    sel = np.arange(n_dirs)
    c_e, ce_e, t_e = energy(c, sel), energy(ce, sel), energy(t, sel)
    all_results[layer].append(dict(
        xval   = sv_sum(layer, sel),
        auc_ce = auc_score(c_e, ce_e),
        auc_t  = auc_score(c_e, t_e),
        kind   = 'head',
    ))

    # randoms
    rng2 = np.random.default_rng(42)
    for _ in range(n_runs):
        rand_idx = np.sort(rng2.choice(D, size=n_dirs, replace=False))
        c_e, ce_e, t_e = energy(c, rand_idx), energy(ce, rand_idx), energy(t, rand_idx)
        all_results[layer].append(dict(
            xval   = sv_sum(layer, rand_idx),
            auc_ce = auc_score(c_e, ce_e),
            auc_t  = auc_score(c_e, t_e),
            kind   = 'random',
        ))

# ── draw ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
axes = axes.flatten()

for idx, layer in enumerate(layers):
    ax   = axes[idx]
    recs = all_results[layer]

    rand_pts = [r for r in recs if r['kind'] == 'random']
    ax.scatter([p['xval']   for p in rand_pts],
               [p['auc_ce'] for p in rand_pts],
               color='gray',   marker='o', s=25, alpha=0.6, label='random (ce)')
    ax.scatter([p['xval']  for p in rand_pts],
               [p['auc_t'] for p in rand_pts],
               color='silver', marker='s', s=25, alpha=0.6, label='random (t)')

    # trend lines
    xs = np.array([p['xval'] for p in rand_pts])
    for ys_key, color, ls in [('auc_ce', 'gray', '--'), ('auc_t', 'silver', ':')]:
        ys = np.array([p[ys_key] for p in rand_pts])
        if len(xs) > 1:
            m, b = np.polyfit(xs, ys, 1)
            x_line = np.linspace(xs.min(), xs.max(), 100)
            ax.plot(x_line, m * x_line + b, color=color, lw=1, ls=ls)

    # tail & head stars
    for kind, color in [('tail', 'tomato'), ('head', 'steelblue')]:
        pt = next(r for r in recs if r['kind'] == kind)
        ax.scatter(pt['xval'], pt['auc_ce'],
                   color=color, marker='*', s=220, zorder=6,
                   label=f"{kind} ce={pt['auc_ce']:.2f}")
        ax.scatter(pt['xval'], pt['auc_t'],
                   color=color, marker='^', s=130, zorder=6,
                   label=f"{kind} t={pt['auc_t']:.2f}")

    ax.set_title(layer, fontsize=7)
    ax.set_xlabel('sum of singular values', fontsize=6)
    ax.set_ylabel('AUC', fontsize=6)
    ax.set_ylim(0.45, 1.05)
    ax.axhline(0.5, color='k', lw=0.5, ls='--')
    ax.legend(fontsize=4)
    ax.grid(True, alpha=0.3)

for i in range(n_layers, len(axes)):
    axes[i].set_visible(False)

plt.suptitle('AUC vs sum of singular values — random probes (WRN28-standard)', fontsize=12)
plt.tight_layout(); plt.subplots_adjust(top=0.96)
out = f"{plot_base}/scatter_auc_vs_svsum.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {out}")
print("\nDone.")