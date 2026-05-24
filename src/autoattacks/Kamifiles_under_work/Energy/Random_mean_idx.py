import numpy as np
import matplotlib.pyplot as plt
from tensordict import PersistentTensorDict
from pathlib import Path
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score

# ── Paths ──────────────────────────────────────────────────────────────
ds_base   = "/srv/newpenny/XAI/generated_data/Kami_attacks/datasets/CIFAR100"
cv_base   = "/srv/newpenny/XAI/generated_data/Kami_attacks/CIFAR100_WRN28-standard/corevectors"
plot_base = "/srv/newpenny/XAI/generated_data/Kami_attacks/CIFAR100_WRN28-standard/random_directions"
Path(plot_base).mkdir(parents=True, exist_ok=True)

# ── Load & filter ──────────────────────────────────────────────────────
print("Loading test dataset...")
test_ds   = PersistentTensorDict.from_h5(f"{ds_base}/dss.CIFAR100-test.WRN28-standard")
good_mask = test_ds['result'][:].bool()
print(f"Correctly classified: {good_mask.sum().item()} / 10000")

print("Loading corevectors...")
clean_cv   = PersistentTensorDict.from_h5(f"{cv_base}/corevectors.svd.CIFAR100-test-WRN28-standard")
apgd_ce_cv = PersistentTensorDict.from_h5(f"{cv_base}/corevectors.svd.CIFAR100-test-APGD-ce-WRN28-standard")
apgd_t_cv  = PersistentTensorDict.from_h5(f"{cv_base}/corevectors.svd.CIFAR100-test-APGD-t-WRN28-standard")

layers  = list(clean_cv.keys())
n_dirs  = 100
n_runs  = 30
rng     = np.random.default_rng(42)

def energy(mat, indices):
    return (mat[:, indices] ** 2).sum(axis=1)

def auc_score(c_e, adv_e):
    labels = np.concatenate([np.zeros(len(c_e)), np.ones(len(adv_e))])
    scores = np.concatenate([c_e, adv_e])
    return roc_auc_score(labels, scores)

def plot_kde(ax, c_e, ce_e, t_e, auc_ce, auc_t, title):
    """Draw KDE for clean / apgd-ce / apgd-t on a given axis."""
    x_min  = min(c_e.min(), ce_e.min(), t_e.min())
    x_max  = max(c_e.max(), ce_e.max(), t_e.max())
    x_grid = np.linspace(x_min, x_max, 300)
    for arr, label, color in [
        (c_e,  'clean',                          'steelblue'),
        (ce_e, f'apgd-ce (AUC={auc_ce:.2f})',   'tomato'),
        (t_e,  f'apgd-t  (AUC={auc_t:.2f})',    'seagreen'),
    ]:
        if np.std(arr) > 1e-6:
            kde = gaussian_kde(arr)
            ax.plot(x_grid, kde(x_grid), label=label, color=color)
            ax.fill_between(x_grid, kde(x_grid), alpha=0.15, color=color)
    ax.set_title(title, fontsize=7)
    ax.set_xlabel('energy', fontsize=6)
    ax.set_ylabel('density', fontsize=6)
    ax.legend(fontsize=5)
    ax.grid(True, alpha=0.3)

# ══════════════════════════════════════════════════════════════════════
# PASS 1 — KDE for tail and head (one big figure, all layers)
# ══════════════════════════════════════════════════════════════════════
print("\n── KDE: tail & head ──")
n_layers = len(layers)
n_cols   = 5
n_rows   = (n_layers + n_cols - 1) // n_cols

for mode in ('tail', 'head'):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    axes = axes.flatten()

    for idx, layer in enumerate(layers):
        print(f"  [{mode}] Layer {idx+1}/{n_layers}: {layer}")
        c  = clean_cv[layer][:].float()[good_mask].numpy()
        ce = apgd_ce_cv[layer][:].float()[good_mask].numpy()
        t  = apgd_t_cv[layer][:].float()[good_mask].numpy()
        D  = c.shape[1]

        sel = np.arange(D - n_dirs, D) if mode == 'tail' else np.arange(n_dirs)

        c_e  = energy(c,  sel)
        ce_e = energy(ce, sel)
        t_e  = energy(t,  sel)
        auc_ce = auc_score(c_e, ce_e)
        auc_t  = auc_score(c_e, t_e)

        plot_kde(axes[idx], c_e, ce_e, t_e, auc_ce, auc_t, layer)

    for idx in range(n_layers, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'Energy KDE — {mode} {n_dirs} directions (WRN28-standard)', fontsize=13)
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    out = f"{plot_base}/kde_{mode}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out}")

# ══════════════════════════════════════════════════════════════════════
# PASS 2 — KDE for each of the 30 random draws
#          One big figure per run (all layers), saved as kde_random_run_{i:02d}.png
# ══════════════════════════════════════════════════════════════════════
print("\n── KDE: 30 random runs ──")

# Pre-draw all random index sets so they're the same across layers
rand_index_sets = [
    np.sort(rng.choice(500, size=n_dirs, replace=False))   # will trim per-layer below
    for _ in range(n_runs)
]

for run_i, rand_idx_full in enumerate(rand_index_sets):
    print(f"  Random run {run_i+1}/{n_runs}")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    axes = axes.flatten()

    for idx, layer in enumerate(layers):
        c  = clean_cv[layer][:].float()[good_mask].numpy()
        ce = apgd_ce_cv[layer][:].float()[good_mask].numpy()
        t  = apgd_t_cv[layer][:].float()[good_mask].numpy()
        D  = c.shape[1]

        # clip indices to valid range for this layer's D
        rand_idx = rand_idx_full[rand_idx_full < D]
        # if we lost too many, top up from valid range
        if len(rand_idx) < n_dirs:
            extra = rng.choice(D, size=n_dirs - len(rand_idx), replace=False)
            rand_idx = np.sort(np.unique(np.concatenate([rand_idx, extra])))

        c_e  = energy(c,  rand_idx)
        ce_e = energy(ce, rand_idx)
        t_e  = energy(t,  rand_idx)
        auc_ce = auc_score(c_e, ce_e)
        auc_t  = auc_score(c_e, t_e)

        mean_idx = rand_idx.mean()
        plot_kde(axes[idx], c_e, ce_e, t_e, auc_ce, auc_t,
                 f"{layer}\n(mean_idx={mean_idx:.0f})")

    for idx in range(n_layers, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'Energy KDE — random run {run_i+1:02d}/{n_runs} (WRN28-standard)', fontsize=13)
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    out = f"{plot_base}/kde_random_run_{run_i+1:02d}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()

print(f"  Saved 30 random KDE figures to {plot_base}/")

# ══════════════════════════════════════════════════════════════════════
# PASS 3 — Scatter: AUC vs mean_idx  (all points collected during KDE passes)
# ══════════════════════════════════════════════════════════════════════
print("\n── Scatter plot ──")

all_results = {layer: [] for layer in layers}

# re-collect all results cleanly (fast, no plotting)
for idx, layer in enumerate(layers):
    c  = clean_cv[layer][:].float()[good_mask].numpy()
    ce = apgd_ce_cv[layer][:].float()[good_mask].numpy()
    t  = apgd_t_cv[layer][:].float()[good_mask].numpy()
    D  = c.shape[1]

    # tail
    sel = np.arange(D - n_dirs, D)
    c_e, ce_e, t_e = energy(c, sel), energy(ce, sel), energy(t, sel)
    all_results[layer].append(dict(
        mean_idx = float(sel.mean()),
        auc_ce   = auc_score(c_e, ce_e),
        auc_t    = auc_score(c_e, t_e),
        kind     = 'tail',
    ))

    # head
    sel = np.arange(n_dirs)
    c_e, ce_e, t_e = energy(c, sel), energy(ce, sel), energy(t, sel)
    all_results[layer].append(dict(
        mean_idx = float(sel.mean()),
        auc_ce   = auc_score(c_e, ce_e),
        auc_t    = auc_score(c_e, t_e),
        kind     = 'head',
    ))

    # randoms
    rng2 = np.random.default_rng(42)   # same seed → same draws as KDE pass
    for _ in range(n_runs):
        rand_idx = np.sort(rng2.choice(D, size=n_dirs, replace=False))
        c_e, ce_e, t_e = energy(c, rand_idx), energy(ce, rand_idx), energy(t, rand_idx)
        all_results[layer].append(dict(
            mean_idx = float(rand_idx.mean()),
            auc_ce   = auc_score(c_e, ce_e),
            auc_t    = auc_score(c_e, t_e),
            kind     = 'random',
        ))

# ── draw scatter ───────────────────────────────────────────────────────
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
axes = axes.flatten()

for idx, layer in enumerate(layers):
    ax   = axes[idx]
    recs = all_results[layer]
    D    = clean_cv[layer][:].shape[1]

    # random cloud
    rand_pts = [r for r in recs if r['kind'] == 'random']
    ax.scatter([p['mean_idx'] for p in rand_pts],
               [p['auc_ce']  for p in rand_pts],
               color='gray',   marker='o', s=25, alpha=0.6, label='random (ce)')
    ax.scatter([p['mean_idx'] for p in rand_pts],
               [p['auc_t']   for p in rand_pts],
               color='silver', marker='s', s=25, alpha=0.6, label='random (t)')

    # trend lines through random cloud
    xs = np.array([p['mean_idx'] for p in rand_pts])
    for ys_key, color, ls in [('auc_ce', 'gray', '--'), ('auc_t', 'silver', ':')]:
        ys = np.array([p[ys_key] for p in rand_pts])
        if len(xs) > 1:
            m, b = np.polyfit(xs, ys, 1)
            x_line = np.linspace(xs.min(), xs.max(), 100)
            ax.plot(x_line, m * x_line + b, color=color, lw=1, ls=ls)

    # tail & head stars on top
    for kind, color in [('tail', 'tomato'), ('head', 'steelblue')]:
        pt = next(r for r in recs if r['kind'] == kind)
        ax.scatter(pt['mean_idx'], pt['auc_ce'],
                   color=color, marker='*', s=220, zorder=6,
                   label=f"{kind} ce={pt['auc_ce']:.2f}")
        ax.scatter(pt['mean_idx'], pt['auc_t'],
                   color=color, marker='^', s=130, zorder=6,
                   label=f"{kind} t={pt['auc_t']:.2f}")

    ax.set_title(layer, fontsize=7)
    ax.set_xlabel(f'mean SVD index  (head≈{n_dirs//2} … tail≈{D - n_dirs//2})', fontsize=6)
    ax.set_ylabel('AUC', fontsize=6)
    ax.set_xlim(-5, D + 5)
    ax.set_ylim(0.45, 1.05)
    ax.axhline(0.5, color='k', lw=0.5, ls='--')
    ax.legend(fontsize=4)
    ax.grid(True, alpha=0.3)

for idx in range(n_layers, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('AUC vs mean SVD index — random 100-direction probes (WRN28-standard)', fontsize=12)
plt.tight_layout()
plt.subplots_adjust(top=0.96)
out = f"{plot_base}/scatter_auc_vs_mean_svd_index.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {out}")
print("\nDone. Outputs:")
print(f"  kde_tail.png")
print(f"  kde_head.png")
print(f"  kde_random_run_01.png … kde_random_run_{n_runs:02d}.png")
print(f"  scatter_auc_vs_mean_svd_index.png")