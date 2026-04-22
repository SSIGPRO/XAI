import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/autoattacks').as_posix())

from configs.eval.dim_eval import *

import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

reducers_std = reducers['standard']
reducers_rob = reducers['robust']

std_layers = list(reducers_std.keys())
rob_layers = list(reducers_rob.keys())

assert len(std_layers) == len(rob_layers), \
    f"Layer count mismatch: standard={len(std_layers)}, robust={len(rob_layers)}"

n = len(std_layers)
ncols = 6
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
axes = axes.flatten()

for i, (sl, rl) in enumerate(zip(std_layers, rob_layers)):
    sv_std = reducers_std[sl]._svd['s'].cpu()
    sv_rob = reducers_rob[rl]._svd['s'].cpu()

    ax = axes[i]
    ax.plot(sv_std.numpy(), label='standard', linewidth=1.0)
    ax.plot(sv_rob.numpy(), label='robust',   linewidth=1.0, linestyle='--')
    ax.set_title(f'[{i}] {sl}', fontsize=6)
    ax.set_xlabel('index', fontsize=6)
    ax.set_ylabel('sv', fontsize=6)
    ax.tick_params(labelsize=5)
    ax.legend(fontsize=5)
    ax.grid(True, alpha=0.3)

# hide unused subplots
for j in range(n, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(f'{args.model} — Singular value profiles: standard vs robust', fontsize=11)
plt.tight_layout()

out_file = 'sv_profiles_all_layers.png'
plt.savefig(out_file, dpi=150, bbox_inches='tight')
print(f'Saved to {out_file}')

# --- second plot: normalised singular values (sum to 1) with energy in legend ---
fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
axes2 = axes2.flatten()

for i, (sl, rl) in enumerate(zip(std_layers, rob_layers)):
    sv_std = reducers_std[sl]._svd['s'].cpu()
    sv_rob = reducers_rob[rl]._svd['s'].cpu()

    sv_std_norm = sv_std / sv_std.sum()
    sv_rob_norm = sv_rob / sv_rob.sum()

    # energy = sum of squares of original singular values (proportion explained)
    total_energy_std = (sv_std ** 2).sum().item()
    total_energy_rob = (sv_rob ** 2).sum().item()

    ax = axes2[i]
    ax.plot(sv_std_norm.numpy(),
            label=f'standard (E={total_energy_std:.2e})', linewidth=1.0)
    ax.plot(sv_rob_norm.numpy(),
            label=f'robust (E={total_energy_rob:.2e})',   linewidth=1.0, linestyle='--')
    ax.set_title(f'[{i}] {sl}', fontsize=6)
    ax.set_xlabel('index', fontsize=6)
    ax.set_ylabel('sv (normalised)', fontsize=6)
    ax.tick_params(labelsize=5)
    ax.legend(fontsize=5)
    ax.grid(True, alpha=0.3)

for j in range(n, len(axes2)):
    axes2[j].set_visible(False)

fig2.suptitle(f'{args.model} — Normalised singular value profiles: standard vs robust', fontsize=11)
plt.tight_layout()

out_file2 = 'sv_profiles_normalised_all_layers.png'
plt.savefig(out_file2, dpi=150, bbox_inches='tight')
print(f'Saved to {out_file2}')