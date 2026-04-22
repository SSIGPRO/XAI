import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/autoattacks').as_posix())

from configs.eval.cluster_eval import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

drillers_std = drillers['standard']
drillers_rob = drillers['robust']

std_layers = list(drillers_std.keys())
rob_layers = list(drillers_rob.keys())

assert len(std_layers) == len(rob_layers), \
    f"Layer count mismatch: standard={len(std_layers)}, robust={len(rob_layers)}"

# Extract NLL per layer
nll_std = [drillers_std[_l]._classifier.nll_ for _l in std_layers]
nll_rob = [drillers_rob[_l]._classifier.nll_ for _l in rob_layers]

x = list(range(len(std_layers)))
layer_labels = [f'[{i}] {l}' for i, l in enumerate(std_layers)]

fig, ax = plt.subplots(figsize=(max(10, len(x) * 0.6), 5))

ax.plot(x, nll_std, marker='o', label='standard', linewidth=1.5)
ax.plot(x, nll_rob, marker='s', label='robust',   linewidth=1.5, linestyle='--')

ax.set_xticks(x)
ax.set_xticklabels(layer_labels, rotation=45, ha='right', fontsize=6)
ax.set_xlabel('Layer', fontsize=9)
ax.set_ylabel('NLL', fontsize=9)
ax.set_title(f'{args.model} — GMM NLL per layer: standard vs robust', fontsize=11)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

out_file = f'nll_per_layer.{args.reducer}.png'
plt.savefig(out_file, dpi=150, bbox_inches='tight')
print(f'Saved to {out_file}')

# ── Empirical posterior matrices ──────────────────────────────────────────────
out_dir = Path(f'empp_plots.{args.reducer}')
out_dir.mkdir(exist_ok=True)

for i, (_ls, _lr) in enumerate(zip(std_layers, rob_layers)):
    empp_std = drillers_std[_ls]._empp.cpu().numpy()  # (nl_class, nl_model)
    empp_rob = drillers_rob[_lr]._empp.cpu().numpy()
    print(empp_std.sum(), empp_rob.sum())  # Should be close to 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, empp, title in zip(axes, [empp_std, empp_rob], ['standard', 'robust']):
        im = ax.imshow(1-empp, cmap='bone',vmax=1, vmin=0, aspect='auto')
        ax.set_title(f'{title}', fontsize=10)
        ax.set_xlabel('Model class', fontsize=8)
        ax.set_ylabel('GMM component', fontsize=8)
        ax.tick_params(labelsize=6)

    fig.suptitle(f'{args.model} — empp  [{i}] {_ls}', fontsize=11)
    plt.tight_layout()

    out_file = out_dir / f'empp_{i:03d}_{_ls}.png'
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_file}')
