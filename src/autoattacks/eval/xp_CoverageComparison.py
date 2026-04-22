import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/autoattacks').as_posix())

from configs.eval.cluster_eval import *
from peepholelib.utils.viz_empp import empp_coverage_scores

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

drillers_std = drillers['standard']
drillers_rob = drillers['robust']

std_layers = list(drillers_std.keys())
rob_layers = list(drillers_rob.keys())

assert len(std_layers) == len(rob_layers), \
    f"Layer count mismatch: standard={len(std_layers)}, robust={len(rob_layers)}"

# Load fitted classifiers
for _l in std_layers:
    drillers_std[_l].load()

for _l in rob_layers:
    drillers_rob[_l].load()

# Compute coverage scores
scores_std = empp_coverage_scores(drillers=drillers_std, threshold=0.5)
scores_rob = empp_coverage_scores(drillers=drillers_rob, threshold=0.5)

# Plot
x = list(range(len(std_layers)))
layer_labels = [f'[{i}] {l}' for i, l in enumerate(std_layers)]

std_vals = [scores_std.get(_l, 0) for _l in std_layers]
rob_vals = [scores_rob.get(_l, 0) for _l in rob_layers]

fig, axes = plt.subplots(2, 1, figsize=(max(8, len(x) * 0.6), 10), sharey=True)

for ax, vals, title in zip(axes, [std_vals, rob_vals], ['standard', 'robust']):
    ax.plot(x, vals, marker='o', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, rotation=45, ha='right', fontsize=6)
    ax.set_xlabel('Layer', fontsize=9)
    ax.set_ylabel('Avg Coverage', fontsize=9)
    ax.set_title(f'{title}', fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

fig.suptitle(f'{args.model} — Coverage per layer (threshold=0.9)', fontsize=11)
plt.tight_layout()

out_file = f'coverage_comparison.{args.reducer}.png'
plt.savefig(out_file, dpi=150, bbox_inches='tight')
print(f'Saved to {out_file}')
