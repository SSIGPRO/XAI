import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/autoattacks').as_posix())

from configs.eval.cluster_eval import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
key = 'standard'
drillers_std = drillers[key]
std_layers = list(drillers_std.keys())

for _l in std_layers:
    drillers_std[_l].load()

out_dir = Path(f'empp_plots_{key}')
out_dir.mkdir(exist_ok=True)

for i, _l in enumerate(std_layers):
    empp_std = drillers_std[_l]._empp.cpu().numpy()  # (nl_class, nl_model)
    print(f'[{i}] {_l}  sum={empp_std.sum():.4f}')

    fig, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(1 - empp_std, cmap='bone', vmax=1, vmin=0, aspect='auto')
    ax.set_xlabel('Model class', fontsize=8)
    ax.set_ylabel('GMM component', fontsize=8)
    ax.tick_params(labelsize=6)

    fig.suptitle(f'{args.model} — empp ({key})  [{i}] {_l}', fontsize=11)
    plt.tight_layout()

    out_file = out_dir / f'empp_{i:03d}_{_l}.png'
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_file}')
