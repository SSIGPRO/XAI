import sys
from pathlib import Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/autoattacks').as_posix())

import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from peepholelib.datasets.parsedDataset import ParsedDataset
from peepholelib.coreVectors.coreVectors import CoreVectors

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', choices=['WRN28', 'WRN70', 'XCiT'], default='WRN28')
parser.add_argument('-p', '--path', default=Path('/srv/newpenny/XAI/generated_data/attacks').as_posix())
parser.add_argument('-n', '--ndims',     type=int, default=100,
                    help='Number of SVD directions to select (default: 100)')
parser.add_argument('--mc-trials', type=int, default=200,
                    help='Number of random subsets of size n (default: 200)')
args, _ = parser.parse_known_args()
n_dims    = args.ndims
mc_trials = args.mc_trials

from configs.common import *

if __name__ == '__main__':
    if args.model == 'WRN28':
        from configs.models.wrn28_10 import *
    elif args.model == 'WRN70':
        from configs.models.wrn70_16 import *
    elif args.model == 'XCiT':
        from configs.models.xcit_l12 import *
    else:
        raise RuntimeError("Select a model <WRN28|WRN70|XCiT>")

    version    = 'standard'
    model_name = f'{args.model}-{version}'
    base_path  = Path(args.path)
    ds_path    = base_path / 'datasets' / dataset_name
    cvs_path   = base_path / f'{dataset_name}_{model_name}' / 'corevectors'
    svd_path   = base_path / f'{dataset_name}_{model_name}' / 'dim_reduction' / 'svd'
    verbose    = True

    if args.model == 'XCiT':
        target_layers = cfg[version]['layers']
    else:
        target_layers = cfg[version]['layers']['linear']

    atk_types = ['APGD-ce', 'APGD-t', 'FAB-t', 'Square']
    atk_keys  = [f'{dataset_name}-test-{a}-{model_name}' for a in atk_types]
    clean_key = f'{dataset_name}-test-{model_name}'

    inference_names = {
        f'{dataset_name}-test': [model_name] + [f'{a}-{model_name}' for a in atk_types],
    }

    dataset  = ParsedDataset(path=ds_path)
    corevecs = CoreVectors(path=cvs_path, name='corevectors.svd')

    n_layers  = len(target_layers)
    n_attacks = len(atk_types)

    fig, axes = plt.subplots(
        n_layers, n_attacks,
        figsize=(5 * n_attacks, 4 * n_layers),
        squeeze=False,
    )

    rng = np.random.default_rng(42)

    def compute_auc(e_clean, e_atk):
        y_true  = np.concatenate([np.zeros(len(e_clean)), np.ones(len(e_atk))])
        y_score = np.concatenate([e_clean, e_atk])
        if len(np.unique(y_true)) < 2:
            return float('nan')
        return roc_auc_score(y_true, y_score)

    with dataset as ds, corevecs as cv:
        ds.load_only(
            loaders=list(inference_names.keys()),
            inference_names=inference_names,
            verbose=verbose,
        )
        cv.load_only(
            loaders=list(ds._dss.keys()),
            names={layer: None for layer in target_layers},
            verbose=verbose,
        )

        clean_result = ds._dss[clean_key]['result'].bool()

        for row, layer in enumerate(target_layers):
            print(f'[{row+1}/{n_layers}] {layer}')
            svd_data = torch.load(svd_path / layer, weights_only=True)
            s = svd_data['s'].cpu().numpy()
            q = len(s)

            if n_dims > q:
                print(f'  WARNING: n_dims={n_dims} > q={q}, skipping layer')
                continue

            clean_sq = cv._corevds[clean_key][layer].pow(2).cpu().numpy()  # [N, q]

            for col, (atk_name, atk_key) in enumerate(zip(atk_types, atk_keys)):
                ax = axes[row, col]

                atk_result   = ds._dss[atk_key]['result'].bool()
                success_mask = (clean_result & ~atk_result).numpy()

                if success_mask.sum() < 2:
                    ax.text(0.5, 0.5, 'no attack successes', transform=ax.transAxes,
                            ha='center', va='center', fontsize=7)
                    ax.set_title(f'{atk_name}  |  {layer}', fontsize=6)
                    continue

                atk_sq = cv._corevds[atk_key][layer].pow(2).cpu().numpy()

                # top-n: first n_dims directions (highest SVs)
                top_idx = np.arange(n_dims)
                top_x   = s[top_idx].sum()
                top_auc = compute_auc(
                    clean_sq[success_mask][:, top_idx].sum(-1),
                    atk_sq[success_mask][:, top_idx].sum(-1),
                )

                # bottom-n: last n_dims directions (lowest SVs)
                bot_idx = np.arange(q - n_dims, q)
                bot_x   = s[bot_idx].sum()
                bot_auc = compute_auc(
                    clean_sq[success_mask][:, bot_idx].sum(-1),
                    atk_sq[success_mask][:, bot_idx].sum(-1),
                )

                # Monte Carlo: mc_trials random subsets of size n_dims
                mc_x, mc_auc = [], []
                for _ in range(mc_trials):
                    idx = rng.choice(q, size=n_dims, replace=False)
                    auc = compute_auc(
                        clean_sq[success_mask][:, idx].sum(-1),
                        atk_sq[success_mask][:, idx].sum(-1),
                    )
                    if not np.isnan(auc):
                        mc_x.append(s[idx].sum())
                        mc_auc.append(auc)

                ax.scatter(mc_x, mc_auc, s=10, color='gray', alpha=0.4, zorder=1,
                           label=f'random (n={mc_trials})')
                ax.scatter([top_x], [top_auc], s=80, color='tab:blue', marker='*', zorder=3,
                           label=f'top-{n_dims} (high SV)')
                ax.scatter([bot_x], [bot_auc], s=80, color='tab:red',  marker='*', zorder=3,
                           label=f'bot-{n_dims} (low SV)')
                ax.axhline(0.5, color='black', linewidth=0.7, linestyle='--', alpha=0.5)

                ax.set_title(f'{atk_name}  |  {layer}', fontsize=6)
                ax.set_xlabel('sum of singular values', fontsize=6)
                ax.set_ylabel('AUC', fontsize=6)
                ax.set_ylim(0, 1)
                ax.tick_params(labelsize=5)
                ax.grid(True, alpha=0.3)

                if row == 0 and col == 0:
                    ax.legend(fontsize=5, loc='lower right')

    fig.suptitle(
        f'{args.model} {version} — AUC vs sum of singular values  (n={n_dims} directions)\n'
        f'blue★=top-{n_dims} (highest SVs)  red★=bot-{n_dims} (lowest SVs)  gray=MC random ({mc_trials} trials)',
        fontsize=9,
    )
    plt.tight_layout()
    out_file = f'auc_vs_svsum_{args.model}_{version}_n{n_dims}.png'
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_file}')
