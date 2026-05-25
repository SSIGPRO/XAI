import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/autoattacks').as_posix())

import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from matplotlib.lines import Line2D

from peepholelib.datasets.parsedDataset import ParsedDataset
from peepholelib.coreVectors.coreVectors import CoreVectors

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', choices=['WRN28', 'WRN70'], default='WRN28')
parser.add_argument('-s', '--size', type=int, default=10)
parser.add_argument('-p', '--path', default=Path('/srv/newpenny/XAI/generated_data/attacks').as_posix())
args, _ = parser.parse_known_args()
tail_size = args.size

from configs.common import *

if __name__ == "__main__":

    if args.model == 'WRN28':
        from configs.models.wrn28_10 import *
    elif args.model == 'WRN70':
        from configs.models.wrn70_16 import *
    else:
        raise RuntimeError("Select a model <WRN28|WRN70>")

    base_path = Path(args.path)
    ds_path   = base_path / 'datasets_' / dataset_name
    verbose   = True

    versions = ['standard', ] #'robust'

    dataset = ParsedDataset(
        path=ds_path
        )

    for version in versions:

        model_name = f'{args.model}-{version}'
        cvs_path = base_path / f'{dataset_name}_{model_name}_' / 'corevectors'
        target_layers = cfg[version]['layers']['linear']
        inference_names = {
            f'{dataset_name}-train': [model_name],
            f'{dataset_name}-test':  [model_name, f'APGD-ce-{model_name}', f'APGD-t-{model_name}', f'FAB-t-{model_name}', f'Square-{model_name}'],
            }
        
        corevecs = CoreVectors(
            path = cvs_path,
            name = 'corevectors.svd',
            )

        with dataset as ds, corevecs as cv:
            ds.load_only(
                loaders = list(inference_names.keys()),
                inference_names = inference_names,
                verbose = verbose,
            )

            cv.load_only(
                loaders=list(ds._dss.keys()),
                names={layer: None for layer in target_layers},
                verbose=verbose
            )

            clean_key = f'{dataset_name}-test-{model_name}'
            atk_types = ['APGD-ce', 'APGD-t', 'FAB-t', 'Square']
            atk_keys  = [f'{dataset_name}-test-{a}-{model_name}' for a in atk_types]

            # correctly classified by the clean model
            clean_result = ds._dss[clean_key]['result'].bool()  # [N]

            # tail energy: squared projection onto the last `tail_size` SVD directions (lowest SVs)
            def tail_energy(cv_tensor):
                return cv_tensor[:, -tail_size:].pow(2).sum(dim=-1).cpu()  # [N]

            cs = {'clean (correct)': 'xkcd:cobalt',
                  'attack success':  'xkcd:dark hot pink',
                  'attack fail':     'xkcd:bluish green',
                  'misclassified':   'xkcd:pumpkin orange'}

            n_layers  = len(target_layers)
            n_attacks = len(atk_types)
            n_cols    = n_attacks + 1
            fig, axes = plt.subplots(
                n_layers, n_cols,
                figsize=(6 * n_cols, 4 * n_layers),
                squeeze=False,
            )

            # collect per-attack AUCs across all layers for the suptitle
            all_aucs = {a: [] for a in atk_types}
            clean_aucs = []

            for row, layer in enumerate(target_layers):
                clean_energy = tail_energy(cv._corevds[clean_key][layer])  # [N]

                for col, (atk_name, atk_key) in enumerate(zip(atk_types, atk_keys)):
                    ax = axes[row, col]

                    atk_result   = ds._dss[atk_key]['result'].bool()
                    success_mask = clean_result & ~atk_result   # was right → now wrong
                    fail_mask    = clean_result &  atk_result   # was right → still right

                    atk_energy = tail_energy(cv._corevds[atk_key][layer])

                    groups = {
                        'clean (correct)': clean_energy[success_mask],
                        'attack success':  atk_energy[success_mask],
                        'attack fail':     atk_energy[fail_mask],
                    }

                    df = pd.concat([
                        pd.DataFrame({'energy': e.numpy(), 'group': label})
                        for label, e in groups.items()
                        if len(e) > 1
                    ], ignore_index=True)

                    sb.kdeplot(
                        data=df, ax=ax,
                        x='energy', hue='group',
                        hue_order=[k for k in cs if k in df['group'].unique()],
                        palette=cs,
                        common_norm=False,
                        fill=True, alpha=0.25, linewidth=1.2,
                    )

                    # AUC: can tail energy separate clean-correct from attack-success?
                    e_clean = clean_energy[success_mask].numpy()
                    e_atk   = atk_energy[success_mask].numpy()
                    y_true  = np.concatenate([np.zeros(len(e_clean)), np.ones(len(e_atk))])
                    y_score = np.concatenate([e_clean, e_atk])
                    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else float('nan')
                    all_aucs[atk_name].append(auc)

                    n_success = success_mask.sum().item()
                    n_fail    = fail_mask.sum().item()
                    ax.set_title(
                        f'{atk_name}  |  {layer}  |  AUC={auc:.3f}\n'
                        f'success={n_success}, fail={n_fail}, tail={tail_size} dims',
                        fontsize=8
                    )
                    ax.set_xlim(left=0)
                    ax.set_xlabel(f'Energy in last {tail_size} SVD components', fontsize=8)
                    ax.set_ylabel('Density', fontsize=8)
                    ax.grid(True, alpha=0.3)

                # third column: correct vs incorrect on the clean test set
                ax = axes[row, n_attacks]
                correct_energy   = clean_energy[clean_result]
                incorrect_energy = clean_energy[~clean_result]

                clean_groups = {
                    'clean (correct)': correct_energy,
                    'misclassified':   incorrect_energy,
                }
                df_clean = pd.concat([
                    pd.DataFrame({'energy': e.numpy(), 'group': label})
                    for label, e in clean_groups.items()
                    if len(e) > 1
                ], ignore_index=True)

                sb.kdeplot(
                    data=df_clean, ax=ax,
                    x='energy', hue='group',
                    hue_order=[k for k in cs if k in df_clean['group'].unique()],
                    palette=cs,
                    common_norm=False,
                    fill=True, alpha=0.25, linewidth=1.2,
                )

                e_correct   = correct_energy.numpy()
                e_incorrect = incorrect_energy.numpy()
                y_true_c  = np.concatenate([np.zeros(len(e_correct)), np.ones(len(e_incorrect))])
                y_score_c = np.concatenate([e_correct, e_incorrect])
                auc_c = roc_auc_score(y_true_c, y_score_c) if len(np.unique(y_true_c)) == 2 else float('nan')
                clean_aucs.append(auc_c)

                ax.set_title(
                    f'clean test set  |  {layer}  |  AUC={auc_c:.3f}\n'
                    f'correct={clean_result.sum().item()}, incorrect={(~clean_result).sum().item()}, tail={tail_size} dims',
                    fontsize=8
                )
                ax.set_xlim(left=0)
                ax.set_xlabel(f'Energy in last {tail_size} SVD components', fontsize=8)
                ax.set_ylabel('Density', fontsize=8)
                ax.grid(True, alpha=0.3)

            mean_auc_str = '  |  '.join(
                f'{a}: mean AUC={np.mean(vs):.3f}' for a, vs in all_aucs.items()
            )
            mean_auc_str += f'  |  clean (correct vs incorrect): mean AUC={np.mean(clean_aucs):.3f}'
            fig.suptitle(
                f'{args.model} {version} — tail energy distribution (lower SVs, tail={tail_size})\n{mean_auc_str}',
                fontsize=10,
            )
            plt.tight_layout()
            out_file = f'energy_kde_{args.model}_{version}.png'
            plt.savefig(out_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {out_file}')
