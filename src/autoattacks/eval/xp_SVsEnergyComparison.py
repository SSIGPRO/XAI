import sys
from pathlib import Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/autoattacks').as_posix())

import argparse
import math
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from peepholelib.datasets.parsedDataset import ParsedDataset
from peepholelib.coreVectors.coreVectors import CoreVectors

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', choices=['WRN28', 'WRN70', 'XCiT'], default='WRN28')
parser.add_argument('-p', '--path', default=Path('/srv/newpenny/XAI/generated_data/attacks').as_posix())
args, _ = parser.parse_known_args()

if args.model == 'WRN28':
    from configs.models.wrn28_10 import *
elif args.model == 'WRN70':
    from configs.models.wrn70_16 import *
elif args.model == 'XCiT':
    from configs.models.xcit_l12 import *
else:
    raise RuntimeError("Select a model <WRN28|WRN70|XCiT>")

if __name__ == '__main__':
    version = 'standard'
    model_name = f'{args.model}-{version}'
    base_path  = Path(args.path)
    verbose    = True

    if args.model == 'XCiT':
        target_layers = cfg[version]['layers']
    else:
        target_layers = cfg[version]['layers']['linear']

    ds_path  = base_path / 'datasets' / dataset_name
    cvs_path = base_path / f'{dataset_name}_{model_name}' / 'corevectors'
    svd_path = base_path / f'{dataset_name}_{model_name}' / 'dim_reduction' / 'svd'

    inference_names = {
        f'{dataset_name}-test': [model_name, f'APGD-ce-{model_name}', f'APGD-t-{model_name}'],
    }

    clean_key  = f'{dataset_name}-test-{model_name}'
    apgd_ce_key = f'{dataset_name}-test-APGD-ce-{model_name}'
    apgd_t_key  = f'{dataset_name}-test-APGD-t-{model_name}'

    dataset  = ParsedDataset(path=ds_path)
    corevecs = CoreVectors(path=cvs_path)

    n     = len(target_layers)
    ncols = 6
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
    axes = axes.flatten()

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

        for i, layer in enumerate(target_layers):
            svd_data = torch.load(svd_path / layer, weights_only=True)
            sv = svd_data['s'].cpu()

            clean_energy  = cv._corevds[clean_key][layer].pow(2).mean(dim=0).cpu() 
            apgd_ce_energy = cv._corevds[apgd_ce_key][layer].pow(2).mean(dim=0).cpu() 
            apgd_t_energy  = cv._corevds[apgd_t_key][layer].pow(2).mean(dim=0).cpu() 

            ax  = axes[i]
            ax2 = ax.twinx()

            l1, = ax.plot(sv.numpy(),              color='tab:blue',   linewidth=1.0, label='singular values')
            l2, = ax2.plot(clean_energy.numpy(),   color='tab:orange', linewidth=1.0, linestyle='--', label='clean energy')
            l3, = ax2.plot(apgd_ce_energy.numpy(), color='tab:red',    linewidth=1.0, linestyle=':',  label='APGD-ce energy')
            l4, = ax2.plot(apgd_t_energy.numpy(),  color='tab:green',  linewidth=1.0, linestyle='-.', label='APGD-t energy')

            ax.set_title(f'[{i}] {layer}', fontsize=6)
            ax.set_xlabel('direction index', fontsize=6)
            ax.set_ylabel('singular value', fontsize=6, color='tab:blue')
            ax2.set_ylabel('mean energy', fontsize=6, color='tab:orange')
            ax.tick_params(labelsize=5)
            ax2.tick_params(labelsize=5)
            ax.legend(handles=[l1, l2, l3, l4], fontsize=5, loc='upper right')
            ax.grid(True, alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f'{args.model} {version} — Singular values vs mean energy per SVD direction\n'
        f'blue=SV (left axis), orange=clean, red=APGD-ce, green=APGD-t (right axis)',
        fontsize=10,
    )
    plt.tight_layout()

    out_file = f'sv_vs_energy_{args.model}_{version}.png'
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f'Saved to {out_file}')
