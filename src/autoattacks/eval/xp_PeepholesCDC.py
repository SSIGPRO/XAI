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
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.scores.protoclass import conceptogram_protoclass_score
from peepholelib.plots.conceptograms import plot_conceptogram

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model',   choices=['WRN28', 'WRN70'], default='WRN28')
parser.add_argument('-p', '--path',    default=Path('/srv/newpenny/XAI/generated_data/attacks').as_posix())
parser.add_argument('--proto_th',      type=float, default=40)
args, _ = parser.parse_known_args()

if __name__ == "__main__":

    if args.model == 'WRN28':
        from configs.models.wrn28_10 import *
    elif args.model == 'WRN70':
        from configs.models.wrn70_16 import *
    else:
        raise RuntimeError("Select a model <WRN28|WRN70>")

    version = 'standard' #['standard', 'robust']
    reduction = 'svd' # ['random', 

    base_path = Path(args.path)
    ds_path = base_path / 'datasets' / dataset_name
    phs_path = base_path / f'{dataset_name}_{args.model}-{version}' / 'peepholes'
    phs_name = f'peepholes_cdc.{reduction}'
    verbose   = True

    model_name = f'{args.model}-{version}'

    inference_names = {
            f'{dataset_name}-train': [model_name],
            f'{dataset_name}-test':  [
                model_name, 
                f'APGD-ce-{model_name}', 
                f'APGD-t-{model_name}'],
        }

    # /home/lorenzocapelli/newpenny/XAI/generated_data/attacks/CIFAR100_WRN28-standard/peepholes
    dataset = ParsedDataset(
        path=ds_path
        )
    
    with dataset as ds:
            ds.load_only(
                loaders = list(inference_names.keys()),
                inference_names = inference_names,
                verbose = verbose,
            )

            peepholes = Peepholes(
                path=phs_path, 
                name=phs_name, 
                device=device
                )

            with peepholes as ph:

                ph.load_only(
                    loaders=list(ds._dss.keys()),
                    verbose=verbose
                    )

                train_key = f'{dataset_name}-train-{model_name}'
                test_key  = [
                    f'{dataset_name}-test-{model_name}',
                    f'{dataset_name}-test-APGD-ce-{model_name}',
                    f'{dataset_name}-test-APGD-t-{model_name}',
                ]
                target_modules = list(ph._phs[train_key].keys())

                # Build norm_ref: for each layer sort the training peepholes once
                norm_ref = {
                    layer: torch.sort(ph._phs[train_key][layer], dim=0).values
                    for layer in target_modules
                }

                # plot_conceptogram(
                #     path             = Path.cwd(),
                #     name             = 'conceptogram',
                #     datasets         = ds,
                #     peepholes        = ph,
                #     loaders          = test_key,
                #     samples          = list(range(5)),
                #     target_modules   = target_modules,
                #     ticks            = target_modules,
                #     classes          = Cifar100.get_classes(meta_path=Path(cifar_path)/'cifar-100-python/meta'),
                #     norm_ref         = norm_ref,
                #     verbose          = verbose,
                # )

                ret, proto = conceptogram_protoclass_score(
                        datasets = ds,
                        peepholes = ph,
                        loaders = list(ds._dss.keys()),
                        target_modules = target_modules,
                        proto_key = f'{dataset_name}-train-{model_name}',
                        proto_threshold = 0.9, 
                        verbose = verbose,
                    )

                clean_key = f'{dataset_name}-test-{model_name}'
                atk_keys  = [
                    f'{dataset_name}-test-APGD-ce-{model_name}',
                    f'{dataset_name}-test-APGD-t-{model_name}',
                ]
                score_name = 'Proto-Class'

                clean_scores = ret[clean_key][score_name].cpu().numpy()

                fig, axes = plt.subplots(len(atk_keys), 2, figsize=(12, 5 * len(atk_keys)))

                for i, atk_key in enumerate(atk_keys):
                    # successful attacks: model fooled (result == 0)
                    success_mask = (ds._dss[atk_key]['result'] == 0).cpu().numpy()

                    atk_scores  = ret[atk_key][score_name].cpu().numpy()[success_mask]
                    orig_scores = clean_scores[success_mask]

                    scores_all = np.concatenate([orig_scores, atk_scores])
                    labels_all = np.concatenate([np.zeros(len(orig_scores)), np.ones(len(atk_scores))])
                    auc = roc_auc_score(labels_all, -scores_all)  # lower score → more anomalous
                    print(f"{atk_key}: {success_mask.sum()} successful attacks, AUC={auc:.4f}")

                    # distribution plot
                    ax = axes[i, 0]
                    ax.hist(orig_scores, bins=50, density=True, alpha=0.6, label='clean', color='steelblue')
                    ax.hist(atk_scores,  bins=50, density=True, alpha=0.6, label='attack', color='tomato')
                    ax.set_title(f'{atk_key.split("-")[3]} — score distribution (successful attacks)')
                    ax.set_xlabel(score_name)
                    ax.legend()

                    # ROC curve
                    fpr, tpr, _ = roc_curve(labels_all, -scores_all)
                    ax = axes[i, 1]
                    ax.plot(fpr, tpr, color='darkorange', lw=1.5, label=f'AUC={auc:.3f}')
                    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
                    ax.set_xlabel('FPR')
                    ax.set_ylabel('TPR')
                    ax.set_title(f'{atk_key.split("-")[3]} — ROC')
                    ax.legend()

                plt.tight_layout()
                fig.savefig('auc_scores.png', dpi=150)
                plt.close(fig)
                print("Saved auc_scores.png")
