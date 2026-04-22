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

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model',   choices=['WRN28', 'WRN70'], default='WRN28')
parser.add_argument('-p', '--path',    default=Path('/srv/newpenny/XAI/generated_data/attacks').as_posix())
parser.add_argument('--proto_th',      type=float, default=40)
parser.add_argument('-o', '--out_dir', default='.')
args, _ = parser.parse_known_args()

def loader_keys(model_version):
    return {
        'proto': f'{dataset_name}-train-{model_version}',
        'eval': [
            f'{dataset_name}-test-{model_version}',
            f'{dataset_name}-test-APGD-ce-{model_version}',
            f'{dataset_name}-test-APGD-t-{model_version}',
        ],
    }

# -------------------------------------------------------------------------
# Metric helpers
# -------------------------------------------------------------------------

def risk_coverage_from_score(score, results):
    """(coverage, risk) arrays sorted by descending score."""
    order = score.argsort(descending=True)
    correct = results.float()[order]
    n = len(correct)
    coverage = torch.arange(1, n + 1).float() / n
    risk = 1.0 - correct.cumsum(0) / torch.arange(1, n + 1).float()
    return coverage.numpy(), risk.numpy()

def compute_metrics(score, results):
    """Returns (AUC, FPR@TPR=95%)."""
    y_score = score.numpy()
    y_true  = results.int().numpy()
    auc = roc_auc_score(y_true, y_score)
    fprs, tprs, _ = roc_curve(y_true, y_score)
    idx   = next((i for i, t in enumerate(tprs) if t >= 0.95), -1)
    fpr95 = fprs[idx]
    return auc, fpr95

if __name__ == "__main__":

    if args.model == 'WRN28':
        from configs.models.wrn28_10 import *
    elif args.model == 'WRN70':
        from configs.models.wrn70_16 import *
    else:
        raise RuntimeError("Select a model <WRN28|WRN70>")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_path = Path(args.path)
    ds_path   = base_path / 'datasets' / dataset_name
    verbose   = True

    versions = ['standard', 'robust']
    reducers = ['random', 'svd']

    scores = {v: {r: {} for r in reducers} for v in versions}
    results_cache = {v: {} for v in versions}       # results[version][key]
    atk_succ_cache = {v: {} for v in versions}       # attack_success[version][key]
    conf_cache = {v: {} for v in versions}           # max-softmax confidence[version][clean_key]
    sm = torch.nn.Softmax(dim=1)
    proto_thr = {
        'standard': 0.9,
        'robust': 0.5,
    }

    dataset = ParsedDataset(
        path=ds_path
        )

    for version in versions:
        model_name = f'{args.model}-{version}'

        keys = loader_keys(model_version=model_name)
        proto_key = keys['proto']
        eval_keys = keys['eval']

        inference_names = {
            f'{dataset_name}-train': [model_name],
            f'{dataset_name}-test':  [
                model_name, 
                f'APGD-ce-{model_name}', 
                f'APGD-t-{model_name}'],
        }

        with dataset as ds:
            ds.load_only(
                loaders = list(inference_names.keys()),
                inference_names = inference_names,
                verbose = verbose,
            )

            # cache tensors needed for plotting (must happen inside the context)
            for key in eval_keys:
                results_cache[version][key] = ds._dss[key]['result'].clone()
                if 'APGD' in key:
                    atk_succ_cache[version][key] = ds._dss[key]['attack_success'].clone()
                else:
                    conf_cache[version][key] = sm(ds._dss[key]['output']).max(dim=1).values.clone()

            for reducer in reducers:
                phs_path = base_path / f'{dataset_name}_{model_name}' / 'peepholes'
                phs_name = f'peepholes.{reducer}'

                peepholes = Peepholes(path=phs_path, name=phs_name, device=device)

                with peepholes as ph:

                    ph.load_only(
                        loaders=list(ds._dss.keys()),
                        verbose=verbose
                        )
                    
                    target_modules = list(ph._phs[proto_key].keys())

                    ret, proto = conceptogram_protoclass_score(
                        datasets       = ds,
                        peepholes      = ph,
                        loaders        = list(ds._dss.keys()),
                        target_modules = target_modules,
                        proto_key      = proto_key,
                        proto_threshold= proto_thr[version],
                        verbose        = verbose,
                    )

                scores[version][reducer] = {k: ret[k]['Proto-Class'] for k in eval_keys}

    for version in versions:
        for reducer in reducers:
            for eval_key, score in scores[version][reducer].items():
               n_nan = score.isnan().sum().item()
               print(f'{eval_key}: {n_nan} NaNs / {len(score)} samples')
    
    # -------------------------------------------------------------------------
    # Shared layout
    # -------------------------------------------------------------------------
    cases = [
        # ('standard', 'random'),
        ('standard', 'svd'),
        # ('robust',   'random'),
        ('robust',   'svd'),
    ]
    case_labels   = [f'{v}/{r}' for v, r in cases]
    eval_key_types = ['clean', 'APGD-ce', 'APGD-t']   # indices 0,1,2 in eval_keys

    colors_cases = {
        'standard/random': 'xkcd:cobalt',
        'standard/svd':    'xkcd:cerulean',
        'robust/random':   'xkcd:bluish green',
        'robust/svd':      'xkcd:grass green',
    }

    def get_key(version, key_type):
        """Return the full dataset key for (version, 'clean'|'APGD-ce'|'APGD-t')."""
        idx = eval_key_types.index(key_type)
        return loader_keys(f'{args.model}-{version}')['eval'][idx]

    # -------------------------------------------------------------------------
    # Figure 1 — Risk-Coverage Curves (clean test set only)
    #   Single subplot, 4 lines — one per case
    # -------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(7, 5))

    for (version, reducer), label in zip(cases, case_labels):
        key     = get_key(version, 'clean')
        score   = scores[version][reducer][key]
        results = results_cache[version][key]
        cov, risk = risk_coverage_from_score(score, results)
        ax1.plot(cov, risk, label=label, color=colors_cases[label], linewidth=0.8, alpha=0.7)

        # model confidence (max softmax) as additional score — dashed line
        conf = conf_cache[version][key]
        cov_conf, risk_conf = risk_coverage_from_score(conf, results)
        ax1.plot(cov_conf, risk_conf, label=f'{label} (conf)',
                 color=colors_cases[label], linewidth=0.8, linestyle='--', alpha=0.7)

    ax1.set_xlabel('Coverage')
    ax1.set_ylabel('Risk')
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)

    fig1.suptitle(f'{args.model} — Risk-Coverage Curve (Proto-Class Score)', fontsize=13)
    plt.tight_layout()
    out1 = out_dir / 'rc_curve_proto.png'
    plt.savefig(out1.as_posix(), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out1}')

    # -------------------------------------------------------------------------
    # Figure 2 — KDE: correct vs wrong, with AUC and FPR@TPR=95%
    #   One figure per eval type, 2×2 subplots for the 4 cases
    # -------------------------------------------------------------------------
    c_correct   = 'xkcd:cobalt'
    c_incorrect = 'xkcd:dark hot pink'

    for key_type in ['clean']:
        fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
        axes2 = axes2.flatten()

        for idx, ((version, reducer), label) in enumerate(zip(cases, case_labels)):
            ax      = axes2[idx]
            key     = get_key(version, key_type)
            score   = scores[version][reducer][key]
            results = results_cache[version][key]

            auc, fpr95 = compute_metrics(score, results)

            df = pd.DataFrame({
                'score': score.numpy(),
                'cls': ['correct' if r else 'incorrect' for r in results.tolist()],
            })
            sb.kdeplot(data=df[df['cls'] == 'correct'],   ax=ax,
                       x='score', color=c_correct,   common_norm=False,
                       clip=[0., 1.], label='correct')
            sb.kdeplot(data=df[df['cls'] == 'incorrect'], ax=ax,
                       x='score', color=c_incorrect, common_norm=False,
                       clip=[0., 1.], label='incorrect')
            ax.set_title(f'{label}\nAUC={auc:.3f}   FPR@95%TPR={fpr95:.3f}', fontsize=9)
            ax.set_xlabel('Proto-Class Score')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        safe_name = key_type.replace('-', '_')
        fig2.suptitle(
            f'{args.model} — Proto-Class score distribution: correct vs incorrect ({key_type})',
            fontsize=13,
        )
        plt.tight_layout()
        out2 = out_dir / f'kde_correct_wrong_{safe_name}.png'
        plt.savefig(out2.as_posix(), dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved: {out2}')

    # -------------------------------------------------------------------------
    # Figure 3 — Successful attacks vs corresponding original samples
    #   One figure per attack type (APGD-ce, APGD-t), 2×2 subplots for 4 cases
    #   Per sample: adversarial score vs the score of the same sample when clean
    # -------------------------------------------------------------------------
    c_atk  = 'xkcd:dark hot pink'
    c_orig = 'xkcd:cobalt'

    for atk_type in ['APGD-ce', 'APGD-t']:
        fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))
        axes3 = axes3.flatten()

        for idx, ((version, reducer), label) in enumerate(zip(cases, case_labels)):
            ax        = axes3[idx]
            atk_key   = get_key(version, atk_type)
            clean_key = get_key(version, 'clean')

            if atk_key not in atk_succ_cache[version]:
                ax.set_visible(False)
                continue

            mask       = atk_succ_cache[version][atk_key]   # bool: attack succeeded
            score_atk  = scores[version][reducer][atk_key][mask]
            score_orig = scores[version][reducer][clean_key][mask]

            n_success = mask.sum().item()

            # AUC: higher proto-class score → more likely original (clean)
            y_labels = np.array([1] * len(score_orig) + [0] * len(score_atk))
            y_scores = np.concatenate([score_orig.numpy(), score_atk.numpy()])
            auc_atk = roc_auc_score(y_labels, y_scores)

            df3 = pd.DataFrame({
                'score': y_scores,
                'source': (
                    ['original (clean)']  * len(score_orig) +
                    ['successful attack'] * len(score_atk)
                ),
            })
            sb.kdeplot(data=df3[df3['source'] == 'successful attack'], ax=ax,
                       x='score', color=c_atk,  common_norm=False,
                       clip=[0., 1.], label=f'successful attack (n={n_success})')
            sb.kdeplot(data=df3[df3['source'] == 'original (clean)'],  ax=ax,
                       x='score', color=c_orig, common_norm=False,
                       clip=[0., 1.], label='original (clean)')
            ax.set_title(f'{label}\nAUC={auc_atk:.3f}', fontsize=9)
            ax.set_xlabel('Proto-Class Score')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        safe_atk = atk_type.replace('-', '_')
        fig3.suptitle(
            f'{args.model} — Proto-Class score: successful {atk_type} vs original',
            fontsize=13,
        )
        plt.tight_layout()
        out3 = out_dir / f'atk_vs_orig_{safe_atk}.png'
        plt.savefig(out3.as_posix(), dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved: {out3}')

    # -------------------------------------------------------------------------
    # Figure 4 — Joint KDE: APGD-ce + APGD-t combined vs originals
    #   2×2 subplots for the 4 configurations, single AUROC per subplot
    # -------------------------------------------------------------------------
    fig4, axes4 = plt.subplots(2, 2, figsize=(12, 10))
    axes4 = axes4.flatten()

    for idx, ((version, reducer), label) in enumerate(zip(cases, case_labels)):
        ax        = axes4[idx]
        clean_key = get_key(version, 'clean')
        atk_key_ce = get_key(version, 'APGD-ce')
        atk_key_t  = get_key(version, 'APGD-t')

        mask_ce = atk_succ_cache[version].get(atk_key_ce)
        mask_t  = atk_succ_cache[version].get(atk_key_t)
        if mask_ce is None or mask_t is None:
            ax.set_visible(False)
            continue

        # Proto scores of successful attacks (both methods)
        score_atk_ce  = scores[version][reducer][atk_key_ce][mask_ce]
        score_atk_t   = scores[version][reducer][atk_key_t][mask_t]
        score_atk_all = torch.cat([score_atk_ce, score_atk_t])

        # Proto scores of the corresponding originals (clean) at the same indices
        score_orig_ce  = scores[version][reducer][clean_key][mask_ce]
        score_orig_t   = scores[version][reducer][clean_key][mask_t]
        score_orig_all = torch.cat([score_orig_ce, score_orig_t])

        # AUROC: original=1, attack=0  (higher proto-score → more likely original)
        y_labels = np.array([1] * len(score_orig_all) + [0] * len(score_atk_all))
        y_scores = np.concatenate([score_orig_all.numpy(), score_atk_all.numpy()])
        valid = ~np.isnan(y_scores)
        auc_joint = roc_auc_score(y_labels[valid], y_scores[valid])

        df4 = pd.DataFrame({
            'score':  y_scores,
            'source': (
                ['original (clean)'] * len(score_orig_all) +
                ['attack']           * len(score_atk_all)
            ),
        })
        sb.kdeplot(data=df4[df4['source'] == 'original (clean)'], ax=ax,
                   x='score', color=c_orig, common_norm=False,
                   clip=[0., 1.], label='original (clean)')
        sb.kdeplot(data=df4[df4['source'] == 'attack'], ax=ax,
                   x='score', color=c_atk,  common_norm=False,
                   clip=[0., 1.], label='APGD-ce + APGD-t')
        ax.set_title(f'{label}\nAUROC = {auc_joint:.3f}', fontsize=9)
        ax.set_xlabel('Proto-Class Score')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig4.suptitle(
        f'{args.model} — Proto-Class score distribution\n'
        f'successful APGD-ce + APGD-t (joint) vs original',
        fontsize=13,
    )
    plt.tight_layout()
    out4 = out_dir / 'proto_score_dist_joint.png'
    plt.savefig(out4.as_posix(), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out4}')
