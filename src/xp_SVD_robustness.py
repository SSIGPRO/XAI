import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# Our stuff
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.models.svd_fns import linear_svd 

# python stuff
from time import time
from functools import partial
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import simpson as simps
import json
from typing import Union
import requests

from scipy.stats import spearmanr, kendalltau

# torch stuff
import torch
from torchvision.models import vgg16
from cuda_selector import auto_cuda

# robustbench stuff
import robustbench
from robustbench.model_zoo import model_dicts
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from robustbench.utils import load_model
from robustbench.utils import get_leaderboard_latex

# argparse stuff
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=["cifar10", "cifar100", "imagenet"], help="Dataset used as benchmark")
parser.add_argument("threat", choices=["Linf", "L2", "corruptions"], help="Threat to test the robustness against")
args = parser.parse_args()

dataset = args.dataset
threat = args.threat

if sys.argv[1] == 'cifar10':
    benchmark = BenchmarkDataset.cifar_10 
elif sys.argv[1] == 'cifar100':
    benchmark = BenchmarkDataset.cifar_100 
elif sys.argv[1] == 'imagenet':
    benchmark = BenchmarkDataset.imagenet
else:
    raise RuntimeError('Select a configuration by runing \'python xp_get_corevectors.py <cifar10|cifar100|imagenet>\'')

if sys.argv[2] == 'Linf':
    threat_model = ThreatModel.Linf
elif sys.argv[2] == 'L2':
    threat_model = ThreatModel.L2
elif sys.argv[2] == 'corruptions':
    threat_model = ThreatModel.corruptions
else:
    raise RuntimeError('Select a configuration by runing \'python xp_get_corevectors.py dataset <Linf|L2|corruptions>\'')

def get_leaderboard_df(
    dataset_name: str,                     # e.g. 'cifar10' (kept for compatibility; not used here)
    dataset: Union[str, BenchmarkDataset], # 'cifar10' | 'cifar100' | 'imagenet' | BenchmarkDataset
    threat_model: Union[str, ThreatModel], # 'Linf' | 'L2' | 'corruptions' | ThreatModel
    sort_by: str = 'external'              # or 'autoattack_acc'
):
    dataset_ = BenchmarkDataset(dataset)
    threat_model_ = ThreatModel(threat_model)

    # list of model IDs available for this (dataset, threat_model)
    model_ids = list(model_dicts[dataset_][threat_model_].keys())

    base_url = (
        "https://raw.githubusercontent.com/RobustBench/robustbench/master/"
        f"model_info/{dataset_.value}/{threat_model_.value}"
    )

    entries = []
    for mid in model_ids:
        url = f"{base_url}/{mid}.json"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            info = r.json()
        except Exception:
            # some repos/versions may use slightly different filenames; skip if not found
            # (you could add custom fallbacks here if needed)
            continue

        # prefer 'external', fallback to 'autoattack_acc'
        score = info.get(sort_by, info.get('autoattack_acc', None))
        if score is not None:
            entries.append({'model': mid, 'score': float(score)})

    if not entries:
        return pd.DataFrame(columns=['model', 'score', 'rank'])

    # sort descending by score and add rank
    df = pd.DataFrame(entries).sort_values('score', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1
    return df

def compare_rankings(df_my: pd.DataFrame,
                     df_lb: pd.DataFrame,
                     score_cols=('area_s','area_ds','s_max','s_max_min_ratio'),
                     higher_is_better=None,
                     topk_list=(1, 2, 3, 5,10,20)):
    """
    Confronta, per ogni score in score_cols, il ranking ottenuto con quello ufficiale (df_lb['rank']).
    - df_my: DataFrame con colonne ['model', <score_cols...>]
    - df_lb: DataFrame con colonne ['model', 'rank'] (1 = migliore)
    - higher_is_better: dict opzionale {score_col: bool}; default True per tutti
    - topk_list: tuple di k per calcolare l'overlap nei top-k
    Ritorna: results_df, ranks_by_score (dict score->DataFrame con ['model','my_rank','official_rank'])
    """
    if higher_is_better is None:
        higher_is_better = {c: True for c in score_cols}

    # allineo i due DF sui model_id (che coincideranno)
    base = df_my.merge(df_lb[['model','rank']], on='model', how='inner').rename(columns={'rank':'official_rank'})

    results = []
    ranks_by_score = {}

    for col in score_cols:
        if col not in base.columns:
            continue

        # Se higher_is_better[col] è True ordino desc, altrimenti asc
        ascending = not higher_is_better.get(col, True)

        tmp = base[['model', col, 'official_rank']].dropna().copy()
        tmp = tmp.sort_values(col, ascending=ascending).reset_index(drop=True)
        tmp['my_rank'] = tmp.index + 1

        # correlazioni di ranking (minore è meglio per rank, ma Spearman/Kendall sono invarianti a monotone)
        rho, rho_p = spearmanr(tmp['my_rank'], tmp['official_rank'])
        tau, tau_p = kendalltau(tmp['my_rank'], tmp['official_rank'])

        # top-k overlap
        overlaps = {}
        my_sorted_models = tmp.sort_values('my_rank')['model'].tolist()
        off_sorted_models = tmp.sort_values('official_rank')['model'].tolist()
        for k in topk_list:
            A = set(my_sorted_models[:k])
            B = set(off_sorted_models[:k])
            overlaps[f'top{k}_overlap'] = len(A & B) / max(1, min(k, len(tmp)))

        results.append({
            'score': col,
            'higher_is_better': higher_is_better.get(col, True),
            'n_common': len(tmp),
            'spearman_rho': rho,
            'spearman_p': rho_p,
            'kendall_tau': tau,
            'kendall_p': tau_p,
            **overlaps
        })

        ranks_by_score[col] = tmp[['model','my_rank','official_rank', col]]

    results_df = pd.DataFrame(results).sort_values('spearman_rho', ascending=False)
    return results_df, ranks_by_score

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    
    svds_path = Path.cwd()/f'../data/robustness/{dataset}_{threat}'
    svds_name = 'svds' 

    verbose = True
    seed = 29
    
    #--------------------------------
    # Model analysis
    #--------------------------------

    df_lb = get_leaderboard_df(dataset, dataset, threat)

    csv_path = svds_path / f"{svds_name}_areas.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        all_records = {}

        for key in list(model_dicts[benchmark][threat_model].keys()):
            try:
        
                nn = load_model(model_name=key, dataset=dataset, threat_model=threat) 
                model = ModelWrap(
                        model = nn,
                        device = device
                        )
                
                target_layers = [list(nn._modules.keys())[-1]]

                model.set_target_modules(
                        target_modules = target_layers,
                        verbose = verbose
                        )

                #--------------------------------
                # SVDs 
                #--------------------------------
                svd_fns = {
                        target_layers[0]: partial(
                            linear_svd,
                            device = device,
                            ),
                        }

                t0 = time()
                model.get_svds(
                        path = svds_path,
                        name = svds_name+'_'+key,
                        target_modules = target_layers,
                        sample_in = torch.randn(3,32,32),
                        svd_fns = svd_fns,
                        verbose = verbose
                        )
                print('time: ', time()-t0)

                print('\n----------- svds:')
                for k in model._svds.keys():
                    for kk in model._svds[k].keys():
                        print('svd shapes: ', k, kk, model._svds[k][kk].shape)

                    s = np.asarray(model._svds[k]['s'], dtype=float)
                    ds = -np.gradient(s)

                    # metrics
                    x = np.arange(len(s))
                    area_s  = float(simps(s, x=x))
                    area_ds = float(simps(ds, x=x))
                    s_max   = float(np.max(s))
                    s_min   = float(np.min(s))
                    eps = 1e-12
                    s_max_min_ratio = float(s_max / max(s_min, eps))  # safe divide

                    all_records[key] = {
                        'area_s': area_s,
                        'area_ds': area_ds,
                        's_max': s_max,
                        's_max_min_ratio': s_max_min_ratio
                    }
                    print(f"[{key}] area_s={area_s:.6f} | area_ds={area_ds:.6f} | s_max={s_max:.6f} | ratio={s_max_min_ratio:.6f}")

                    if len(s.shape) == 1:
                        # plt.figure()
                        # plt.plot(s, '-')
                        # plt.xlabel('Rank')
                        # plt.ylabel('EigenVec')
                        fig, axes = plt.subplots(2, 1, figsize=(6,6), sharex=True)

                        axes[0].plot(s, '-')
                        axes[0].set_ylabel('Singular values')
                        axes[0].set_title('Singular value profile')

                        axes[1].plot(ds, '--', color='orange', label=f'∫ds={area_ds:.2f}')
                        axes[1].set_ylabel('Derivative')
                        axes[1].set_xlabel('Rank')
                        axes[1].set_title('Derivative of singular value profile')
                        axes[1].legend()

                        plt.tight_layout()
                        plt.show()

                    else:
                        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                        _x = torch.linspace(0, s.shape[1]-1, s.shape[1])
                        for r in range(s.shape[0]):
                            plt.plot(xs=_x, ys=s[r,:], zs=r, zdir='y')
                        ax.set_xlabel('Rank')
                        ax.set_ylabel('Channel')
                        ax.set_zlabel('EigenVec')
                    
                    plt.savefig(svds_path/(svds_name+'_'+key+'/'+k+'/singularValues.png'), dpi=300, bbox_inches='tight')
                    plt.close()

            except Exception as e:
                print(f"!! Skipping {key} due to error: {e}")
                all_records[key] = {
                    'area_s': np.nan,
                    'area_ds': np.nan,
                    's_max': np.nan,
                    's_max_min_ratio': np.nan,
                    'error': str(e)
                }

        df = pd.DataFrame.from_dict(all_records, orient='index').rename_axis('model').reset_index()
        df = df[['model', 'area_s', 'area_ds', 's_max', 's_max_min_ratio'] + ([ 'error' ] if 'error' in df.columns else [])]
        df = df.sort_values('s_max_min_ratio')  # or 'area_s' if you prefer
        
        df.to_csv(csv_path, index=False)
        print("\nSaved metrics to:", csv_path)

    score_cols = ['area_s','area_ds','s_max','s_max_min_ratio', 'combo_score']

    # Direzione della metrica (True = più alto è meglio)
    # Se ad es. pensi che 's_max_min_ratio' sia "meglio se più basso", mettilo a False
    higher_is_better = {
        'area_s': False,
        'area_ds': False,
        's_max': False,
        's_max_min_ratio': False,
        'combo_score' : False
    }

    df['area_s_norm'] = (df['area_s'] - df['area_s'].min()) / (df['area_s'].max() - df['area_s'].min())
    df['s_max_norm'] = (df['s_max'] - df['s_max'].min()) / (df['s_max'].max() - df['s_max'].min())

    df['combo_score'] = df['s_max_norm'] + df['area_s_norm']
    df_sorted = df.sort_values('combo_score', ascending=True).reset_index(drop=True)

    print(df_sorted[['model','s_max','area_s','combo_score']])

    results_df, ranks = compare_rankings(df_sorted, df_lb, score_cols, higher_is_better)
    print(results_df)
    quit()
    df_sorted = df.sort_values('s_max', ascending=True).reset_index(drop=True)

    # Show only the model IDs in order
    print(df_sorted['model'].tolist())

 