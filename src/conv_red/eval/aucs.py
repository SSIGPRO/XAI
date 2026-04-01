# python stuff
import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/conv_red').as_posix())
from statistics import geometric_mean as geomean

# plotting
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

# torch
import torch

# Our stuff
from configs.common import *
from utils.pareto import find_pareto

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

def plot_isometrics(**kwargs):
    step = kwargs.get('step', 0.1)
    n_points = kwargs.get('n_points', 100)
    lb_x = kwargs.get('lb_x', 0)
    ub_x = kwargs.get('ub_x', 1)
    lb_y = kwargs.get('lb_y', 0)
    ub_y = kwargs.get('ub_y', 1)

    auc_ood = r'$\overline{\Lambda}_{\rm{OoD}}$'
    auc_aa = r'$\overline{\Lambda}_{\rm{AA}}$'

    isos = torch.linspace(step, 1, int(1/step))
    for iso in isos:
        x = torch.linspace(1/n_points, 1, n_points)
        y = iso/x
        idx = torch.logical_and(x <= 1, y <= 1)
        x = x[idx]
        y = y[idx]
        plt.plot(x, y, c='xkcd:light grey', alpha=0.5, lw=0.2)
        plt.text(x[-1], y[-1], f'{iso:.1f}', c='xkcd:light grey', alpha=0.5, size='small')
    plt.legend(f'{auc_ood}.{auc_aa} isometrics')
    return


if __name__ == "__main__":
    hyperp_files = list(Path(args.data_path).glob('*/peepholes/*/*/hyperparams.pickle')) 
   
    dfs = []
    best_configs = pd.DataFrame({})

    for hf in hyperp_files:  
        _df = pd.read_pickle(hf)[['AUC OoD', 'AUC AA', 'model', 'reduction', 'analysis']]
        print(hf, len(_df))

        model = _df['model'][0]
        reduction = _df['reduction'][0]
        analysis = _df['analysis'][0]

        dfs.append(_df) 

        # same as utils/get_best_configs.py
        x = _df[['AUC OoD', 'AUC AA']].values
        idx = find_pareto(x)
        pareto = torch.tensor(_df.iloc[idx][['AUC OoD', 'AUC AA']].values)
        best = (pareto[:,0]*pareto[:,1]).argmax().item()
        best_config = _df[idx[best]:idx[best]+1] # I hate dataframes
        best_configs = pd.concat([best_configs, best_config])
    
    # change the names to plot the bests according to the hue 
    best_configs = best_configs.replace({
        'avgpooling': 'best avgpooling',
        'toeplitz': 'best toeplitz',
        'kernel': 'best kernel'
        })

    df = pd.concat(dfs, ignore_index=True)
    df = pd.concat([df, best_configs], ignore_index=True)
    auc_ood = r'$\overline{\Lambda}_{\rm{OoD}}$'
    auc_aa = r'$\overline{\Lambda}_{\rm{AA}}$'
    df = df.rename(columns={'AUC OoD': auc_ood, 'AUC AA': auc_aa})

    # plotting
    grid = sb.FacetGrid(
            data = df,
            row = 'analysis',
            col = 'model',
            col_order = ['VGG', 'MobileNet', 'ResNet', 'ConvNeXt'],
            hue = 'reduction',
            hue_order = ['avgpooling', 'toeplitz', 'kernel','best avgpooling', 'best toeplitz', 'best kernel'],
            hue_kws = dict(
                marker = ['.', '.', '.', 'd', 'd', 'd'],
                ),
            palette = ['xkcd:ocean blue', 'xkcd:tangerine', 'xkcd:grass green', 'xkcd:electric blue', 'xkcd:bright orange', 'xkcd:forest']
            )
    
    grid.map(
            plot_isometrics,
            lb_x = df[auc_ood].min(),
            ub_x = df[auc_ood].max(),
            lb_y = df[auc_aa].min(),
            ub_y = df[auc_aa].max(),
            )

    grid.map(
            sb.scatterplot,
            auc_ood,
            auc_aa,
            alpha = 0.6,
            s = 50,
            )

    grid.set_titles('{col_name} | {row_name}')
    grid.add_legend()

    plots_path.mkdir(parents=True, exist_ok=True)
    plt.savefig((plots_path/f'aucs.png').as_posix(), dpi=300, bbox_inches='tight')
