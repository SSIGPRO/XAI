# python stuff
import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/conv_red').as_posix())

import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

# Our stuff
from configs.common import *

if __name__ == "__main__":
    hyperp_files = list(Path(args.data_path).glob('*/peepholes/*/*/hyperparams.pickle')) 
   
    dfs = [pd.read_pickle(hf)[['ood_auc', 'aa_auc', 'model', 'reduction', 'analysis']] for hf in hyperp_files] 
    df = pd.concat(dfs, ignore_index=True)

    # Captalize names
    _cn_map = {
            'ood_auc': 'AUC OoD',
            'aa_auc': 'AUC AA'
            }
    df = df.rename(columns=_cn_map)
    df['analysis'] = df['analysis'].apply(lambda x: x.upper())
    df['model'] = df['model'].apply(lambda x: x.upper() if x == 'vgg' else 'MobileNet')

    # plotting
    grid = sb.FacetGrid(data=df, row='model', col='analysis', hue='reduction')
    grid.map(
            sb.scatterplot,
            'AUC OoD',
            'AUC AA',
            alpha = 0.75,
            )
    grid.set_titles('{col_name} | {row_name}')
    grid.add_legend()

    plots_path.mkdir(parents=True, exist_ok=True)
    plt.savefig((plots_path/f'aucs.png').as_posix(), dpi=300, bbox_inches='tight')
