# python stuff
import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/conv_red').as_posix())

import pandas as pd
import torch

# Our stuff
from utils.pareto import find_pareto

def get_best_config(hyperp_file):
    hf = Path(hyperp_file)
    _df = pd.read_pickle(hf)
   
    x = _df[['AUC OoD', 'AUC AA']].values
    idx = find_pareto(x)
    pareto = torch.tensor(_df.iloc[idx][['AUC OoD', 'AUC AA']].values)
    best = (pareto[:,0]*pareto[:,1]).argmax().item()

    best_config = _df.iloc[idx[best]]
    print('best config: ', best_config)

    return best_config

if __name__ == '__main__':
    bc = get_best_config()
