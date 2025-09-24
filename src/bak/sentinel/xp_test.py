from functools import partial
import sys
from time import time
from matplotlib import pyplot as plt
import numpy as np

import torch
from cuda_selector import auto_cuda
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'newpenny/XAI/EP').as_posix())
from tqdm import tqdm
import pandas as pd
from math import floor 

#wombats stuff

from wombats.anomalies.increasing import *
from wombats.anomalies.invariant import *
from wombats.anomalies.decreasing import *

# Our stuff
from peepholelib.datasets.sentinel import Sentinel

from torchmetrics.classification import BinaryROC, BinaryAUROC

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--emb_size", required=True, type=str, help="Model type to use")
args = parser.parse_args()

emb_size = args.emb_size

if __name__ == "__main__":
    # use_cuda = torch.cuda.is_available()
    # device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    # print(f"Using {device} device")

    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 1
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")

    ws = 16

    test_file = Path('/srv/newpenny/dataset/TASI/sentinel/sentinel_4s_clean_std/test_data.pkl')
    data_test_std = pd.read_pickle(test_file.as_posix())

    _data = torch.tensor(data_test_std.values, dtype=torch.float32)
    nw = floor(_data.shape[0]/ws) # num windows

    data = _data[:ws*nw]
    data = data.reshape(-1, ws, data.shape[-1]) # 16 is the number of signals
    data = data.permute(0, 2, 1).unsqueeze(dim=1) ## B, 1, nc, nw

    idx = data.isnan().any(dim=(2,3)).logical_not()#used dim instead of axis

    time_stamps = data_test_std.index.to_numpy()
    time_stamps = time_stamps[:ws*nw]
    time_stamps = time_stamps.reshape(-1, ws)
    
    time_stamps = time_stamps[idx.squeeze(dim=1).numpy()]

    #--------------------------------
    # Directories definitions
    #--------------------------------
    parsed_path = f'/srv/newpenny/XAI/generated_data/AE_sentinel/datasets_{emb_size}'
    configs = ['all', 'single', 'RW']
    cis = ['high']#, 'medium', 'low']

    loaders = ['val', 'test', 'test_ori']
    verbose = True 

    loaders_c = [f"test-{p}-c-{config}-{ci}"  for config in configs for ci in cis for p in loaders]
    loaders_RW = [f"test-{p}-c-RW-{ci}" for ci in cis for p in loaders]
    loaders_single = [f"test-{p}-c-single-{ci}" for ci in cis for p in loaders]
    
   # loaders += loaders_c

    #--------------------------------
    # Dataset
    #--------------------------------

    sentinel = Sentinel(
        path = parsed_path
    )

    column_names = ['RW1_motcurr', 'RW1_therm', 'RW1_cmd_volt', 'RW1_speed', 'RW2_motcurr',
                'RW2_therm', 'RW2_cmd_volt', 'RW2_speed', 'RW3_motcurr', 'RW3_therm',
                'RW3_cmd_volt', 'RW3_speed', 'RW4_motcurr', 'RW4_therm', 'RW4_cmd_volt',
                'RW4_speed']
    
    rw = range(4)
    c = range(16)
    
    corr_names = ['Constant', 'Step', 'Impulse', 'GWN','PSA']
    n_corr = len(corr_names)
    
    with sentinel as s:
        s.load_only(
            loaders = loaders,
            verbose = verbose
        )

        loss = torch.nn.MSELoss(reduction='none')        

        scores = {key: loss(dss['data'], dss['output']).mean(dim=(2,3)) for key, dss in s._dss.items()}

        sample_mask = (s._dss['test_ori']['label'] == 0).all(dim=(1,2))
        
        sc = scores['test_ori'][sample_mask]

        # Indices and values
        peak_idx = [24766, 134493, 143991, 172864, 240753]
        peak_vals = sc[peak_idx]
        peak_timestamps = time_stamps[peak_idx]
        peak_dates = [pd.to_datetime(ts[0]).strftime('%m.%d') for ts in peak_timestamps]

        print("peak indices:", peak_idx)
        print("peak values:", peak_vals)
        print(time_stamps[peak_idx])

        plt.plot(sc)
        plt.scatter(peak_idx, peak_vals, color='red', s=50)
        plt.xticks(peak_idx, peak_dates, fontsize=8)

        plt.savefig('score_ori_cleaned.png')