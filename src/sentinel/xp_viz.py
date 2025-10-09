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

    #--------------------------------
    # Directories definitions
    #--------------------------------
    parsed_path = '/srv/newpenny/XAI/generated_data/AE_sentinel/datasets_single'
    configs = ['all', 'single', 'RW']
    cis = ['high']#, 'medium', 'low']

    loaders = ['val','test']
    verbose = True 

    loaders_c = [f"{p}-{q}-c-{config}-{ci}"  for config in configs for ci in cis for p in loaders for q in loaders]
    loaders_RW = [f"{p}-{q}-c-RW-{ci}" for ci in cis for p in loaders for q in loaders]
    loaders_single = [f"{p}-{q}-c-single-{ci}" for ci in cis for p in loaders for q in loaders]
    
    loaders += loaders_c

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
        print(s._dss['val-val-c-RW-high']['data'].shape)
        
        
        seed = 42  # choose your seed (or parametrize it)
        g = torch.Generator(device='cpu').manual_seed(seed)

        perm = torch.randperm(len(s._dss['val']) , generator=g)
        val_idx  = perm[:100]
        test_idx = perm[100: 2*100]
        print(val_idx)

        for idx in range(80):
            idx *= 100

            c = s._dss['val-val-c-single-high']['data'][idx][0]
            o = s._dss['val']['data'][115374][0]
            
            fig, axs = plt.subplots(4, 4, figsize=(8, 8))
            for i, ax in enumerate(axs.flat):
                ax.plot(c[i].cpu().numpy(), label='c')
                ax.plot(o[i].cpu().numpy(), label='o')
                ax.set_title(f"{i}")
                ax.legend()
                ax.axis('tight')
            plt.tight_layout()
            plt.savefig(f'c_o{idx}_single.png')

        for idx in range(20):
            idx *= 100

            c = s._dss['val-val-c-RW-high']['data'][idx][0]
            o = s._dss['val']['data'][115374][0]
            
            fig, axs = plt.subplots(4, 4, figsize=(8, 8))
            for i, ax in enumerate(axs.flat):
                ax.plot(c[i].cpu().numpy(), label='c')
                ax.plot(o[i].cpu().numpy(), label='o')
                ax.set_title(f"{i}")
                ax.legend()
                ax.axis('tight')
            plt.tight_layout()
            plt.savefig(f'c_o{idx}_RW.png')

        for idx in range(5):
            idx *= 100

            c = s._dss['val-val-c-all-high']['data'][idx][0]
            o = s._dss['val']['data'][115374][0]
            
            fig, axs = plt.subplots(4, 4, figsize=(8, 8))
            for i, ax in enumerate(axs.flat):
                ax.plot(c[i].cpu().numpy(), label='c')
                ax.plot(o[i].cpu().numpy(), label='o')
                ax.set_title(f"{i}")
                ax.legend()
                ax.axis('tight')
            plt.tight_layout()
            plt.savefig(f'c_o{idx}_RW.png')

        