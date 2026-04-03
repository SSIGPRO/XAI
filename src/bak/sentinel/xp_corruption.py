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

# Our stuff
from peepholelib.datasets.sentinel import Sentinel
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.models.sentinel.model_sentinel import CONV_AE2D

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--emb_size", required=True, type=str, help="Model type to use")
parser.add_argument("--ci", required=True, type=str, help="Corruption intensity")
args = parser.parse_args()

emb_size = args.emb_size
ci = args.ci

if ci == 'high': 
    from config_anomalies import ch as corruptions
elif ci == 'medium':
    from config_anomalies import cm as corruptions
elif ci == 'low':
    from config_anomalies import cl as corruptions
else:
    raise RuntimeError('The configuration is not available choose among [low|medium|high]')

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    model_path = '/srv/newpenny/SPACE/FIORIRE2_Maurizio/src/Artifacts'
    model_name = f"conv2dAE_SENT_L16_K3-3_Emb{emb_size}_Lay0_C16_S42.pth"
    
    parsed_path = f'/srv/newpenny/XAI/generated_data/AE_sentinel/datasets_{emb_size}_all'

    loaders = ['val', 'test']
    bs = 2**18
    verbose = True 

    num_sensors = 16
    seq_len = 16
    kernel = [3, 3]
    lay3 = False   
    n_samples = 10000

    #--------------------------------
    # Model
    #--------------------------------
    
    sentinel_model = CONV_AE2D(
              num_sensors=num_sensors,
              seq_len=seq_len,
              kernel_size=kernel,
              embedding_size=emb_size,
              lay3=lay3
          )

    model = ModelWrap(
            model = sentinel_model,
            device = device
            )

    model.load_checkpoint(
            path = model_path,
            name = model_name,
            sd_key = 'model_state_dict'
            )

    sentinel = Sentinel(
        path = parsed_path
    )
    
    with sentinel as s:
        s.load_only(
            loaders = loaders,
            verbose = verbose
        )

        # s.get_corruptions_single(
        #         loaders = loaders,
        #         model = model,
        #         corruptions = corruptions,
        #         n_samples = 1500,
        #         bs = bs,
        #         verbose = verbose,
        #         suffix = ci,
        #         thr = 0.003
        #         )

        s.get_corruptions_all(
                loaders = loaders,
                model = model,
                corruptions = corruptions,
                n_samples = 10**4,
                bs = bs,
                verbose = verbose,
                suffix = ci,
                thr = 0.003
                )

        # s.get_corruptions_RW(
        #         loaders = loaders,
        #         model = model,
        #         corruptions = corruptions,
        #         n_samples = 2000,
        #         bs = bs,
        #         verbose = verbose,
        #         suffix = ci,
        #         thr = 0.003
        #         )
        
