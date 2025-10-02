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

#wombats stuff

from wombats.anomalies.increasing import *
from wombats.anomalies.invariant import *
from wombats.anomalies.decreasing import *

# Our stuff
from peepholelib.datasets.sentinel import Sentinel
from peepholelib.models.model_wrap import ModelWrap

from torcheval.metrics import BinaryAUROC as AUC


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    parsed_path = Path.cwd()/'../../data/datasets'
    svds_path = Path.cwd()/'../../data'
    svds_name = 'svds' 
    
    cvs_path = Path.cwd()/'../../data/corevectors'
    cvs_name = 'cvs'

    loaders = ['train', 'val', 'test']
    bs = 1024*5*10
    verbose = True 
    input_key = 'data'
    #model = "ae1Dregn16_2_ns0_k001.pth"

    model_dir = '/srv/newpenny/XAI/EP/Model_ready/ae1Dregn16_2_ns0_k001.pth'
    
    #--------------------------------
    # Model
    #--------------------------------
    
    sentinel_model = torch.load("/srv/newpenny/XAI/EP/Model_ready/ae1Dregn16_2_ns0_k001.pth",map_location=device, weights_only=False)
    target_layers = ['encoder.linear']

    layer_svd_rank = 100
    layer_cv_dim = 5
       
    model = ModelWrap(
            model = sentinel_model,
            device = device
            )

    sentinel = Sentinel(
        path = parsed_path
    )

    column_names = ['RW1_motcurr', 'RW1_therm', 'RW1_cmd_volt', 'RW1_speed', 'RW2_motcurr',
                'RW2_therm', 'RW2_cmd_volt', 'RW2_speed', 'RW3_motcurr', 'RW3_therm',
                'RW3_cmd_volt', 'RW3_speed', 'RW4_motcurr', 'RW4_therm', 'RW4_cmd_volt',
                'RW4_speed']
    delta=0.8
    corruptions = {
        'GWN': GWN(delta),
        'Constant': Constant(delta),
        # 'Step': Step(delta),
        # 'Impulse': Impulse(delta),
        # 'PrincipalSubspaceAlteration': PrincipalSubspaceAlteration(delta)
    }
    
    with sentinel as s:
        s.load_only(
            loaders = ['val', 'test', 'val-c', ],
            verbose = verbose
        )
        loss = torch.nn.MSELoss(reduction='none')

        scores = {key: loss(dss['data'], dss['output']).mean(dim=(1,2)) for key, dss in s._dss.items()}
        print(scores['val'].shape)
        
        for j, corruption in enumerate(corruptions):
            for i, channel in enumerate(column_names):
                print(s._dss['val-c']['corruption']==1)
                idx = torch.argwhere((s._dss['val-c']['corruption']==j)&(s._dss['val-c']['channel']==i))#.squeeze(-1).tolist()
                print(idx, len(idx))
                quit()
                print(scores['val-c'][idx].shape)
                score = torch.stack((scores['val-c'][idx], scores['val']))
                results = torch.stack(torch.ones(len(idx)), torch.zeros(len(scores['val'])))
                auc = AUC().update(scores, results.int()).compute().item()
                if verbose: print(f'AUC for val {channel} & {corruption} split: {auc:.4f}')
                 


        