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
from peepholelib.datasets.sentinel import SentinelWrap
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.models.svd_fns import conv2d_toeplitz_svd, linear_svd

from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import linear_svd_projection, conv2d_toeplitz_svd_projection, conv2d_kernel_svd_projection


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
    
    delta = 0.8
    corruptions = {
        'GWN': GWN(delta),
        # 'Constant': Constant(delta),
        # 'Step': Step(delta),
        # 'Impulse': Impulse(delta),
        # 'PrincipalSubspaceAlteration': PrincipalSubspaceAlteration(delta)
    }

    #--------------------------------
    # Model
    #--------------------------------
    
    sentinel_model = torch.load("/srv/newpenny/XAI/EP/Model_ready/ae1Dregn16_2_ns0_k001.pth",map_location=device, weights_only=False)

    model = ModelWrap(
            model = sentinel_model,
            device = device
            )

    sentinel = Sentinel(
        path = parsed_path
    )
    
    with sentinel as s:
        s.load_only(
            loaders = ['val'],
            verbose = verbose
        )
        s.get_corruptions(loaders = ['val'],
                          model = model,
                          corruptions = corruptions,
                          bs = 2**20)

        