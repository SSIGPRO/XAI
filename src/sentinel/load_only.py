from functools import partial
import sys
from time import time


import torch
from cuda_selector import auto_cuda
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'newpenny/XAI/EP').as_posix())

# Our stuff
from peepholelib.datasets.sentinel import Sentinel
from peepholelib.datasets.sentinel import SentinelWrap
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.models.svd_fns import conv2d_toeplitz_svd, linear_svd

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    parsed_path = Path.cwd()/'../../data/datasets'
    svds_path = Path.cwd()/'../data'
    svds_name = 'svds' 
    
    verbose = True 
    #model = "ae1Dregn16_2_ns0_k001.pth"

    model_dir = '/srv/newpenny/XAI/EP/Model_ready/ae1Dregn16_2_ns0_k001.pth'
    
    #--------------------------------
    # Model
    #--------------------------------
    
    sentinel_model = torch.load("/srv/newpenny/XAI/EP/Model_ready/ae1Dregn16_2_ns0_k001.pth",map_location=device, weights_only=False)
    target_layers = ['encoder.linear']

    layer_svd_rank = 10
    layer_cv_dim = 5
       
    model = ModelWrap(
            model = sentinel_model,
            device = device
            )

    sentinel = Sentinel(
        path = parsed_path
    )

    model.set_target_modules(
        target_modules = target_layers
    )
    
    with sentinel as s:
        s.load_only(
            loaders = ['train', 'val', 'test'],
            verbose = verbose
        )




        svd_fns = {
        
            'encoder.linear': partial(
            linear_svd, 
            rank = layer_svd_rank,
            device = device,
            ),

        }
    
        t0 = time()
        #d=sentinel._dss['train'][0]['data']
        #print(f'{d.shape}, {type(d)}')
        #quit()
        model.get_svds(
                path = svds_path,
                name = svds_name,
                target_modules = target_layers,
                sample_in = sentinel._dss['train'][0]['data'],
                svd_fns = svd_fns,
                verbose = verbose
                )
        print('time: ', time()-t0)

        for k in model._svds.keys():
            #print(f'model._svds.keys()={k}')
            for kk in model._svds[k].keys():
                print(f'model._svds[k].keys()={kk}')
                continue
                print('svd shapes: ', k, kk, model._svds[k][kk].shape)
            s = model._svds[k]['s']
            print(f's={s}')