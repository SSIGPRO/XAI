import sys

import torch
from cuda_selector import auto_cuda
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'newpenny/XAI/EP').as_posix())
sys.path.insert(0, (Path.home()/'repos/FIORIRE/src').as_posix())

# Our stuff
from peepholelib.datasets.sentinel import Sentinel
from peepholelib.datasets.sentinel import SentinelWrap
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.models.sentinel.model_sentinel import CONV_AE2D


# XPs stuff
#from configs.common import *

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    ds_path = '/srv/newpenny/dataset/TASI/sentinel/sentinel_4s_clean_std'
    parsed_path = Path.cwd()/'../../data/datasets'
    #trial_path = Path.cwd()/'../../data/datasets'
    
    model_path = Path.home()/'repos/FIORIRE/data/train_cps' 
    model_name = 'checkpoints.149.pt'
    input_key = 'data'
    emb_size = 'large'
    num_sensors = 16
    seq_len = 16
    kernel = [3, 3]
    lay3 = False   

    seed = 42
    bs = 2**17

    verbose = True 
    #sentinel_model = CONV_AE2D()
    
    #print(sentinel_model)
    #quit()

    #------------------CONV 1D--------------#    
    
    #model_dir = '/srv/newpenny/XAI/EP/Model_ready/ae1Dregn16_2_ns0_k001.pth'
    #sentinel_model = torch.load(model_dir,map_location=device, weights_only=False)
    #print(next(sentinel_model.parameters()).dtype)
    #print(self.data.dtype)
    
    #-------------------       --------------#
    
    #--------------------------------
    # Dataset 
    #--------------------------------
    sentinel_model = CONV_AE2D(
              num_sensors = num_sensors,
              seq_len = seq_len,
              kernel_size = kernel,
              embedding_size = emb_size,
              lay3 = False
          )

    model = ModelWrap(
            model = sentinel_model,
            device = device
            )

    model.load_checkpoint(
            path = model_path,
            name = model_name,
            sd_key = 'state_dict'
            )
    
    #--------------------------------
    # Dataset 
    #--------------------------------

    ds_wrap = SentinelWrap(
            path = ds_path,
            parsed_path = parsed_path
            )

    ds_wrap.__load_data__(
            seed = seed,
            verbose = verbose
            )
    
    parsed_ds = Sentinel.parse_ds(
            sentinel_wrap = ds_wrap,
            parsed_path = parsed_path,
            verbose = verbose,
            model = model,
            batch_size = bs
        )
    print(parsed_ds)
    '''
    with sentinel_model as s:
        s.load_only(
            loaders = loaders,
            verbose = verbose
        )
        #print(s._dss['val'][0])
        create_ds = sentinel_model.create_ds(
            path = parsed_path,
            sentinel_wrap = ds_wrap,
            verbose = verbose
        )
        parsed_ds = sentinel_model.parse_ds(
            sentinel_wrap = ds_wrap,
            parsed_path = parsed_path,
            verbose = verbose,
            model = model,
            batch_size = bs
        )
        print(parsed_ds)
    
    parsed_ds = Sentinel.parse_ds(
        sentinel_wrap = ds_wrap,
        parsed_path = parsed_path,
        verbose = verbose,
        model = model,
        batch_size = bs 
    )
    '''
    '''
    model = ModelWrap(
            model = sentinel_model,
            device = device
            )

    ds_wrap = SentinelWrap(
            path = ds_path,
            dataset = dataset,
            parsed_path = parsed_path
            )

    ds_wrap.__load_data__(
            seed = seed,
            verbose = verbose
            )
    
    sentinel = Sentinel(
        path = parsed_path
    )

    with sentinel as s:
        s.load_only(
            loaders = loaders,
            verbose = verbose
        )
        #print(s._dss['val'][0])
        create_ds = sentinel.create_ds(
            path = parsed_path,
            sentinel_wrap = ds_wrap,
            verbose = verbose
        )
        parsed_ds = sentinel.parse_ds(
                sentinel_wrap = ds_wrap,
                path = parsed_path,
                verbose = verbose,
                model = model,
                #loaders = loaders,
        )
        print(parsed_ds)
    #done parsing
    '''
