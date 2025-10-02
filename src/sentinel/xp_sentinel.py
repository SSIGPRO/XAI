import sys

import torch
from cuda_selector import auto_cuda
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'newpenny/XAI/EP').as_posix())

# Our stuff
from peepholelib.datasets.sentinel import Sentinel
from peepholelib.datasets.sentinel import SentinelWrap
from peepholelib.models.model_wrap import ModelWrap

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
    seed = 42
    dataset = Sentinel
    
    verbose = True 
    #model = "ae1Dregn16_2_ns0_k001.pth"

    model_dir = '/srv/newpenny/XAI/EP/Model_ready/ae1Dregn16_2_ns0_k001.pth'
    
    #--------------------------------
    # Dataset 
    #--------------------------------
    
    sentinel_model = torch.load("/srv/newpenny/XAI/EP/Model_ready/ae1Dregn16_2_ns0_k001.pth",map_location=device, weights_only=False)
    #print(sentinel_model)
    #print(next(sentinel_model.parameters()).dtype)
    #print(self.data.dtype)
       
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
    
    parsed_ds = Sentinel.parse_ds(
        sentinel_wrap = ds_wrap,
        parsed_path = parsed_path,
        verbose = verbose,
        model = model
    )
    print(parsed_ds)
    #done parsing

