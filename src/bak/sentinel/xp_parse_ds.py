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
from peepholelib.models.sentinel.model_sentinel import CONV_AE2D

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--emb_size", required=True, type=str, help="Model type to use")
args = parser.parse_args()

emb_size = args.emb_size

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    model_path = '/srv/newpenny/SPACE/FIORIRE2_Maurizio/src/Artifacts'
    model_name = f"conv2dAE_SENT_L16_K3-3_Emb{emb_size}_Lay0_C16_S42.pth"

    ds_path = '/srv/newpenny/dataset/TASI/sentinel/sentinel_4s_clean_std'
    parsed_path = f'/srv/newpenny/XAI/generated_data/AE_sentinel/datasets_{emb_size}_RW'

    seed = 42
    dataset = Sentinel
    
    verbose = True 
    
    #--------------------------------
    # Model 
    #--------------------------------

    num_sensors = 16
    seq_len = 16
    kernel = [3, 3]
    lay3 = False   
 
    sentinel_model = CONV_AE2D(
              num_sensors = num_sensors,
              seq_len = seq_len,
              kernel_size = kernel,
              embedding_size = emb_size,
              lay3 = lay3
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
    
    #--------------------------------
    # Dataset 
    #--------------------------------

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
        model = model,
        bs = 2**11
    )
