import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/SAM').as_posix())

# python stuff
from functools import partial

# torch stuff
import torch
from cuda_selector import auto_cuda

###### Our stuff
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.datasets.functional.samplers import random_subsampling 
from peepholelib.datasets.functional.inference_fns import img_classification_full as inference_fn 

from configs.common import *

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    '''
    just for testing
    '''
    dss_samplers = {
            k: partial(
                random_subsampling, 
                perc = 0.0003
                ) for k in dss.keys()
            }

    model = ModelWrap(
                    model = Model(weights=weights),
                    device = device
                    )

    model.normalize_model(mean=means['ImageNet'], std= stds['ImageNet'])

    dataset = ParsedDataset.parse_ds(
            path = ds_path,
            dataset_wraps = dss,
            ds_samplers = dss_samplers, 
            keys_to_copy = ['image', 'label'],
            inference_fn = partial(inference_fn, model=model), # comment for fine tuning the model
            batch_size = bs_base,
            n_threads = 1,
            verbose = verbose
            ) 
    
