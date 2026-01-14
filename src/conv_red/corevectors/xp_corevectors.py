# python stuff
import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/conv_red').as_posix())

# torch stuff
import torch
from cuda_selector import auto_cuda

# Our stuff
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.coreVectors.coreVectors import CoreVectors

from configs.common import *
    
if __name__ == "__main__":
    print(f'{args}') 
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Dataset 
    #--------------------------------

    datasets = ParsedDataset(
            path = ds_path,
            )

    #--------------------------------
    # Reducers 
    #--------------------------------
    with datasets as ds:
        ds.load_only(
                loaders = ['CIFAR100-train'],
                verbose = verbose
                )
        sample_in = ds._dss['CIFAR100-train']['image'][0]

    reducers_kwargs = get_reducer_kwargs(model._target_modules) 
    reducers = {} 
    for _layer in target_layers:
        reducers[_layer] = Reducer(
                path = svds_path/args.reduction,
                model = model,
                layer = _layer,
                sample_in = sample_in,
                **reducers_kwargs[_layer],
                verbose = verbose
                ) 
