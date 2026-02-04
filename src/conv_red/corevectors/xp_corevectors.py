# python stuff
import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/conv_red').as_posix())

from filelock import FileLock

# torch stuff
import torch
from cuda_selector import auto_cuda

# Peephoelib stuff
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.coreVectors.coreVectors import CoreVectors

from configs.common import *
    
if __name__ == "__main__":
    print(f'{args}') 
    lock_file = '../locks/corevectors.cuda.lock'
    lock = FileLock(lock_file)
    with lock.acquire(timeout=-1):
        use_cuda = torch.cuda.is_available()
        device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
        print(f"Using {device} device")

        #------------------
        # Model 
        #------------------
        model = ModelWrap(
                model = Model(),
                target_modules = target_layers,
                device = device
                )
                                            
    model.update_output(
            output_layer = output_layer, 
            to_n_classes = n_classes,
            overwrite = True 
            )
                                            
    model.load_checkpoint(
            path = model_path,
            name = model_name,
            verbose = True 
            )

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
                path = svds_path,
                model = model,
                layer = _layer,
                sample_in = sample_in,
                **reducers_kwargs[_layer],
                verbose = verbose
                ) 

    #--------------------------------
    # Corevectors 
    #--------------------------------
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            model = model,
            )
    
    with datasets as ds, corevecs as cv: 
        ds.load_only(
                loaders = loaders,
                verbose = verbose
                )

        # computing the corevectors
        cv.get_coreVectors(
                datasets = ds,
                reducers = reducers,
                activations_parser = act_parser,
                save_input = save_input,
                save_output = save_output,
                batch_size = int(bs_base*bs_model_scale*bs_red_scale),
                n_threads = n_threads,
                verbose = verbose
                )

        if not (cvs_path/(cvs_name+'.normalization.pt')).exists():
            cv.normalize_corevectors(
                    wrt = 'CIFAR100-train',
                    to_file = cvs_path/(cvs_name+'.normalization.pt'),
                    #from_file = cvs_path/(cvs_name+'.normalization.pt'),
                    #loaders = ['CIFAR100-val', 'CIFAR100-test'],
                    batch_size = int(bs_base*bs_red_scale),
                    n_threads = n_threads,
                    verbose=verbose
                    )
