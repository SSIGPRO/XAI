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

    dataset = ParsedDataset(
            path = ds_path,
            )

    #--------------------------------
    # Model 
    #--------------------------------
    model = ModelWrap(
            model = Model(),
            device = device
            )

    model.update_output(
            output_layer = output_layer, 
            to_n_classes = n_classes,
            overwrite = True 
            )

    model.load_checkpoint(
            name = model_name,
            path = model_dir,
            verbose = verbose
            )

    model.set_target_modules(
            target_modules = target_layers,
            verbose = verbose
            )
    
    #--------------------------------
    # SVDs 
    #--------------------------------
    with dataset as ds: 
        ds.load_only(
                loaders = loaders,
                verbose = verbose
                )

        model.get_svds(
                path = svds_path,
                name = svds_name,
                target_modules = target_layers,
                sample_in = ds._dss['CIFAR100-train']['image'][0],
                svd_fns = svd_fns,
                verbose = verbose
                )

        #--------------------------------
        # CoreVectors 
        #--------------------------------
        
        corevecs = CoreVectors(
                path = cvs_path,
                name = cvs_name,
                model = model,
                )
        
        with corevecs as cv: 
            # add svd to reduction_fns
            for _layer in reduction_fns:
                if 'features' in _layer:
                    reduction_fns[_layer].keywords['layer'] = model._target_modules[_layer]
                reduction_fns[_layer].keywords['svd'] = model._svds[_layer] 
            
            # computing the corevectors
            cv.get_coreVectors(
                    datasets = ds,
                    reduction_fns = reduction_fns,
                    save_input = save_input,
                    save_output = save_output,
                    batch_size = bs,
                    n_threads = n_threads,
                    verbose = verbose
                    )

            if not (cvs_path/(cvs_name+'.normalization.pt')).exists():
                cv.normalize_corevectors(
                        wrt = 'CIFAR100-train',
                        to_file = cvs_path/(cvs_name+'.normalization.pt'),
                        #from_file = cvs_path/(cvs_name+'.normalization.pt'),
                        #loaders = ['SVHN-test', 'SVHN-val'],
                        batch_size = bs,
                        n_threads = n_threads,
                        verbose=verbose
                        )
