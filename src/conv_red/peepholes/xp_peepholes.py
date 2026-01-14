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
from peepholelib.peepholes.peepholes import Peepholes

from configs.common import *
    
if __name__ == "__main__":
    print(f'{args}') 
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
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
    # Corevectors 
    #--------------------------------
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            model = model,
            )
    
    #--------------------------------
    # Peepholes
    #--------------------------------
    # TODO: temp for testing
    hyperps = test_configs(model._target_modules)

    # Function analysis specific kwargs for drillers
    drillers_kwargs = get_drillers_kwargs(
            path = drill_path,
            name = drill_name,
            target_layers = target_layers,
            nl_model = n_classes,
            model = model,
            configs = hyperps,
            act_parser = act_parser, 
            save_input = save_input,
            save_output = save_output,
            device = device
            ) 

    # instantiate the drillers
    drillers = {}
    for _l in target_layers:
        # instantiate with cv_size
        reducer = Reducer(
                path = svds_path,
                model = model,
                layer = _l,
                cv_dim = drillers_kwargs[_l]['n_features'],
                verbose = verbose
                ) 

        drillers[_l] = Driller(
                **drillers_kwargs[_l],
                reducer = reducer
                )

    peepholes = Peepholes(
            path = phs_path,
            name = phs_name,
            device = device
            )

    with datasets as ds, corevecs as cv: 
        ds.load_only(
                loaders = loaders,
                verbose = verbose
                )

        cv.load_only(
                loaders = loaders,
                verbose = verbose
                ) 

        for drill_key, driller in drillers.items():
            if not driller.load():
                driller.fit(
                        datasets = ds,
                        corevectors = cv,
                        loader = 'CIFAR100-train',
                        verbose=verbose
                        )
                driller.save()

        with peepholes as ph:
            ph.get_peepholes(
                    datasets = ds,
                    corevectors = cv,
                    target_modules = target_layers,
                    batch_size = int(bs_base*bs_model_scale*bs_red_scale*bs_analysis_scale),
                    drillers = drillers,
                    n_threads = n_threads,
                    verbose = verbose
                    )
