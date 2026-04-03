# python stuff
import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/conv_red').as_posix())

from statistics import geometric_mean as geomean
from filelock import FileLock

# torch stuff
import torch
from cuda_selector import auto_cuda

# Peepholelib stuff
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.plots.atks import auc_atks 

from configs.common import *
    
if __name__ == "__main__":
    print(f'{args}') 
    # TODO: find an way to lock cpus for multiprocesses make
    lock_file = '../locks/peepholes.cuda.lock'
    lock = FileLock(lock_file)
        use_cuda = torch.cuda.is_available()
        device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
        print(f"Using {device} device")

        bs = int(bs_base*bs_model_scale*bs_red_scale*bs_analysis_scale)

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
            verbose = verbose 
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
    # testing values
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
            name = phs_name+'.test',
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
                    batch_size = bs,
                    drillers = drillers,
                    n_threads = n_threads,
                    verbose = verbose
                    )
            
            scores = {}
            for score_name, score_fn in score_fns.items():
                scores = score_fn(
                        datasets = ds,
                        peepholes = ph,
                        score_name = score_name,
                        batch_size = bs,
                        target_modules = target_layers,
                        append_scores = scores,
                        verbose = verbose
                        )
                if type(scores) == tuple: scores = scores[0]

            aucs_ood = auc_atks(
                    datasets = ds,
                    scores = scores,
                    **auc_kwargs_ood,
                    verbose = verbose
                    )

            aucs_aa = auc_atks(
                    datasets = ds,
                    scores = scores,
                    **auc_kwargs_aa,
                    verbose = verbose
                    )

            _aucs = [list(aucs_ood[d].values())[0] for d in auc_kwargs_ood['atk_loaders']] 
            avg_auc_ood = geomean(_aucs)

            _aucs = [list(aucs_aa[d].values())[0] for d in auc_kwargs_aa['atk_loaders']] 
            avg_auc_aa = geomean(_aucs) 
            print('geom auc OOD: ', avg_auc_ood, 'geom auch AA', avg_auc_aa)
