import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
import numpy as np

# Our stuff
import peepholelib
from peepholelib.coreVectors.coreVectors import CoreVectors 
from peepholelib.peepholes.parsers import trim_corevectors
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.utils.analyze import compute_top_k_accuracy

# torch stuff
import torch
from cuda_selector import auto_cuda

# Tuner
import tempfile
from functools import partial
import ray
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.train.torch import get_device as ray_get_device
import ray.cloudpickle as pickle

ray.init(runtime_env = {"py_modules": [peepholelib]})

def peephole_wrap(config, **kwargs):
    peep_size = config['peep_size']
    n_cls = config['n_classifier']
    
    cv = kwargs['corevectors']
    ph_path = kwargs['ph_path']
    ph_name = kwargs['ph_name']
    peep_layer = kwargs['peep_layer']
    bs = kwargs['batch_size']
    verbose = kwargs['verbose']
    k = kwargs['k']
    #--------------------------------
    # Peepholes
    #--------------------------------
    ph_config_name = ph_name+f'.{peep_size}.{n_cls}'
    
    _device = ray_get_device() 
    g = list(ph_path.glob(f'{ph_config_name}.*')) 
    if len(g) > 0:
        print('Already ran this configuration, skipping peepholes computation')
        peepholes = Peepholes(
                path = ph_path,
                name = ph_config_name,
                device = _device
                )

        with peepholes as ph: 
            ph.load_only(
                    loaders = ['train', 'test', 'val'],
                    verbose = True
                    )
            
            topk_acc = compute_top_k_accuracy(
                    peepholes = ph._phs['val'][peep_layer]['peepholes'],    
                    targets = cv._dss['val']['label'],
                    k=k
                    )
    else:
        parser_cv = partial(
                trim_corevectors,
                module = peep_layer,
                cv_dim = peep_size,
                ) 

        driller = tGMM(
                path = ph_path,
                name = 'classifier',
                nl_classifier = n_cls,
                nl_model = 100,
                n_features = peep_size, 
                parser = parser_cv,
                device = _device
                )
                                                                         
        driller.fit(corevectors = cv._corevds['train'], verbose=verbose)
        driller.compute_empirical_posteriors(
                dataset = cv._dss['train'],
                corevectors = cv._corevds['train'],
                bs = bs,
                verbose=verbose,
                )
        driller.save()

        peepholes = Peepholes(
                path = ph_path,
                name = ph_config_name,
                device = _device
                )

        with peepholes as ph:
            ph.get_peepholes(
                corevectors = cv,
                target_modules = [peep_layer],
                batch_size = bs,
                drillers = {peep_layer: driller},
                n_threads = 1,
                verbose = verbose 
                )
            
            topk_acc = compute_top_k_accuracy(
                    peepholes = ph._phs['val'][peep_layer]['peepholes'],
                    targets = cv._dss['val']['label'],
                    k=k
                    )

    # Final report: no checkpoint anymore
    with tempfile.TemporaryDirectory() as tempdir: 
        checkpoint = Checkpoint.from_directory(tempdir)
        train.report({'topk_acc': topk_acc}, checkpoint=checkpoint)

    return 

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------

    # parameters
    bs = 2**9 
    n_threads = 1

    cvs_name = 'corevectors'
    cvs_path = Path.cwd()/'../data/corevectors'
    
    drill_path = Path.cwd()/'../data/drillers'
    drill_name = 'classifier'

    phs_path = Path.cwd()/'../data/peepholes'
    phs_name = 'peepholes'

    tune_storage_path = Path.cwd()/'../data/tuning'

    verbose = True 
    
    corr_path = Path.cwd()/'temp_plots/correlations'
    corr_path.mkdir(parents=True, exist_ok=True)

    # Peepholelib
    target_layers = [
            'features.24',
            'features.26',
            'features.28',
            'classifier.0',
            'classifier.3',
            'classifier.6',
            ]

    # Ray Tune
    num_samples = 50

    #--------------------------------
    # CoreVectors 
    #--------------------------------
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )

    with corevecs as cv: 
        # load corevds 
        cv.load_only(
                loaders = ['train', 'test', 'val'],
                verbose = verbose 
                ) 

        #--------------------------------
        # Tunning 
        #--------------------------------

        config = {
                'peep_size': tune.randint(1, 300+1),
                'n_classifier': tune.randint(1, 300+1),
                }

        if device == 'cpu':
            resources = {"cpu": 32}
        else:
            resources = {"cpu": 32, "gpu": 4}

        for peep_layer in target_layers:
            hyper_params_file = phs_path/f'hyperparams.{peep_layer}.pickle'
            if hyper_params_file.exists():
                print("Already tunned parameters fount in %s. Skipping"%(hyper_params_file.as_posix()))
            else: 
                searcher = OptunaSearch(metric='topk_acc', mode='max')
                algo = ConcurrencyLimiter(searcher, max_concurrent=4)
                scheduler = AsyncHyperBandScheduler(grace_period=5, max_t=100, metric="topk_acc", mode="max") 

                trainable = tune.with_resources(
                        partial(
                            peephole_wrap,
                            corevectors = cv,
                            ph_path = phs_path,
                            ph_name = phs_name+'.'+peep_layer,
                            peep_layer = peep_layer,
                            batch_size = bs,
                            k = 3,
                            verbose = verbose
                            ),
                        resources 
                        )

                tuner = tune.Tuner(
                        trainable,
                        tune_config = tune.TuneConfig(
                            search_alg = algo,
                            num_samples = num_samples, 
                            scheduler = scheduler,
                            ),
                        run_config = train.RunConfig(
                            storage_path = tune_storage_path
                            ),
                        param_space = config,
                        )
                result = tuner.fit()

                results_df = result.get_dataframe()
                print('results: ', results_df)
                results_df.to_pickle(hyper_params_file)
    
