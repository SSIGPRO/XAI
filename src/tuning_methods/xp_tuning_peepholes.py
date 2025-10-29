import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
import pandas

# Our stuff
import peepholelib
from peepholelib.coreVectors.coreVectors import CoreVectors 
from peepholelib.peepholes.parsers import trim_corevectors
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.utils.analyze import compute_top_k_accuracy

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

# Load one configuration file here
if sys.argv[1] == 'vgg-cifar100':
    from config_cifar100_vgg16 import *
elif sys.argv[1] == 'vgg-cifar10':
    from config_cifar10_vgg16 import *
elif sys.argv[1] == 'vit-cifar100':
    from config_cifar100_ViT import *
elif sys.argv[1] == 'vit-cifar10':
    from config_cifar10_ViT import *
else:
    raise RuntimeError('Select a configuration by runing \'python xp_get_corevectors.py <vgg|vit>\'')
ray.init(runtime_env = {"py_modules": [peepholelib]})

def peephole_wrap(config, **kwargs):
    peep_size = config['peep_size']
    n_cls = config['n_classifier']
    cv_path = kwargs['cv_path']
    cv_name = kwargs['cv_name']
    drill_path = kwargs['drill_path']
    drill_name = kwargs['drill_name']
    ph_path = kwargs['ph_path']
    ph_name = kwargs['ph_name']
    peep_layer = kwargs['peep_layer']
    bs = kwargs['batch_size']
    verbose = kwargs['verbose']
    k = kwargs['k']

    #--------------------------------
    # CoreVectors 
    #--------------------------------
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )

    #--------------------------------
    # Peepholes
    #--------------------------------
    ph_config_name = ph_name+f'.{peep_size}.{n_cls}'
   
    with corevecs as cv: 
        # load corevds 
        cv.load_only(
                loaders = ['train', 'test', 'val'],
                verbose = verbose 
                ) 

        _device = ray_get_device() 
        print('Running trial at device: ', _device)

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
                        loaders = ['train', 'test'],
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
                    path = drill_path,
                    name = drill_name+'.'+peep_layer,
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
    #--------------------------------
    # Tunning 
    #--------------------------------

    if device == 'cpu':
        resources = {"cpu": 32}
    else:
        resources = {"cpu": 16, "gpu": 1}

    for peep_layer in target_layers:
        config = {
                'peep_size': tune.randint(min_peep_size, max_peep_size[peep_layer]+1),
                'n_classifier': tune.randint(min_n_classifier, max_n_classifier+1),
                }

        hyper_params_file = phs_path/f'hyperparams.{peep_layer}.pickle'
        
        searcher = OptunaSearch(metric='topk_acc', mode='max')
        algo = ConcurrencyLimiter(searcher, max_concurrent=4)
        scheduler = AsyncHyperBandScheduler(grace_period=5, max_t=100, metric="topk_acc", mode="max") 

        trainable = tune.with_resources(
                partial(
                    peephole_wrap,
                    cv_path = cvs_path,
                    cv_name = cvs_name,
                    drill_path = drill_path,
                    drill_name = drill_name,
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

        if hyper_params_file.exists():
            results_df = results_df._append(pandas.read_pickle(hyper_params_file), ignore_index=True)
        results_df = results_df.drop_duplicates(subset=['config/peep_size', 'config/n_classifier', 'topk_acc'])

        results_df.to_pickle(hyper_params_file)
