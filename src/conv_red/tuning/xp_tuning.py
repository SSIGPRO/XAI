# python stuff
import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/conv_red').as_posix())

import pandas as pd
from statistics import geometric_mean as geomean

# torch stuff
import torch
from cuda_selector import auto_cuda

# Our stuff
import peepholelib
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.coreVectors.coreVectors import CoreVectors 
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.plots.atks import auc_atks 
from configs.common import *

# Tuner
from functools import partial
import ray
from ray import tune
from ray import train
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.train.torch import get_device as ray_get_device
import ray.cloudpickle as pickle

ray.init(runtime_env = {"py_modules": [peepholelib]})

def peephole_wrap(config, **kwargs):
    print(f'Running config: {config}') 

    model_path = kwargs['model_path']
    model_name = kwargs['model_name']
    ds = kwargs['datasets']
    cv = kwargs['corevectors']
    ph_path = kwargs['ph_path']
    ph_name = kwargs['ph_name']
    bs = kwargs['batch_size']
    verbose = kwargs['verbose']

    _device = ray_get_device() 
    # concatenate config for creating unique ph name
    _ph_name = ph_name
    for _l, _c in config.items():
        if type(_c) == dict:
            _ph_name += f'.{_l}'
            for _cn, _cv in _c.items():
                _ph_name += f'.{_cn}.{_cv}'

    #--------------------------------
    # instances
    #--------------------------------
    model = ModelWrap(
            model = Model(),
            target_modules = target_layers,
            device = _device
            )
                                            
    model.update_output(
            output_layer = output_layer,
            to_n_classes = n_classes,
            overwrite = True 
            )
                                            
    model.load_checkpoint(
            name = model_name,
            path = model_path,
            verbose = verbose
            )
    
    drillers_kwargs = get_drillers_kwargs(
            path = drill_path,
            name = drill_name,
            target_layers = target_layers,
            nl_model = n_classes,
            model = model,
            configs = config,
            act_parser = act_parser, 
            save_input = save_input,
            save_output = save_output,
            device = _device
            ) 

    # instantiate the drillers
    drillers = {}
    for _l in target_layers:
        drillers[_l] = Driller(
                **drillers_kwargs[_l],
                reducer = Reducer(
                    path = svds_path,
                    model = model,
                    layer = _l,
                    cv_dim = drillers_kwargs[_l]['n_features'],
                    verbose = verbose
                    )
                )

        if not drillers[_l].load():
            drillers[_l].fit(
                    datasets = ds,
                    corevectors = cv,
                    loader = 'CIFAR100-train',
                    verbose=verbose
                    )
            drillers[_l].save()

    peepholes = Peepholes(
            path = ph_path,
            name = _ph_name,
            device = _device
            )

    with peepholes as ph:
        ph.get_peepholes(
            datasets = ds,
            corevectors = cv,
            target_modules = target_layers,
            batch_size = bs,
            drillers = drillers,
            n_threads = 1,
            verbose = verbose 
            )

        # Evaluation
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

        train.report({
            'ood_auc': avg_auc_ood,
            'aa_auc': avg_auc_aa,
            })
    return 

if __name__ == "__main__":
    print(f'{args}') 
    use_cuda = torch.cuda.is_available()

    #--------------------------------
    # Instances 
    #--------------------------------
    dummy_model = ModelWrap(
            model = Model(),
            target_modules = target_layers,
            )

    datasets = ParsedDataset(
            path = ds_path,
            )

    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
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

        #--------------------------------
        # Tunning 
        #--------------------------------
        red_kwargs = get_reducer_kwargs(
                dummy_model._target_modules
                ) 
        param_space = reduction_param_space(
                red_kwargs
                )
        param_space = analysis_param_space(
                configs = param_space,
                args = args
                )

        # resources per trial, leave gpu=1
        if use_cuda:
            resources = {"cpu": 32, "gpu": 1}
        else:
            resources = {"cpu": 32}

        if hyper_params_file.exists():
            print('Already tunned parameters found in %s. Runing agains and appending results.'%(hyper_params_file.as_posix()))
            _prev_results_df = pd.read_pickle(hyper_params_file)
        else:
            _prev_results_df = None

        searcher = OptunaSearch(metric=['ood_auc', 'aa_auc'], mode=['max', 'max'])
        algo = ConcurrencyLimiter(searcher, max_concurrent=4)
        scheduler = AsyncHyperBandScheduler(grace_period=5, max_t=100, metric='ood_auc', mode='max') 

        trainable = tune.with_resources(
                partial(
                    peephole_wrap,
                    model_path = model_path,
                    model_name = model_name,
                    datasets = ds,
                    corevectors = cv,
                    ph_path = phs_path,
                    ph_name = phs_name,
                    batch_size = int(bs_base*bs_model_scale*bs_red_scale*bs_analysis_scale),
                    verbose = verbose
                    ),
                resources 
                )

        tuner = tune.Tuner(
                trainable,
                tune_config = tune.TuneConfig(
                    search_alg = algo,
                    num_samples = tune_num_samples, 
                    scheduler = scheduler,
                    ),
                run_config = train.RunConfig(
                    storage_path = tune_storage_path
                    ),
                param_space = param_space,
                )
        result = tuner.fit()

        results_df = result.get_dataframe()
        # remove config/ from the DF coluns names
        _cn_map = {_cn: _cn.replace('config/', '') for _cn in results_df.columns if 'config/' in _cn}
        results_df = results_df.rename(columns=_cn_map)
        if _prev_results_df is not None:
            results_df = pd.concat([_prev_results_df, results_df], ignore_index=True)
        print('results: ', results_df)

        results_df.to_pickle(hyper_params_file)
    
