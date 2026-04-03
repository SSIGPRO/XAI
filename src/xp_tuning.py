import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# torch stuff
import torch
from cuda_selector import auto_cuda
from torchvision.models import vgg16

# Our stuff
import peepholelib

from peepholelib.models.model_wrap import ModelWrap 

from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.coreVectors.coreVectors import CoreVectors 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.utils.topk_acc import compute_top_k_accuracy

from peepholelib.coreVectors.dimReduction.svds.linear_svd import LinearSVD
from peepholelib.coreVectors.dimReduction.svds.conv2d_toeplitz_svd import Conv2dToeplitzSVD
from peepholelib.coreVectors.dimReduction.svds.conv2d_avg_kernel_svd import Conv2dAvgKernelSVD

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
    peep_size = config['peep_size']
    n_cls = config['n_classifier']
     
    model_path = kwargs['model_path']
    model_name = kwargs['model_name']
    ds = kwargs['datasets']
    cv = kwargs['corevectors']
    ph_path = kwargs['ph_path']
    ph_name = kwargs['ph_name']
    peep_layer = kwargs['peep_layer']
    Reducer = kwargs['reducer_class']
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
                    loaders = ['CIFAR100-train', 'CIFAR100-val'],
                    verbose = True
                    )
            
            topk_acc = compute_top_k_accuracy(
                    peepholes = ph._phs['CIFAR100-val'][peep_layer],    
                    targets = ds._dss['CIFAR100-val']['label'],
                    k=k
                    )
    else:
        #--------------------------------
        # instances
        #--------------------------------
        model = ModelWrap(
                model = vgg16(),
                target_modules = [peep_layer],
                device = _device
                )
                                                
        model.update_output(
                output_layer = 'classifier.6', 
                to_n_classes = 100,
                overwrite = True 
                )
                                                
        model.load_checkpoint(
                name = model_name,
                path = model_path,
                verbose = verbose
                )

        svd = Reducer(
                path = svds_path,
                layer = peep_layer,
                model = model,
                cv_dim = peep_size,
                device = _device
                )

        driller = tGMM(
                path = ph_path,
                name = 'classifier',
                target_module = peep_layer,
                nl_classifier = n_cls,
                nl_model = 100,
                n_features = peep_size, 
                cls_kwargs = {
                    'covariance_regularization': 1e-4,
                    'convergence_tolerance': 1e-2
                    },
                reducer = svd,
                device = _device
                )

        driller.fit(
                datasets = ds,
                corevectors = cv,
                loader = 'CIFAR100-train',
                verbose=verbose
                )
        driller.save()

        peepholes = Peepholes(
                path = ph_path,
                name = ph_config_name,
                device = _device
                )

        with peepholes as ph:
            ph.get_peepholes(
                datasets = ds,
                corevectors = cv,
                target_modules = [peep_layer],
                batch_size = bs,
                drillers = {peep_layer: driller},
                n_threads = 1,
                verbose = verbose 
                )
            
            topk_acc = compute_top_k_accuracy(
                    peepholes = ph._phs['CIFAR100-val'][peep_layer],
                    targets = ds._dss['CIFAR100-val']['label'],
                    k=k
                    )

        train.report({'topk_acc': topk_acc})

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

    ds_path = Path.cwd()/'../data/datasets'

    svds_path = Path.cwd()/'../data/svds'

    model_path = '/srv/newpenny/XAI/models'
    model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'

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
    loaders = ['CIFAR100-train', 'CIFAR100-test', 'CIFAR100-val']

    # Peepholelib
    target_layers = [
            'features.26',
            'features.28',
            'classifier.0',
            ]

    # Ray Tune
    num_samples = 2 
    red_classes = {
            'features.26': Conv2dToeplitzSVD,
            'features.28': Conv2dAvgKernelSVD,
            'classifier.0': LinearSVD,
            }

    #--------------------------------
    # Datasets and CoreVectors 
    #--------------------------------
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
        config = {
                'peep_size': tune.randint(1, 30+1),
                'n_classifier': tune.randint(1, 30+1),
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
                            model_path = model_path,
                            model_name = model_name,
                            datasets = ds,
                            corevectors = cv,
                            ph_path = phs_path,
                            ph_name = phs_name+'.'+peep_layer,
                            peep_layer = peep_layer,
                            batch_size = bs,
                            reducer_class = red_classes[peep_layer],
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
    
