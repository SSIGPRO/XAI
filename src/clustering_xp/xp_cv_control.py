import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial

# torch stuff
import torch
from torchvision.models import vgg16
from cuda_selector import auto_cuda

###### Our stuff

# Model
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.models.svd_fns import linear_svd, conv2d_toeplitz_svd

# datasets
from peepholelib.datasets.cifar100 import Cifar100
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.datasets.functional.parsers import from_dataset
from peepholelib.datasets.functional.transforms import vgg16_cifar100 as ds_transform 
from peepholelib.datasets.functional.samplers import random_subsampling 

# corevecs
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import linear_svd_projection, conv2d_toeplitz_svd_projection

# peepholes
from peepholelib.peepholes.parsers import trim_corevectors
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.utils.viz_empp import *


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    torch.cuda.empty_cache()

    #device = torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    cifar_path = '/srv/newpenny/dataset/CIFAR100'
    ds_path = Path.cwd()/'../data/datasets'

    # model parameters
    seed = 29
    bs = 64
    n_threads = 1

    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
     
    svds_path = Path.cwd()/'../data'
    svds_name = 'svds' 
    
    cvs_path = Path.cwd()/'../data/corevectors'
    cvs_name = 'corevectors'

    drill_path = Path.cwd()/'../data/drillers'
    drill_name = 'classifier'

    phs_path = Path.cwd()/'../data/peepholes'
    phs_name = 'peepholes'

    plots_path = Path.cwd()/'temp_plots/coverage/'
    
    verbose = True 
    
    # Peepholelib
    target_layers = [ 'features.0', 'features.2', 'features.5', 'features.7', 'features.10', 'features.12', 'features.14', 'features.17', 'features.19', 'features.21',
                        'features.24','features.26','features.28','classifier.0','classifier.3', 'classifier.6',
                    ]
    features24_svd_rank = 100 
    features26_svd_rank = 100 
    features28_svd_rank = 100
    classifier_svd_rank = 100 
    n_cluster = 100
    features_cv_dim = 100

    n_conceptograms = 2 
    
    loaders = [
            'CIFAR100-train', 'CIFAR100-val', 'CIFAR100-test', 
            ]

    #--------------------------------
    # Model 
    #--------------------------------
    
    nn = vgg16()
    n_classes = len(Cifar100.get_classes(meta_path = Path(cifar_path)/'cifar-100-python/meta')) 
    print("dict keys: ", nn.state_dict().keys())

    model = ModelWrap(
            model = nn,
            device = device
            )
                                            
    model.update_output(
            output_layer = 'classifier.6', 
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
    # Datasets 
    #--------------------------------
    
    # Assuming we have a parsed dataset in ds_path
    datasets = ParsedDataset(
            path = ds_path,
            )
    #--------------------------------
    # SVDs 
    #--------------------------------
    svd_fns = {
        'features.0': partial(conv2d_toeplitz_svd, 
                rank = features24_svd_rank,
                device = device,
        ),
        'features.2': partial(conv2d_toeplitz_svd, 
                rank = features24_svd_rank,
                device = device,
        ),
        'features.5': partial(conv2d_toeplitz_svd, 
                rank = features24_svd_rank,
                device = device,
        ),
        'features.7': partial(conv2d_toeplitz_svd, 
                rank = features24_svd_rank,
                device = device,
        ),
        'features.10': partial(conv2d_toeplitz_svd, 
                rank = features24_svd_rank,
                device = device,
        ),
        'features.12': partial(conv2d_toeplitz_svd, 
                rank = features24_svd_rank,
                device = device,
        ),
        'features.14': partial(conv2d_toeplitz_svd, 
                rank = features24_svd_rank,
                device = device,
        ),
        'features.17': partial(conv2d_toeplitz_svd, 
                rank = features24_svd_rank,
                device = device,
        ),
        'features.19': partial(conv2d_toeplitz_svd, 
                rank = features24_svd_rank,
                device = device,
        ),
        'features.21': partial(conv2d_toeplitz_svd, 
                rank = features24_svd_rank,
                device = device,
        ),
        'features.24': partial(conv2d_toeplitz_svd,   
                rank = features24_svd_rank,
                channel_wise = False,
                device = device,
        ),
        'features.26': partial(conv2d_toeplitz_svd, 
                rank = features26_svd_rank,
                channel_wise = False,
                device = device,
        ),
        'features.28': partial(conv2d_toeplitz_svd, 
                rank = features26_svd_rank,
                channel_wise = False,
                device = device,
        ),
        'classifier.0': partial(linear_svd,
                rank = classifier_svd_rank,
                device = device,
        ),
        'classifier.3': partial(linear_svd,
                rank = classifier_svd_rank,
                device = device,
        ),
        'classifier.6': partial(linear_svd,
                rank = classifier_svd_rank,
                device = device,
        ),
        }

#     t0 = time()

    with datasets as ds:
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
#     print('time: ', time()-t0)

    #--------------------------------
    # CoreVectors 
    #--------------------------------
    
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            model = model,
            )
    
    # define a dimensionality reduction function for each layer
    reduction_fns = {
        'features.0': partial(conv2d_toeplitz_svd_projection, 
                svd = model._svds['features.0'], 
                layer = model._target_modules['features.0'], 
                use_s = True,
                device=device
                ),
        'features.2': partial(conv2d_toeplitz_svd_projection, 
                svd = model._svds['features.2'], 
                layer = model._target_modules['features.2'], 
                use_s = True,
                device=device
                ),
        'features.5': partial(conv2d_toeplitz_svd_projection, 
                svd = model._svds['features.5'], 
                layer = model._target_modules['features.5'], 
                use_s = True,
                device=device
                ),
        'features.7': partial(conv2d_toeplitz_svd_projection, 
                svd = model._svds['features.7'], 
                layer = model._target_modules['features.7'], 
                use_s = True,
                device=device
                ),
        'features.10': partial(conv2d_toeplitz_svd_projection, 
                svd = model._svds['features.10'], 
                layer = model._target_modules['features.10'], 
                use_s = True,
                device=device
                ),
        'features.12': partial(conv2d_toeplitz_svd_projection, 
                svd = model._svds['features.12'], 
                layer = model._target_modules['features.12'], 
                use_s = True,
                device=device
                ),
        'features.14': partial(conv2d_toeplitz_svd_projection,
                svd = model._svds['features.14'], 
                layer = model._target_modules['features.14'], 
                use_s = True,
                device=device
                ),
        'features.17': partial(conv2d_toeplitz_svd_projection,
                svd = model._svds['features.17'], 
                layer = model._target_modules['features.17'], 
                use_s = True,
                device=device
                ),
        'features.19': partial(conv2d_toeplitz_svd_projection,
                svd = model._svds['features.19'], 
                layer = model._target_modules['features.19'], 
                use_s = True,
                device=device
                ),
        'features.21': partial(conv2d_toeplitz_svd_projection,
                svd = model._svds['features.21'], 
                layer = model._target_modules['features.21'], 
                use_s = True,
                device=device
                ),
        'features.24': partial(conv2d_toeplitz_svd_projection,   
                svd = model._svds['features.24'], 
                layer = model._target_modules['features.24'], 
                use_s = True,
                device = device
        ),
        'features.26': partial(conv2d_toeplitz_svd_projection, 
                svd = model._svds['features.26'], 
                layer = model._target_modules['features.26'], 
                use_s = True,
                device = device
                ),
        'features.28': partial(conv2d_toeplitz_svd_projection, 
            svd = model._svds['features.28'], 
            layer = model._target_modules['features.28'], 
            use_s = True,
            device = device
            ),
        'classifier.0': partial(linear_svd_projection,
            svd = model._svds['classifier.0'], 
            use_s = True,
            device=device
            ),
        'classifier.3': partial(linear_svd_projection,
            svd = model._svds['classifier.3'], 
            use_s = True,
            device=device
            ),
        'classifier.6': partial(linear_svd_projection,
            svd = model._svds['classifier.6'], 
            use_s = True,
            device=device
            ),
        }

    with datasets as ds, corevecs as cv: 
        ds.load_only(
                loaders = loaders,
                verbose = verbose
                )

        # computing the corevectors
        cv.get_coreVectors(
                datasets = ds,
                reduction_fns = reduction_fns,
                save_input = True,
                save_output = False,
                batch_size = bs,
                n_threads = n_threads,
                verbose = verbose
                )

        if not (cvs_path/(cvs_name+'.normalization.pt')).exists():
            cv.normalize_corevectors(
                    wrt = 'CIFAR100-train',
                    to_file = cvs_path/(cvs_name+'.normalization.pt'),
                    batch_size = bs,
                    n_threads = n_threads,
                    verbose=verbose
                    )

    #--------------------------------
    # Peepholes
    #--------------------------------

    cv_parsers = {
            'features.0': partial(trim_corevectors,
                module = 'features.0',
                cv_dim = features_cv_dim
                ),
            'features.2': partial(trim_corevectors,
                module = 'features.2',
                cv_dim = features_cv_dim
                ),
            'features.5': partial(trim_corevectors,
                module = 'features.5',
                cv_dim = features_cv_dim
                ),
            'features.7': partial(trim_corevectors,
                module = 'features.7',
                cv_dim = features_cv_dim
                ),
            'features.10': partial(trim_corevectors,
                module = 'features.10',
                cv_dim = features_cv_dim
                ),
            'features.12': partial(trim_corevectors,
                module = 'features.12',
                cv_dim = features_cv_dim
                ),
            'features.14': partial(trim_corevectors,
                module = 'features.14',
                cv_dim = features_cv_dim
                ),
            'features.17': partial(trim_corevectors,
                module = 'features.17',
                cv_dim = features_cv_dim
                ),
            'features.19': partial(trim_corevectors,
                module = 'features.19',
                cv_dim = features_cv_dim
                ),
            'features.21': partial(trim_corevectors,
                module = 'features.21',
                cv_dim = features_cv_dim
                ),
            'features.24': partial(trim_corevectors,
                module = 'features.24',
                cv_dim = features_cv_dim
                ),
            'features.26': partial(trim_corevectors,
                module = 'features.26',
                cv_dim = features_cv_dim
                ),
            'features.28': partial(trim_corevectors,
                module = 'features.28',
                cv_dim = features_cv_dim
                ),
            'classifier.0': partial(trim_corevectors,
                module = 'classifier.0',
                cv_dim = features_cv_dim
                ),
            'classifier.3': partial(trim_corevectors,
                module = 'classifier.3',
                cv_dim = features_cv_dim
                ),
            'classifier.6': partial(trim_corevectors,
                module = 'classifier.6',
                cv_dim = features_cv_dim
                ),
            }

    feature_sizes = {
        'features.0': features_cv_dim,
        'features.2': features_cv_dim,
        'features.5': features_cv_dim,
        'features.7': features_cv_dim,
        'features.10': features_cv_dim,
        'features.12': features_cv_dim,
        'features.14': features_cv_dim,
        'features.17': features_cv_dim,
        'features.19': features_cv_dim,
        'features.21': features_cv_dim,   
        'features.24': features_cv_dim,
        'features.26': features_cv_dim,
        'features.28': features_cv_dim,
        'classifier.0': features_cv_dim,
        'classifier.3': features_cv_dim,
        'classifier.6': features_cv_dim,
        }

    drillers = {}
    for peep_layer in target_layers:
        drillers[peep_layer] = tGMM(
                path = drill_path,
                name = drill_name+'.'+peep_layer,
                nl_classifier = n_cluster,
                nl_model = n_classes,
                n_features = feature_sizes[peep_layer],
                parser = cv_parsers[peep_layer],
                device = device
                )

    peepholes = Peepholes(
            path = phs_path,
            name = phs_name,
            device = device
            )

    # fitting classifiers
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
            if (driller._empp_file).exists():
                print(f'Loading Classifier for {drill_key}') 
                driller.load()
            else:
                t0 = time()
                print(f'Fitting classifier for {drill_key}')
                driller.fit(
                        corevectors = cv,
                        loader = 'CIFAR100-train',
                        verbose=verbose
                        )
                print(f'Fitting time for {drill_key}  = ', time()-t0)

                driller.compute_empirical_posteriors(
                        datasets = ds,
                        corevectors = cv,
                        loader = 'CIFAR100-train',
                        batch_size = bs,
                        verbose=verbose
                        )
        
                # save classifiers
                print(f'Saving classifier for {drill_key}')
                driller.save()

    with datasets as ds, corevecs as cv, peepholes as ph:
        ds.load_only(
                loaders = loaders,
                verbose = verbose
                )

        cv.load_only(
                loaders = loaders,
                verbose = verbose 
                ) 

        ph.get_peepholes(
                datasets = ds,
                corevectors = cv,
                target_modules = target_layers,
                batch_size = bs,
                drillers = drillers,
                n_threads = n_threads,
                verbose = verbose
                )
        
        coverage = empp_coverage_scores(drillers=ph._drillers, threshold=0.8, plot=True, save_path='/home/claranunesbarrancos/repos/XAI/src/clustering_xp/temp_plots')
