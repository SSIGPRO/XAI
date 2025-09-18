import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial
from matplotlib import pyplot as plt

# Our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.datasets.transforms import vgg16_cifar100 as ds_transform 
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.models.svd_fns import linear_svd, conv2d_toeplitz_svd

from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import linear_svd_projection, conv2d_toeplitz_svd_projection

from peepholelib.peepholes.parsers import trim_corevectors
from peepholelib.peepholes.classifiers.tkmeans import KMeans as tKMeans 
from peepholelib.peepholes.classifiers.birch import Birch as Clustering 
from peepholelib.peepholes.peepholes import Peepholes

from peepholelib.utils.samplers import random_subsampling 
from peepholelib.utils.scores import conceptogram_protoclass_score as proto_score, model_confidence_score as mconf_score 
from peepholelib.utils.plots import plot_confidence, plot_calibration, plot_ood 
from peepholelib.utils.conceptograms import plot_conceptogram

# torch stuff
import torch
from torchvision.models import vgg16
from cuda_selector import auto_cuda


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    ds_path = '/srv/newpenny/dataset/CIFAR100'
    dsc_path = '/srv/newpenny/dataset/CIFAR-100-C'

    # model parameters
    dataset = 'CIFAR100' 
    seed = 29
    bs = 2**10 
    n_threads = 1

    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
    
    svds_path = Path.cwd()/'../../data3'
    svds_name = 'svds' 
    
    cvs_path = Path.cwd()/'../../data3/corevectors'
    cvs_name = 'corevectors'

    drill_path = Path.cwd()/'../../data3/drillers'
    drill_name = 'classifier'

    phs_path = Path.cwd()/'../../data3/peepholes'
    phs_name = 'peepholes'

    plots_path = Path.cwd()/'../temp_plots/toeplitz_proto/'
    
    verbose = True 
    
    # Peepholelib
    target_layers = [
            #'features.0',
            #'features.2',
            #'features.5',
            #'features.7',
            'features.10',
            'features.12',
            'features.14',
            'features.17',
            'features.19',
            'features.21',
            'features.24',
            'features.26',
            'features.28',
            'classifier.0',
            'classifier.3',
            'classifier.6',
            ]

    features_svd_rank = 200 
    classifier_svd_rank = 200
    n_cluster = 1000 
    features_cv_dim = 96 
    classifier_cv_dim = 100
    n_conceptograms = 10
    #--------------------------------
    # Dataset 
    #--------------------------------

    ds = Cifar(
            data_path = ds_path,
            dataset = dataset
            )

    ds.load_data(
            transform = ds_transform,
            corrupted_path = dsc_path,
            seed = seed,
            )
    
    #--------------------------------
    # Model 
    #--------------------------------
    
    nn = vgg16()
    n_classes = len(ds.get_classes()) 
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
    # SVDs 
    #--------------------------------
    svd_fns = {}
    for _layer in target_layers:
        if 'features' in _layer:
            svd_fns[_layer] = partial(
                    conv2d_toeplitz_svd, 
                    rank = features_svd_rank, 
                    channel_wise = False,
                    device = device,
                    )
        elif 'classifier' in _layer:
            svd_fns[_layer] = partial(
                    linear_svd,
                    rank = classifier_svd_rank,
                    device = device,
                    )

    t0 = time()
    model.get_svds(
            path = svds_path,
            name = svds_name,
            target_modules = target_layers,
            sample_in = ds._dss['train'][0][0],
            svd_fns = svd_fns,
            verbose = verbose
            )
    print('time: ', time()-t0)
    #--------------------------------
    # CoreVectors 
    #--------------------------------
    #random_subsampling(ds, 0.025)
    
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            model = model,
            )
    
    # define a dimensionality reduction function for each layer
    reduction_fns = {}
    for _layer in target_layers:
        if 'features' in _layer:
            reduction_fns[_layer] = partial(
                    conv2d_toeplitz_svd_projection, 
                    svd = model._svds[_layer], 
                    layer = model._target_modules[_layer], 
                    use_s = True,
                    device=device
                    )
        elif 'classifier' in _layer:
            reduction_fns[_layer] = partial(
                    linear_svd_projection,
                    svd = model._svds[_layer], 
                    use_s = True,
                    device=device
                    )

    with corevecs as cv: 
        cv.parse_ds(
                batch_size = bs,
                datasets = ds,
                n_threads = n_threads,
                verbose = verbose
                )

        # computing the corevectors
        cv.get_coreVectors(
                batch_size = bs,
                reduction_fns = reduction_fns,
                n_threads = n_threads,
                save_input = True,
                save_output = False,
                verbose = verbose
                )
        if not (cvs_path/(cvs_name+'.normalization.pt')).exists():
            cv.normalize_corevectors(
                    wrt = 'train',
                    to_file = cvs_path/(cvs_name+'.normalization.pt'),
                    #from_file = cvs_path/(cvs_name+'.normalization.pt'),
                    #loaders = ['ood-c0', 'ood-c1', 'ood-c2', 'ood-c3', 'ood-c4'],
                    batch_size = bs,
                    n_threads = n_threads,
                    verbose=verbose
                    )

    #--------------------------------
    # Peepholes
    #--------------------------------

    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )

    cv_parsers = {}
    for _layer in target_layers:
        if 'features' in _layer:
            cv_parsers[_layer] = partial(
                    trim_corevectors,
                    module = _layer,
                    cv_dim = features_cv_dim
                    )
        if 'classifier' in _layer:
            cv_parsers[_layer] = partial(
                    trim_corevectors,
                    module = _layer,
                    cv_dim = classifier_cv_dim
                    )
    
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
            'classifier.0': classifier_cv_dim,
            'classifier.3': classifier_cv_dim,
            'classifier.6': 100#classifier_cv_dim,
            }

    drillers = {}
    for peep_layer in target_layers:
        drillers[peep_layer] = Clustering(
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
    with corevecs as cv:
        cv.load_only(
                loaders = ['train', 'val', 'test'],
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
                        loader = 'train',
                        verbose=verbose
                        )
                print(f'Fitting time for {drill_key}  = ', time()-t0)

                driller.compute_empirical_posteriors(
                        corevectors = cv,
                        loader = 'train',
                        batch_size = bs,
                        verbose=verbose
                        )
        
                # save classifiers
                print(f'Saving classifier for {drill_key}')
                driller.save()

    with corevecs as cv, peepholes as ph:
        cv.load_only(
                loaders = ['train', 'val', 'test', 'ood-c0', 'ood-c1', 'ood-c2', 'ood-c3', 'ood-c4'],
                verbose = verbose 
                ) 

        ph.get_peepholes(
                corevectors = cv,
                target_modules = target_layers,
                batch_size = bs,
                drillers = drillers,
                n_threads = n_threads,
                verbose = verbose
                )

        # get scores
        scores, protoclasses = proto_score(
                peepholes = ph,
                corevectors = cv,
                verbose = verbose
                )
                                        
        scores = mconf_score(
                corevectors = cv,
                append_scores = scores,
                verbose = verbose
                ) 

        # make plots
        plot_confidence(
                corevectors = cv,
                scores = scores,
                loaders = ['test', 'ood-c0', 'ood-c1', 'ood-c2', 'ood-c3', 'ood-c4'],
                max_score = 1.,
                path = plots_path,
                verbose = verbose
                )

        plot_calibration(
                corevectors = cv,
                scores = scores,
                loaders = ['test', 'ood-c0', 'ood-c1', 'ood-c2', 'ood-c3', 'ood-c4'],
                calib_bin = 0.1,
                path = plots_path,
                verbose = verbose
                )

        plot_ood(
                scores = scores,
                id_loader = 'test',
                ood_loaders = ['ood-c0', 'ood-c1', 'ood-c2', 'ood-c3', 'ood-c4'],
                path = plots_path,
                verbose = verbose
                )
        quit()
        idx = [2, 5, 7, 9, 16, 17, 21, 23, 28, 29, 32, 33, 35, 37, 41, 43, 45, 48, 58, 62, 131, 319, 585, 862, 1070, 1289, 1391, 1675, 2510, 2686, 2822, 3873, 4890, 5251, 5431, 5865, 7459, 8414, 8486]
        plot_conceptogram(
                path = plots_path,
                name = 'conceptogram',
                corevectors = cv,
                peepholes = ph,
                loaders = ['test'],
                samples = idx,
                target_modules = target_layers,
                classes = ds._classes,
                protoclasses = protoclasses,
                scores = scores,
                verbose = verbose,
                )
        
