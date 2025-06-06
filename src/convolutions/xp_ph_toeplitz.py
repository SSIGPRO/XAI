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
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes

from peepholelib.utils.samplers import random_subsampling 
from peepholelib.utils.analyze import evaluate_dists, conceptogram_ghl_score, conceptogram_cl_score
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

    # model parameters
    dataset = 'CIFAR100' 
    seed = 29
    bs = 512 
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
    
    verbose = True 
    
    # Peepholelib
    target_layers = [
            'features.24',
            'features.26',
            'features.28',
            'classifier.0',
            'classifier.3',
            'classifier.6',
            ]
    
    svd_rank = 200
    n_cluster = 200
    features_cv_dim = 196
    classifier_cv_dim = 150
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
    svd_fns = {
            'features.24': partial(
                conv2d_toeplitz_svd, 
                rank = svd_rank,
                channel_wise = False,
                device = device,
                ),
            'features.26': partial(
                conv2d_toeplitz_svd, 
                rank = svd_rank,
                channel_wise = False,
                device = device,
                ),
            'features.28': partial(
                conv2d_toeplitz_svd, 
                rank = svd_rank,
                channel_wise = False,
                device = device,
                ),
            'classifier.0': partial(
                linear_svd,
                device = device,
                ),
            'classifier.3': partial(
                linear_svd,
                device = device,
                ),
            'classifier.6': partial(
                linear_svd,
                device = device,
                ),
            }

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
    reduction_fns = {
            'features.24': partial(
                conv2d_toeplitz_svd_projection, 
                svd = model._svds['features.24'], 
                layer = model._target_modules['features.24'], 
                device=device
                ),
            'features.26': partial(
                conv2d_toeplitz_svd_projection, 
                svd = model._svds['features.26'], 
                layer = model._target_modules['features.26'], 
                device = device
                ),
            'features.28': partial(
                conv2d_toeplitz_svd_projection, 
                svd = model._svds['features.28'], 
                layer = model._target_modules['features.28'], 
                device = device
                ),
            'classifier.0': partial(
                linear_svd_projection,
                svd = model._svds['classifier.0'], 
                device=device
                ),
            'classifier.3': partial(
                linear_svd_projection,
                svd = model._svds['classifier.3'], 
                device=device
                ),
            'classifier.6': partial(
                linear_svd_projection,
                svd = model._svds['classifier.6'], 
                device=device
                ),
            }

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
                    #from_file = cvs_path/(cvs_name+'.normalization.pt'),
                    to_file = cvs_path/(cvs_name+'.normalization.pt'),
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

    cv_parsers = {
            'features.24': partial(
                trim_corevectors,
                module = 'features.24',
                cv_dim = features_cv_dim
                ),
            'features.26': partial(
                trim_corevectors,
                module = 'features.26',
                cv_dim = features_cv_dim
                ),
            'features.28': partial(
                trim_corevectors,
                module = 'features.28',
                cv_dim = features_cv_dim
                ),
            'classifier.0': partial(
                trim_corevectors,#
                module = 'classifier.0',
                cv_dim = classifier_cv_dim
                ),
            'classifier.3': partial(
                trim_corevectors,
                module = 'classifier.3',
                cv_dim = classifier_cv_dim
                ),
            'classifier.6': partial(
                trim_corevectors,
                module = 'classifier.6',
                cv_dim = classifier_cv_dim
                ),
            }

    feature_sizes = {
            'features.24': features_cv_dim,
            'features.26': features_cv_dim,
            'features.28': features_cv_dim,
            'classifier.0': classifier_cv_dim,
            'classifier.3': classifier_cv_dim,
            'classifier.6': 100#classifier_cv_dim,
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
    with corevecs as cv:
        cv.load_only(
                loaders = ['train', 'test', 'val'],
                verbose = True
                ) 

        for drill_key, driller in drillers.items():
            if (driller._empp_file).exists():
                print(f'Loading Classifier for {drill_key}') 
                driller.load()
            else:
                t0 = time()
                print(f'Fitting classifier for {drill_key}')
                driller.fit(corevectors = cv._corevds['train'], verbose=verbose)
                print(f'Fitting time for {drill_key}  = ', time()-t0)

                driller.compute_empirical_posteriors(
                        dataset = cv._dss['train'],
                        corevectors = cv._corevds['train'],
                        batch_size = bs,
                        verbose=verbose
                        )
        
                # save classifiers
                print(f'Saving classifier for {drill_key}')
                driller.save()

    with corevecs as cv, peepholes as ph:
        cv.load_only(
                loaders = ['train', 'test', 'val'],
                verbose = True
                ) 

        ph.get_peepholes(
                corevectors = cv,
                target_modules = target_layers,
                batch_size = bs,
                drillers = drillers,
                n_threads = n_threads,
                verbose = verbose
                )

        ph.get_scores(
            batch_size = bs,
            verbose=verbose
            )

        scores, _, _, _, _ = conceptogram_cl_score(
                peepholes = ph,
                corevectors = cv,
                loaders = ['train', 'val', 'test'],
                basis = 'from_output',
                weights = [1, 1, 1, 1, 1, 1],
                bins = 50,
                plot = True,
                verbose = verbose
                )
        
        idx_hh = (scores > 0.90).nonzero().squeeze()[:10]
        idx_mh = (torch.logical_and(scores>0.7, scores<0.8)).nonzero().squeeze()[:10]
        idx_mm = (torch.logical_and(scores>0.45, scores<0.55)).nonzero().squeeze()[:10]
        idx_ml = (torch.logical_and(scores>0.2, scores<0.3)).nonzero().squeeze()[:10]
        idx_ll = (scores<0.1).nonzero().squeeze()[:10]
        # get `n_conceptograms` random samples for each score interval
        idx = torch.hstack([
            idx_hh[torch.randperm(idx_hh.shape[0])[:n_conceptograms]],
            idx_mh[torch.randperm(idx_mh.shape[0])[:n_conceptograms]],
            idx_mm[torch.randperm(idx_mm.shape[0])[:n_conceptograms]],
            idx_ml[torch.randperm(idx_ml.shape[0])[:n_conceptograms]],
            idx_ll[torch.randperm(idx_ll.shape[0])[:n_conceptograms]]
            ]).numpy()  

        plot_conceptogram(
                path = phs_path,
                name = phs_name,
                corevectors = cv,
                peepholes = ph,
                portion = 'test',
                samples = idx,
                target_layers = target_layers,
                classes = ds._classes,
                alt_score = scores[idx],
                alt_score_name = 'CL score',
                verbose = verbose,
                )
