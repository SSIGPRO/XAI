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
from peepholelib.models.get_rand_proj import get_random_projs

# datasets
from peepholelib.datasets.cifar100 import Cifar100
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.datasets.functional.parsers import from_dataset
from peepholelib.datasets.functional.transforms import vgg16_cifar100 as ds_transform 

# corevecs
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import linear_svd_projection, conv2d_toeplitz_svd_projection
from peepholelib.coreVectors.dimReduction.rand_projection import random_projection

# peepholes
from peepholelib.peepholes.parsers import trim_corevectors
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.models.viz import viz_singular_values_2
from peepholelib.utils.viz_empp import *


if __name__ == "__main__":
        use_cuda = torch.cuda.is_available()
        device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
        #device = torch.device('cuda:1') 
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
        bs = 512
        n_threads = 1

        model_dir = Path('/srv/newpenny/XAI/models')
        model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
        
        proj_path = Path('/srv/newpenny/XAI/CN/vgg_data_rand_proj2')
        proj_name = 'rand_projections' 
        
        cvs_path = Path('/srv/newpenny/XAI/CN/vgg_data_rand_proj2/corevectors')
        cvs_name = 'corevectors'

        drill_path = Path('/srv/newpenny/XAI/CN/vgg_data_rand_proj2/drillers_all/drillers_200')
        drill_name = 'classifier'

        phs_path =  Path('/srv/newpenny/XAI/CN/vgg_data_rand_proj2/peepholes')
        phs_name = 'peepholes'

        plots_path = Path.cwd()/'temp_plots/coverage/'
        
        verbose = True 
        
        # Peepholelib
        target_layers = [ 'features.7', 'features.10', 'features.12', 'features.14', 'features.17', 'features.19', 'features.21',
                                'features.24','features.26','features.28','classifier.0','classifier.3', 
                                'classifier.6',
                        ]

        n_cluster = 200     
        loaders = ['CIFAR100-train', 'CIFAR100-val', 'CIFAR100-test']

    #--------------------------------
    # Model 
    #--------------------------------
    
        nn = vgg16()
        n_classes = len(Cifar100.get_classes(meta_path = Path(cifar_path)/'cifar-100-python/meta')) 

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

        datasets = ParsedDataset(
                path = ds_path,
                )

    #--------------------------------
    # Random Projections 
    #--------------------------------

        with datasets as ds:
                ds.load_only(
                        loaders = loaders,
                        verbose = verbose
                        )

                model.get_random_projs(
                        path = proj_path, 
                        name = proj_name, 
                        target_modules = target_layers,
                        proj_dim = 200,
                        sample_in = ds._dss['CIFAR100-train']['image'][0],
                        verbose = verbose,
                        )
    #--------------------------------
    # CoreVectors 
    #--------------------------------
        corevecs = CoreVectors(
                path = cvs_path,
                name = cvs_name,
                model = model,
                )
        # define a dimensionality reduction function for each layer
        reduction_fns = {}

        for layer in target_layers:
                layer_module = model._target_modules[layer]

                reduction_fns[layer] = partial(random_projection,
                        layer = layer_module,
                        proj = model._rand_projs[layer]['proj'],
                        device=device
                )

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
        cv_parsers = {}
        feature_sizes = {}
        for layer in target_layers:
                cv_parsers[layer] = partial(trim_corevectors,
                        module = layer,
                        cv_dim = 100)
                feature_sizes[layer] = 100

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
        
                #coverage = empp_coverage_scores(drillers=ph._drillers, threshold=0.8, plot=True, save_path='/home/claranunesbarrancos/repos/XAI/src/clustering_xp/temp_plots', file_name='coverage_vgg_550clusters.png')
                #empp_relative_coverage_scores(drillers=ph._drillers, threshold=0.8, plot=True, save_path='/home/claranunesbarrancos/repos/XAI/src/clustering_xp/temp_plots', file_name='relative_cluster_coverage_vgg_550clusters.png')
                compare_relative_coverage_all_clusters(
                        root_dir = '/home/claranunesbarrancos/repos/XAI/data/drillers_all',
                        threshold=0.8,
                        )


