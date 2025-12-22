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
from peepholelib.models.viz import viz_singular_values
from peepholelib.utils.viz_empp import *

def load_all_drillers(**kwargs):
    n_cluster_list = kwargs.get('n_cluster_list', None)
    target_layers = kwargs.get('target_layers', None)
    device = kwargs.get('device', None)
    feature_sizes = kwargs.get('feature_sizes', None)
    cv_parsers = kwargs.get('cv_parsers', None)
    base_drill_path = kwargs.get('drill_path', None) 

    all_drillers = {}

    for n_cluster in n_cluster_list:
        # assuming u have a folder with all the drillers and u name it like drillers_{n_cluster}
        drill_path = base_drill_path / f"drillers_{n_cluster}" 

        drillers = {}
        for peep_layer in target_layers:
            drillers[peep_layer] = tGMM(
                path=drill_path,
                name=f"classifier.{peep_layer}",  
                nl_classifier=n_cluster,
                nl_model=n_classes,
                n_features=feature_sizes[peep_layer],
                parser=cv_parsers[peep_layer],
                device=device
            )

        for drill_key, driller in drillers.items():
            if driller._empp_file.exists():
                print(f'Loading Classifier for {drill_key}')
                driller.load()
            else:
                print(f'No Classifier found for {drill_key} at {driller._empp_file}')

        all_drillers[n_cluster] = drillers

    return all_drillers




if __name__ == "__main__":
        use_cuda = torch.cuda.is_available()
        device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
        torch.cuda.empty_cache()

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

        drill_path = Path('/srv/newpenny/XAI/CN/vgg_data_rand_proj2/drillers_all')
        drill_name = 'classifier'

        phs_path =  Path('/srv/newpenny/XAI/CN/vgg_data_rand_proj2/peepholes')
        phs_name = 'peepholes'

        plots_path = Path.cwd()/'temp_plots/coverage/'
        
        verbose = True 

        features_cv_dim = 100

        
        # Peepholelib
        target_layers = [ 'features.7', 'features.10', 'features.12', 'features.14', 'features.17', 'features.19', 'features.21',
                                'features.24','features.26','features.28','classifier.0','classifier.3', 
                                'classifier.6',
                        ]
        
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
    # CoreVectors 
    #--------------------------------
        corevecs = CoreVectors(
                path = cvs_path,
                name = cvs_name,
                model = model,
                )

    #--------------------------------
    # Peepholes
    #--------------------------------

        cv_parsers = {
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

        drillers_dict = load_all_drillers(
            n_cluster_list = [50,100, 200],  
            target_layers = target_layers,
            drill_path = drill_path,
            device = device,
            feature_sizes = feature_sizes,
            cv_parsers = cv_parsers
            )


        # peepholes = Peepholes(
        #         path = phs_path,
        #         name = phs_name,
        #         device = device
        #         )

        with datasets as ds, corevecs as cv:
                ds.load_only(
                        loaders = loaders,
                        verbose = verbose
                        )

                cv.load_only(
                        loaders = loaders,
                        verbose = verbose 
                        ) 
                


                # ph.get_peepholes(
                #         datasets = ds,
                #         corevectors = cv,
                #         target_modules = target_layers,
                #         batch_size = bs,
                #         drillers = drillers,
                #         n_threads = n_threads,
                #         verbose = verbose
                #         )
        
                #coverage = empp_coverage_scores(drillers=ph._drillers, threshold=0.8, plot=True, save_path='/home/claranunesbarrancos/repos/XAI/src/clustering_xp/temp_plots', file_name='coverage_vgg_550clusters.png')
                #empp_relative_coverage_scores(drillers=ph._drillers, threshold=0.8, plot=True, save_path='/home/claranunesbarrancos/repos/XAI/src/clustering_xp/temp_plots', file_name='relative_cluster_coverage_vgg_550clusters.png')
                compare_relative_coverage_all_clusters( all_drillers = drillers_dict,
                        threshold=0.8, plot= True, save_path=plots_path, filename='relative_cluster_coverage_vgg_rand_proj2.png')


