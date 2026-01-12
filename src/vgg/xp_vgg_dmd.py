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
from peepholelib.datasets.functional.samplers import random_subsampling 

# corevecs
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import linear_svd_projection, conv2d_toeplitz_svd_projection
from peepholelib.coreVectors.dimReduction.avgPooling import ChannelWiseMean_conv as parser_act
from peepholelib.coreVectors.get_coreVectors import get_out_activations as activations_parser

# peepholes
from peepholelib.peepholes.parsers import trim_corevectors
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.peepholes.parsers import get_images  
from peepholelib.peepholes.DeepMahalanobisDistance.DMD import DeepMahalanobisDistance as DMD

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
    ds_path = '/srv/newpenny/XAI/generated_data/TPAMI/parsed_datasets/CIFAR100_VGG16'

    # model parameters
    seed = 29
    bs = 128
    n_threads = 1

    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
    
    svds_path = Path('/srv/newpenny/XAI/CN/vgg_data')
    svds_name = 'svds' 
    
    cvs_path = Path('/srv/newpenny/XAI/CN/vgg_data/corevectors')
    cvs_name = 'coreavg'

    drill_path = Path('/srv/newpenny/XAI/CN/vgg_data/drillers_all/drillers_100')
    drill_name = 'DMD'

    phs_path = Path.cwd()/'/srv/newpenny/XAI/CN/vgg_data/peepholes_all/peepholes_100'
    phs_name = 'peepavg'
    
    verbose = True 
    
    # Peepholelib
    target_layers = [ 'features.0', 'features.2', 'features.5','features.7', 'features.10', 'features.12', 'features.14', 'features.17', 'features.19', 'features.21',
                                'features.24','features.26','features.28',
                            # 'classifier.0','classifier.3', 'classifier.6',
                    ]
    features24_svd_rank = 100 
    features26_svd_rank = 100 
    features28_svd_rank = 100
    classifier_svd_rank = 100 
    n_cluster = 100
    features_cv_dim = 100

    n_conceptograms = 2 
    
    loaders = [
        'CIFAR100-train',
        'CIFAR100-val',
        'CIFAR100-test',
        'CIFAR100-C-val-c0',
        'CIFAR100-C-test-c0',
        'CIFAR100-C-val-c1',
        'CIFAR100-C-test-c1',
        'CIFAR100-C-val-c2',
        'CIFAR100-C-test-c2',
        'CIFAR100-C-val-c3',
        'CIFAR100-C-test-c3',
        'CIFAR100-C-val-c4',
        'CIFAR100-C-test-c4',
        'SVHN-val',
        'SVHN-test',
        'Places365-val',
        'Places365-test',
        ]

    #--------------------------------
    # Model 
    #--------------------------------

    nn = vgg16()
    print(target_layers)
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
    # SVDs 
    #--------------------------------
        # svd_fns = {}

        # for layer in target_layers:
        #         svd_fns[layer] = partial(conv2d_toeplitz_svd,
        #                 layer = layer,
        #                 rank = 200,
        #                 device=device
        #         )


        # with datasets as ds:
        #         ds.load_only(
        #                 loaders = loaders,
        #                 verbose = verbose
        #         )

        #         model.get_svds(
        #                 path = svds_path,
        #                 name = svds_name,
        #                 target_modules = target_layers,
        #                 sample_in = ds._dss[f'CIFAR100-train']['image'][0],
        #                 svd_fns = svd_fns,
        #                 verbose = verbose
        #         )

    #--------------------------------
    # CoreVectors 
    #--------------------------------

    # corevecs = CoreVectors(
    #         path = cvs_path,
    #         name = cvs_name,
    #         )
    corevecs = CoreVectors(
        path = cvs_path,
        name = cvs_name,
        model = model,
        )

    # define a dimensionality reduction function for each layer
    reduction_fns = {layer: parser_act for layer in target_layers}

    with datasets as ds, corevecs as cv: 
        ds.load_only(
                    loaders = loaders,
                    verbose = verbose
                    )

        # computing the corevectors
        cv.get_coreVectors(
                datasets = ds,
                reduction_fns = reduction_fns,
                activations_parser = activations_parser,
                save_input = False,
                save_output = True,
                batch_size = bs,
                n_threads = n_threads,
                verbose = verbose
        )

#--------------------------------
# Peepholes
#--------------------------------
    feature_sizes = {
                    'features.0': 64,'features.2': 64, 
                    'features.5': 128, 'features.7': 128, 
                    'features.10': 256,'features.12': 256,'features.14': 256, 
                    'features.17': 512, 'features.19': 512, 'features.21': 512,
                    'features.24': 512,'features.26': 512, 'features.28': 512,
                    }

    with datasets as ds, corevecs as cv: 
            ds.load_only(
                    loaders = loaders,
                    verbose = verbose
                    )

            cv.load_only(
                    loaders = loaders, 
                    verbose = verbose 
                    ) 
            drillers = {}
            for peep_layer in target_layers:
                    print(f'Setting up DMD for layer {peep_layer}')
                    drillers[peep_layer] = DMD(
                            path = drill_path,
                            name = drill_name+'.'+ peep_layer,
                            nl_model = n_classes,
                            n_features = feature_sizes[peep_layer],
                            parser = get_images,
                            model = model,
                            layer = peep_layer,
                            magnitude = 0.004,
                            std_transform = [0.300, 0.287, 0.294],
                            device = device,
                            parser_act = parser_act
                            )

                    if (drill_path/drillers[peep_layer]._suffix/'precision.pt').exists():
                            print(f'Loading DMD for {peep_layer}') 
                            drillers[peep_layer].load()
                    else:   
                            print(f'Fitting DMD for {peep_layer}')
                            drillers[peep_layer].fit(
                                    dataset = ds,
                                    corevectors = cv,
                                    loader = 'CIFAR100-train',
                                    drill_key = peep_layer,
                                    verbose = verbose
                                    )
                    
                            # save classifiers
                            print(f'Saving classifier for {peep_layer}')
                            drillers[peep_layer].save()

            print("Getting Peepholes")
            peepholes = Peepholes(
                    path = phs_path,
                    name = phs_name,
                    device = device
                    )

            with peepholes as ph:
                    print("Computing Peepholes")
                    ph.get_peepholes(
                            datasets = ds,
                            corevectors = cv,
                            target_modules = target_layers,
                            batch_size = bs,
                            drillers = drillers,
                            n_threads = 1,
                            verbose = verbose 
                            )