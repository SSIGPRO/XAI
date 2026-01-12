import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from functools import partial

# torch stuff
import torch
from cuda_selector import auto_cuda

###### Our stuff
# Model
from peepholelib.models.model_wrap import ModelWrap, means, stds 

# datasets
from peepholelib.datasets.imagenet import ImageNet
from peepholelib.datasets.cifar100 import Cifar100
from peepholelib.datasets.cifarC import CifarC
from peepholelib.datasets.SVHN import SVHN 
from peepholelib.datasets.Places import Places 
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.datasets.functional.parsers import from_dataset

if sys.argv[1] == 'vgg_cifar100':
    from config_cifar100_vgg16 import *
elif sys.argv[1] == 'vit_cifar100':
    from config_cifar100_ViT import *
elif sys.argv[1] == 'vgg_imagenet':
    from config_imagenet_vgg16 import *
else:
    raise RuntimeError('Select a configuration by runing \'python xp_get_corevectors.py <vgg|vit>\'')

if __name__ == "__main__":
    #--------------------------------
    # Directories definitions
    #--------------------------------
    imagenet_path = '/srv/newpenny/dataset/ImageNet_torchvision'
    # cifar_path = '/srv/newpenny/dataset/CIFAR100'
    # cifarc_path = '/srv/newpenny/dataset/CIFAR-100-C'
    # svhn_path = '/srv/newpenny/dataset/SVHN' 
    # places_path = '/srv/newpenny/dataset/Places365'

    #--------------------------------
    # Model 
    #--------------------------------
    
    model = ModelWrap(
            model = model,
            device = device
            )
                                            
    # model.update_output(
    #         output_layer = output_layer, 
    #         to_n_classes = 100,
    #         overwrite = True 
    #         )
                                            
    # model.load_checkpoint(
    #         name = model_name,
    #         path = model_dir,
    #         verbose = verbose
    #         )
    
    model.normalize_model(mean=means[dataset], std=stds[dataset])
                                            
    #--------------------------------
    # Datasets 
    #--------------------------------
    # original datasets
    _dss = {
            # 'CIFAR100': Cifar100(
            #     path = cifar_path,
            #     transform = transform,
            #     seed = seed
            #     ),
            'ImageNet': ImageNet(
                path = imagenet_path,
                transform = transform,
                seed = seed
                ),
            # 'CIFARC': CifarC(
            #     path = cifarc_path,
            #     transform = transform,
            #     seed = seed
            #     ),
            # 'SVHN': SVHN(
            #     path = svhn_path,
            #     transform = transform,
            #     seed = seed
            #     ),
            # 'Places': Places(
            #     path = places_path,
            #     transform = transform,
            #     seed = seed
            #     )
            }

    _dss_parsers = {
            'ImageNet': from_dataset,
            #'CIFAR100': from_dataset,
            # 'CIFARC': from_dataset,
            # 'SVHN': from_dataset,
            # 'Places': from_dataset,
            }
    
    
    # just for testing
    from peepholelib.datasets.functional.samplers import random_subsampling 
    _dss_samplers = {
            k: partial(
                random_subsampling, 
                perc = 0.03
                ) for k in _dss.keys()
            }
    
    #######################
    # parsing datasets
    #######################
    
    # parse the original datasets into ds_path
    ParsedDataset.parse_ds(
            save_path = ds_path,
            model = model,
            datasets = _dss,
            ds_parsers = _dss_parsers, 
            ds_samplers = _dss_samplers, 
            batch_size = bs,
            n_threads = n_threads,
            verbose = verbose
            )
    
