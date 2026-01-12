import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial

# torch stuff
import torch
import torch.nn as nn
import torchvision
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
from peepholelib.peepholes.DeepMahalanobisDistance.DMD import DeepMahalanobisDistance as DMD
from peepholelib.peepholes.parsers import get_images  

from peepholelib.peepholes.peepholes import Peepholes

from peepholelib.models.viz import viz_singular_values_2
from peepholelib.utils.viz_empp import *

def feature_sizes_dmd_all_convs(model) -> dict[str, int]:
    feature_sizes = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            feature_sizes[name] = module.out_channels
    return feature_sizes
    
if __name__ == "__main__":
        use_cuda = torch.cuda.is_available()
        #device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
        device = torch.device('cuda:1') 
        torch.cuda.empty_cache()

        #device = torch.device("cpu")
        print(f"Using {device} device")

        #--------------------------------
        # Directories definitions
        #--------------------------------
        cifar_path = '/srv/newpenny/dataset/CIFAR100'
        ds_path = Path('/srv/newpenny/XAI/CN/mobilenet_data/cifar100')

        # model parameters
        seed = 29
        bs = 512
        n_threads = 1

        model_dir = '/srv/newpenny/XAI/models'
        model_name = 'CN_model=mobilenet_v2_dataset=CIFAR100_optim=Adam_scheduler=RoP_lr=0.001_factor=0.1_patience=5.pth'
        
        svds_path = '/srv/newpenny/XAI/CN/mobilenet_data'
        svds_name = 'svds' 
        
        cvs_path = Path.cwd()/'/srv/newpenny/XAI/CN/mobilenet_data/corevectors'
        cvs_name = 'coreavg'

        drill_path = Path.cwd()/'/srv/newpenny/XAI/CN/mobilenet_data/drillers_all/drillers_100'
        drill_name = 'DMD'

        phs_path = Path.cwd()/'/srv/newpenny/XAI/CN/mobilenet_data/peepholes_all/peepholes_100'
        phs_name = 'peepavg'
        
        verbose = True 
        
        target_layers = [ 'features.1.conv.0.0', 'features.1.conv.1','features.2.conv.0.0','features.2.conv.1.0','features.2.conv.2',
        'features.3.conv.0.0', 'features.3.conv.1.0', 'features.3.conv.2',
        'features.4.conv.0.0', 'features.4.conv.1.0', 'features.4.conv.2',
        'features.5.conv.0.0', 'features.5.conv.1.0', 'features.5.conv.2',
        'features.6.conv.0.0','features.6.conv.1.0', 'features.6.conv.2',
        'features.7.conv.0.0', 'features.7.conv.1.0','features.7.conv.2',
        'features.8.conv.0.0', 'features.8.conv.1.0', 'features.8.conv.2',
        'features.9.conv.0.0', 'features.9.conv.1.0', 'features.9.conv.2',  
        'features.10.conv.0.0', 'features.10.conv.1.0', 'features.10.conv.2',
        'features.11.conv.0.0', 'features.11.conv.1.0', 'features.11.conv.2',
        'features.12.conv.0.0', 'features.12.conv.1.0',  'features.12.conv.2',
        'features.13.conv.0.0', 'features.13.conv.1.0', 'features.13.conv.2',
        'features.14.conv.0.0', 'features.14.conv.1.0', 'features.14.conv.2',
        'features.15.conv.0.0', 'features.15.conv.1.0', 'features.15.conv.2',
        'features.16.conv.0.0', 'features.16.conv.1.0', 'features.16.conv.2', 
        'features.17.conv.0.0', 'features.17.conv.1.0', 'features.17.conv.2',
        'features.18.0', #'classifier.1',
        ]

        # target_layers = ['features.2.conv.0.0','features.3.conv.2', 'features.5.conv.1.0','features.6.conv.1.0',
        # 'features.9.conv.1.0','features.11.conv.2','features.17.conv.1.0', 'features.17.conv.2',
        # 'features.18.0', 'classifier.1']


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

        n_cluster = 100

    #--------------------------------
    # Model 
    #--------------------------------
    
        nn = torchvision.models.mobilenet_v2(pretrained=True)
        feature_sizes_dmd = feature_sizes_dmd_all_convs(nn)

        print("Num conv layers:", len(feature_sizes_dmd))
        print(list(feature_sizes_dmd.items()))   

        n_classes = len(Cifar100.get_classes(meta_path = Path(cifar_path)/'cifar-100-python/meta')) 

        model = ModelWrap(
                model = nn,
                device = device
                )
                                                
        model.update_output(
                output_layer = 'classifier.1', 
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
        svd_fns = {}

        for layer in target_layers:
                svd_fns[layer] = partial(conv2d_toeplitz_svd,
                        layer = layer,
                        rank = 200,
                        device=device
                )


        with datasets as ds:
                ds.load_only(
                        loaders = loaders,
                        verbose = verbose
                )

                model.get_svds(
                        path = svds_path,
                        name = svds_name,
                        target_modules = target_layers,
                        sample_in = ds._dss[f'CIFAR100-train']['image'][0],
                        svd_fns = svd_fns,
                        verbose = verbose
                )

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
        reduction_fns = {}
        for layer in target_layers:
                reduction_fns[layer] = partial(conv2d_toeplitz_svd_projection,
                        layer = model._target_modules[layer], 
                        svd = model._svds[layer],
                        use_s=True,
                        device=device
                )
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
        cv_parsers = {}
        feature_sizes = {}
        DMD_FEATURE_SIZES = dict([
        ('features.0.0', 32),('features.1.conv.0.0', 32), ('features.1.conv.1', 16),
        ('features.2.conv.0.0', 96), ('features.2.conv.1.0', 96), ('features.2.conv.2', 24),
        ('features.3.conv.0.0', 144), ('features.3.conv.1.0', 144), ('features.3.conv.2', 24),
        ('features.4.conv.0.0', 144), ('features.4.conv.1.0', 144), ('features.4.conv.2', 32),
        ('features.5.conv.0.0', 192), ('features.5.conv.1.0', 192), ('features.5.conv.2', 32),
        ('features.6.conv.0.0', 192), ('features.6.conv.1.0', 192), ('features.6.conv.2', 32),
        ('features.7.conv.0.0', 192), ('features.7.conv.1.0', 192), ('features.7.conv.2', 64),
        ('features.8.conv.0.0', 384), ('features.8.conv.1.0', 384), ('features.8.conv.2', 64),
        ('features.9.conv.0.0', 384), ('features.9.conv.1.0', 384), ('features.9.conv.2', 64),
        ('features.10.conv.0.0', 384), ('features.10.conv.1.0', 384), ('features.10.conv.2', 64),
        ('features.11.conv.0.0', 384), ('features.11.conv.1.0', 384), ('features.11.conv.2', 96),
        ('features.12.conv.0.0', 576), ('features.12.conv.1.0', 576), ('features.12.conv.2', 96),
        ('features.13.conv.0.0', 576), ('features.13.conv.1.0', 576), ('features.13.conv.2', 96),
        ('features.14.conv.0.0', 576), ('features.14.conv.1.0', 576), ('features.14.conv.2', 160),
        ('features.15.conv.0.0', 960), ('features.15.conv.1.0', 960), ('features.15.conv.2', 160),
        ('features.16.conv.0.0', 960), ('features.16.conv.1.0', 960), ('features.16.conv.2', 160),
        ('features.17.conv.0.0', 960), ('features.17.conv.1.0', 960), ('features.17.conv.2', 320),
        ('features.18.0', 1280),
        ])

        cv_parsers = {}
        feature_sizes = {}

        for layer in target_layers:
                print("in layer parser setup:", layer)
                if layer not in DMD_FEATURE_SIZES:
                        raise KeyError(
                                f"Layer {layer} not found in DMD_FEATURE_SIZES. "
                        )
                features_cv_dim = DMD_FEATURE_SIZES[layer]

                cv_parsers[layer] = partial(
                        trim_corevectors,
                        module=layer,
                        cv_dim=features_cv_dim
                )
                feature_sizes[layer] = features_cv_dim


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
                                        verbose=verbose
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