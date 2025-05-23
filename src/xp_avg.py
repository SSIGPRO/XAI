import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial

# Our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.datasets.transforms import vgg16_cifar100 as ds_transform 
from peepholelib.models.model_wrap import ModelWrap 

from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.avgPooling import ChannelWiseMean_conv
from peepholelib.coreVectors.get_coreVectors import get_out_activations

from peepholelib.peepholes.parsers import get_images 
from peepholelib.peepholes.DeepMahalanobisDistance.DMD import DeepMahalanobisDistance as DMD
from peepholelib.peepholes.peepholes import Peepholes

from peepholelib.utils.samplers import random_subsampling 

# torch stuff
import torch
from torchvision.models import vgg16, VGG16_Weights
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
    bs = 64 
    n_threads = 32

    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
    
    cvs_path = Path.cwd()/'../data/corevectors'
    cvs_name = 'corevectors_avg'

    drill_path = Path.cwd()/'../data/drillers'
    drill_name = 'DMD'

    phs_path = Path.cwd()/'../data/peepholes'
    phs_name = 'peepholes_avg'

    verbose = True 

    # Peepholelib
    target_layers = [
            'features.7',
            'features.14',
            'features.28',
            ]

    #--------------------------------
    # Dataset 
    #--------------------------------

    ds = Cifar(
            data_path = ds_path,
            dataset=dataset
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
            model=nn,
            device=device
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
            target_modules=target_layers,
            verbose=verbose
            )

    #--------------------------------
    # CoreVectors 
    #--------------------------------
    random_subsampling(ds, 0.025)
    
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            model = model,
            )
    
    # for each layer we define the function used to perform dimensionality reduction
    reduction_fns = {
            'features.7': ChannelWiseMean_conv,
            'features.14': ChannelWiseMean_conv,
            'features.28': ChannelWiseMean_conv,
            }
    
    with corevecs as cv: 
        cv.parse_ds(
                batch_size = bs,
                datasets = ds,
                n_threads = n_threads,
                verbose = verbose
                )
        
        '''
        # This occupies a lot of space. Only do if you need it
        # copy dataset to coreVect dataset
        cv.get_activations(
                batch_size = bs,
                n_threads = n_threads,
                save_input = False,
                save_output = True,
                verbose = verbose
                )
        '''

        cv.get_coreVectors(
                batch_size = bs,
                reduction_fns = reduction_fns,
                activations_parser = get_out_activations,
                save_input = False,
                save_output = True,
                n_threads = n_threads,
                verbose = verbose
                )

    #--------------------------------
    # Peepholes
    #--------------------------------

    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )
    
    # number of channels in a conv layer. Get numbers from `nn`
    feature_sizes = {
            'features.7': 128,
            'features.14': 256,
            'features.28': 512,
            }
    
    drillers = {}
    for peep_layer in target_layers:
        drillers[peep_layer] = DMD(
                path = drill_path,
                name = drill_name+'.'+peep_layer,
                nl_model = n_classes,
                n_features = feature_sizes[peep_layer],
                parser = get_images,
                model = model,
                layer = peep_layer,
                magnitude = 0.004,
                std_transform = [0.229, 0.224, 0.225],
                device = device,
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
            if (drill_path/driller._suffix/'precision.pt').exists():
                print(f'Loading DMD for {drill_key}') 
                driller.load()
            else:
                t0 = time()
                print(f'Fitting DMD for {drill_key} time = ', time()-t0)
                driller.fit(
                        corevectors = cv._corevds['train'][drill_key], 
                        dataset = cv._dss['train'], 
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
                verbose = verbose,
                )

        ph.get_scores(
            batch_size = bs,
            verbose=verbose
            )
