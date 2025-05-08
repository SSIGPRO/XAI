import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial
from sklearn import covariance
import numpy as np
from tqdm import tqdm

# Our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.datasets.transforms import vgg16_cifar100 as ds_transform 
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.avgPooling import ChannelWiseMean_conv

from peepholelib.peepholes.parsers import get_images 
from peepholelib.peepholes.DeepMahalanobisDistance.DMD import DeepMahalanobisDistance as DMD
from peepholelib.peepholes.peepholes import Peepholes

from peepholelib.utils.samplers import random_subsampling 

# torch stuff
import torch
from torchvision.models import vgg16, VGG16_Weights
from cuda_selector import auto_cuda
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    ds_path = '/srv/newpenny/dataset/CIFAR100'

    # model parameters
    pretrained = True
    dataset = 'CIFAR100'
    name_model = 'vgg16' 
    seed = 29
    bs = 256 
    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
    
    cvs_path = Path.cwd()/'../data/corevectors'
    cvs_name = 'corevectors_avg'

    drill_path = Path.cwd()/'../data/drillers'
    drill_name = 'DMD'

    phs_path = Path.cwd()/'../data/peepholes'
    phs_name = 'peepholes_avg'

    verbose = True 
    
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
    
    nn = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    in_features = 4096
    n_classes = len(ds.get_classes()) 
    nn.classifier[-1] = torch.nn.Linear(in_features, n_classes)
    model = ModelWrap(
            model=nn,
            path=model_dir,
            name=model_name,
            device=device
            )
    model.load_checkpoint(verbose=verbose)

    target_layers = [
        #     'classifier.0',
            # 'classifier.3',
            #'features.7',
            'features.14',
            'features.28',
            ]
    model.set_target_modules(target_modules=target_layers, verbose=verbose)

    direction = {'save_input':True, 'save_output':False}
    model.add_hooks(**direction, verbose=False) 
    
    dry_img, _ = ds._dss['train'][0]
    dry_img = dry_img.reshape((1,)+dry_img.shape)
    model.dry_run(x=dry_img)

    #--------------------------------
    # CoreVectors 
    #--------------------------------
    #dss = ds._dss
    dss = random_subsampling(ds._dss, 0.025)
    
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            model = model,
            )
    
    # for each layer we define the function used to perform dimensionality reduction
    reduction_fns = {
            'features.14': ChannelWiseMean_conv,
            'features.28': ChannelWiseMean_conv,
            }
    
    with corevecs as cv: 
        # copy dataset to coreVect dataset
        cv.get_activations(
                batch_size = bs,
                datasets = dss,
                verbose = verbose
                )
        
        cv.get_coreVectors(
                batch_size = bs,
                reduction_fns = reduction_fns,
                verbose = verbose
                )

        cv_dl = cv.get_dataloaders(verbose=verbose)
    
        i = 0
        print('\nPrinting some corevecs')
        for data in cv_dl['train']:
            print(data['features.14'].shape)
            print(data['features.14'][34:56,:])
            i += 1
            if i == 3: break

    #--------------------------------
    # Peepholes
    #--------------------------------

    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )
    
    n_classes = 100
    n_cluster = 10
    peep_layers = ['features.14','features.28']

    feature_sizes = {
            'features.14': 256,
            'features.28': 512,
            }
    
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )

    drillers = {}
    for peep_layer in peep_layers:
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
            driller = drillers,
            target_modules = peep_layers,
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
                driller.fit(corevectors = cv._corevds['train'][drill_key], 
                        activations = cv._actds['train'], 
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
                batch_size = bs,
                verbose = verbose,
                )
        i = 0
        print('\nPrinting some peeps')
        ph_dl = ph.get_dataloaders(verbose=verbose)
        for data in ph_dl['test']:
            print('phs\n', data['features.28']['peepholes'])
            i += 1
            if i == 3: break
