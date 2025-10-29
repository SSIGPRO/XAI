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
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.get_coreVectors import get_out_activations
from peepholelib.coreVectors.dimReduction.avgPooling import ChannelWiseMean_conv, cls_token_ViT, TokenWiseMean_ViT

from peepholelib.peepholes.parsers import get_images 
from peepholelib.peepholes.DeepMahalanobisDistance.DMD import DeepMahalanobisDistance as DMD
from peepholelib.peepholes.peepholes import Peepholes

from peepholelib.utils.samplers import random_subsampling 

# torch stuff
import torch
from torchvision.models import vgg16, VGG16_Weights, vit_b_16
from cuda_selector import auto_cuda
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT

if __name__ == "__main__":
#     use_cuda = torch.cuda.is_available()
#     device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
#     print(f"Using {device} device")
    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 4
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    ds_path = '/srv/newpenny/dataset/CIFAR100'

    # model parameters
    pretrained = True
    dataset = 'CIFAR100' 
    seed = 29
    bs = 512
    n_threads = 32
    magnitude = 0.004
    model_dir = '/srv/newpenny/XAI/models'
    name_model = 'ViT'
    #name_model = 'vgg16'
    #model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
    model_name = 'SV_model=vit_b_16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau_withInfo.pth'

    cvs_path = Path.cwd()/f'../data/{name_model}/corevectors' #Path('/srv/newpenny/XAI/Peephole-Analysis/corevectors')  
    cvs_name = 'coreavg'
    
    drill_path = Path.cwd()/f'../data/{name_model}/drillers' #Path('/srv/newpenny/XAI/Peephole-Analysis/drillers_on_class') 
    drill_name = 'DMD'

    phs_path = Path.cwd()/f'../data/{name_model}/peepholes' #Path('/srv/newpenny/XAI/Peephole-Analysis/peepholes_on_class') 
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
            batch_size = bs,
            data_kwargs = {'num_workers': 4, 'pin_memory': True},
            seed = seed,
            )
    
    #--------------------------------
    # Model 
    #--------------------------------
    
    #nn = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    
    nn = vit_b_16()
    n_classes = len(ds.get_classes())

    model = ModelWrap(
            model = nn,
            device = device
            )

    model.update_output(
            output_layer = 'heads.head', #'classifier.6' 
            to_n_classes = n_classes,
            overwrite = True 
            )

    model.load_checkpoint(
            name = model_name,
            path = model_dir,
            verbose = verbose
            )

    target_layers = [f'encoder.layers.encoder_layer_{i}.mlp.3' for i in range(12)]
    
#     target_layers = [
#            'features.3',
#            'features.8',
#            'features.15',
#            'features.22',
#            'features.29'
#             ]
    
#     target_layers = [
#            'features.2',
#            'features.7',
#            'features.14',
#            'features.21',
#            'features.28'
#             ]
    model.set_target_modules(target_modules=target_layers, verbose=verbose)

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
#     reduction_fns = {
#               'features.2': ChannelWiseMean_conv, 

#               'features.7': ChannelWiseMean_conv, 

#               'features.14': ChannelWiseMean_conv, 

#               'features.21': ChannelWiseMean_conv, 

#               'features.28': ChannelWiseMean_conv,     
#             }
    reduction_fns = { f'encoder.layers.encoder_layer_{i}.mlp.3': cls_token_ViT for i in range(12)}
    
    with corevecs as cv: 
        # copy dataset to coreVect dataset

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

    feature_sizes = { f'encoder.layers.encoder_layer_{i}.mlp.3': 768 for i in range(12)}

#     feature_sizes = {

#                 'features.2': 64, 

#                 'features.7': 128, 

#                 'features.14': 256, 

#                 'features.21': 512,

#                 'features.28': 512,

#                 }
    
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
                magnitude = magnitude,
                std_transform = [0.300, 0.287, 0.294],
                device = device,
                parser_act = cls_token_ViT
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

   