# python stuff
import os
import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
import numpy as np
from contextlib import ExitStack

# python stuff
from time import time
from functools import partial

# Attcks
import torchattacks
from peepholelib.adv_atk.attacks_base import ftd
from peepholelib.adv_atk.PGD import myPGD
from peepholelib.adv_atk.BIM import myBIM
from peepholelib.adv_atk.DeepFool import myDeepFool
from peepholelib.adv_atk.CW import myCW

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
from torch.utils.data import DataLoader

if __name__ == '__main__':
#     use_cuda = torch.cuda.is_available()
#     device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
#     print(f"Using {device} device")
    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 2
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    ds_path = '/srv/newpenny/dataset/CIFAR100'

    # model parameters
    #name_model = 'vgg16'
    name_model = 'ViT'
    pretrained = True
    dataset = 'CIFAR100' 
    seed = 29
    bs = 256 
    n_threads= 32
    magnitude = 0.004
    model_dir = '/srv/newpenny/XAI/models'
    #model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
    model_name = 'SV_model=vit_b_16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau_withInfo.pth'

    svds_path = '/srv/newpenny/XAI/Peephole-Analysis' #Path.cwd()/'../data'
    svds_name = 'svds' 
    
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
    # Attacks
    #--------------------------------
    dss = ds._dss
    #dss = random_subsampling(ds._dss, 0.05)
    dss_ = {#'train': dss['train'],
           'val': dss['val'],        
           'test': dss['test']
              }
    
    loaders = {}
    for key in dss_.keys():
        loaders[key] = DataLoader(dss_[key], batch_size=bs, shuffle=False) 

    atcks = {
             'myPGD':
                     {'model': model._model,
                      'eps' : 8/255, 
                      'alpha' : 2/255, 
                      'steps' : 10,
                      'device' : device,
                      'path' : '/srv/newpenny/XAI/generated_data/attacks/PGD',
                      'name' : 'PGD',
                      'dl' : loaders,
                      'name_model' : name_model,
                      'verbose' : True,
                      'mode' : 'random',},
             'myBIM': 
                     {'model': model._model,
                      'eps' : 8/255, 
                      'alpha' : 2/255, 
                      'steps' : 10,
                      'device' : device,
                      'path' : '/srv/newpenny/XAI/generated_data/attacks/BIM',
                      'name' : 'BIM',
                      'dl' : loaders,
                      'name_model' : name_model,
                      'verbose' : True,
                      'mode' : 'random',},
             'myCW':{
                      'model': model._model,
                      'device' : device,
                      'path' : '/srv/newpenny/XAI/generated_data/attacks/CW',
                      'name' : 'CW',
                      'dl' : loaders,
                      'name_model' : name_model,
                      'verbose' : True,
                      'nb_classes' : n_classes,
                      'confidence': 0,
                      'c_range': (1e-3, 1e10),
                      'max_steps': 1000,
                      'optimizer_lr': 1e-2,
                      'verbose': True,},
             'myDeepFool':{
                           'model': model._model,
                            'steps' : 50,
                            'overshoot' : 0.02,
                            'device' : device,
                            'path' : '/srv/newpenny/XAI/generated_data/attacks/DeepFool',
                            'name' : 'DeepFool',
                            'dl' : loaders,
                            'name_model' : name_model,
                            'verbose' : True,
                            }
                  }

    atk_dss_dict = {}
    for atk_type, kwargs in atcks.items():
        atk = eval(atk_type)(**kwargs)
        atk.load_data()

        if not atk.atk_path.exists():
            atk.get_ds_attack()
        
        atk_dss_dict[atk_type] = atk
    
    # define a dimensionality reduction function for each layer

#     reduction_fns = {
#               'features.2': ChannelWiseMean_conv, 

#               'features.7': ChannelWiseMean_conv, 

#               'features.14': ChannelWiseMean_conv, 

#               'features.21': ChannelWiseMean_conv, 

#               'features.28': ChannelWiseMean_conv,     
#             }
    
    reduction_fns = { f'encoder.layers.encoder_layer_{i}.mlp.3': cls_token_ViT for i in range(12)}
   
    #--------------------------------
    # Corevectors attacks 
    #--------------------------------
    
    for atk_type, atk_dss in atk_dss_dict.items():
        print('--------------------------------')
        print('atk type: ', atk_type)
        print('--------------------------------')
        cvs_path_ = cvs_path/f'{atk_type.replace('my', "")}'
        phs_path_ = phs_path/f'{atk_type.replace('my', "")}'

        corevecs = CoreVectors(
            path = cvs_path_,
            name = cvs_name,
            model = model,
            )

        with corevecs as cv: 
            # copy dataset to coreVect dataset
            cv.parse_ds(
                        batch_size = bs,
                        datasets = atk_dss,
                        n_threads = n_threads,
                        ds_parser = partial(ftd, key_list = list(atk_dss._dss['test'].keys())),
                        verbose = verbose
                        )
        
            '''
            # This occupies a lot of space. Only do if you need it
            # copy dataset to activatons file
            cv.get_activations(
                     batch_size = bs,
                     n_threads = n_threads,
                     save_input = True,
                     save_output = False,
                     verbose = verbose
                     )        
            '''

            # computing the corevectors
            cv.get_coreVectors(
                     batch_size = bs,
                     reduction_fns = reduction_fns,
                     activations_parser = get_out_activations,
                     n_threads = n_threads,
                     save_input = False,
                     save_output = True,
                     verbose = verbose
                     )

        #--------------------------------
        # Peepholes
        #--------------------------------
        feature_sizes = { f'encoder.layers.encoder_layer_{i}.mlp.3': 768 for i in range(12)}

        # feature_sizes = {

        #         'features.2': 64, 

        #         'features.7': 128, 

        #         'features.14': 256, 

        #         'features.21': 512,

        #         'features.28': 512,

        #         }
    

        cls_kwargs = {}#{'batch_size':256} 

        drillers = {}

        for peep_layer in target_layers:
                drillers[peep_layer] = DMD(
                                        path = drill_path,
                                        name = drill_name+'.'+peep_layer,
                                        nl_model = n_classes,
                                        n_features = feature_sizes[peep_layer],
                                        parser = get_images,
                                        parser_kwargs = {},
                                        model = model,
                                        layer = peep_layer,
                                        magnitude = magnitude,
                                        std_transform = [0.300, 0.287, 0.294],
                                        device = device,
                                        parser_act = cls_token_ViT
                                        )
               
                # if (drillers[peep_layer].dmd_folder).exists():
                #         print(f'Loading Classifier for {peep_layer}') 
                #         drillers[peep_layer].load()
                # else:
                #         RuntimeError(f'The file {drillers[peep_layer].dmd_folder} does not exist')

        peepholes = Peepholes(
            path = phs_path_,
            name = f'{phs_name}',
            device = device
            )

        # loading classifiers
        # with corevecs as cv:
        #         cv.load_only(
        #                 loaders = ['val','test'],
        #                 verbose = True
        #                 ) 
                
        #         for drill_key, driller in drillers.items():
        #                 print(driller.dmd_folder)
        #                 if (driller.dmd_folder).exists():
        #                         print(f'Loading Classifier for {drill_key}') 
        #                         driller.load()
        #                 else:
        #                       raise RuntimeError(f'The file {drill_path/(driller.dmd_folder)} does not exist')

        with corevecs as cv, peepholes as ph:
                cv.load_only(
                        loaders = ['val','test'],
                        verbose = True
                        ) 
                for drill_key, driller in drillers.items():
                        print(driller.dmd_folder)
                        if (driller.dmd_folder).exists():
                                print(f'Loading Classifier for {drill_key}') 
                                driller.load()
                        else:
                              raise RuntimeError(f'The file {drill_path/(driller.dmd_folder)} does not exist')

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
