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
from peepholelib.coreVectors.dimReduction.svds import svd_Linear, svd_Conv2D
from peepholelib.peepholes.classifiers.parsers import trim_corevectors
from peepholelib.peepholes.classifiers.tkmeans import KMeans as tKMeans 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.utils.samplers import random_subsampling 

# torch stuff
import torch
from torchvision.models import vgg16, VGG16_Weights
from cuda_selector import auto_cuda
from torch.utils.data import DataLoader

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    ds_path = '/srv/newpenny/dataset/CIFAR100'

    # model parameters
    name_model = 'vgg16'
    pretrained = True
    dataset = 'CIFAR100' 
    seed = 29
    bs = 512 
    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
    
    svds_path = Path.cwd()/'../data'
    svds_name = 'svds' 
    
    cvs_path = Path.cwd()/'../data/corevectors'
    cvs_name = 'corevectors'
   
    drill_path = Path.cwd()/'../data/drillers'
    drill_name = 'classifier'

    phs_path = Path.cwd()/'../data/peepholes'
    phs_name = 'peepholes'
    
    verbose = True 

    #--------------------------------
    # Dataset 
    #--------------------------------
    
    ds = Cifar(dataset=dataset, data_path=ds_path)
    ds.load_data(
            batch_size = bs,
            data_kwargs = {'num_workers': 4, 'pin_memory': True},
            seed = seed,
            )

    #--------------------------------
    # Model 
    #--------------------------------
    
    nn = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    in_features = 4096
    n_classes = len(ds.get_classes()) 
    nn.classifier[-1] = torch.nn.Linear(in_features, n_classes)
    model = ModelWrap(device=device)
    model.set_model(model=nn, path=model_dir, name=model_name, verbose=verbose)

    target_layers = [
            'classifier.0',
            'classifier.3',
            #'features.7',
            #'features.14',
            #'features.28',
            ]
    model.set_target_layers(target_layers=target_layers, verbose=verbose)

    direction = {'save_input':True, 'save_output':False}
    model.add_hooks(**direction, verbose=False) 
    
    dry_img, _ = ds._dss['train'][0]
    dry_img = dry_img.reshape((1,)+dry_img.shape)
    model.dry_run(x=dry_img)

    #--------------------------------
    # SVDs 
    #--------------------------------
    print('target layers: ', model.get_target_layers()) 
    model.get_svds(path=svds_path, name=svds_name, verbose=verbose)
    for k in model._svds.keys():
        for kk in model._svds[k].keys():
            print('svd shapes: ', k, kk, model._svds[k][kk].shape)
       
    #--------------------------------
    # Attacks
    #--------------------------------
    #dss = ds._dss
    dss = random_subsampling(ds._dss, 0.05)
    dss = {#'train': dss['train'],
           'test': dss['test']
              }
    print(dss)
    loaders = {}
    for key in dss.keys():
        loaders[key] = DataLoader(dss[key], batch_size=bs, shuffle=False) 

    atcks = {
             'myPGD':
                     {'model': model._model,
                      'eps' : 8/255, 
                      'alpha' : 2/255, 
                      'steps' : 10,
                      'device' : device,
                      'path' : Path.cwd()/'../data/attacks/PGD',
                      'name' : 'PGD',
                      'dl' : loaders,
                      'name_model' : name_model,
                      'verbose' : True,
                      'mode' : 'random',},
        #      'myBIM': 
        #              {'model': model._model,
        #               'eps' : 8/255, 
        #               'alpha' : 2/255, 
        #               'steps' : 10,
        #               'device' : device,
        #               'path' : Path.cwd()/'../data/attacks/BIM',
        #               'name' : 'BIM',
        #               'dl' : loaders,
        #               'name_model' : name_model,
        #               'verbose' : True,
        #               'mode' : 'random',},
        #      'myCW':{
        #               'model': model._model,
        #               'device' : device,
        #               'path' : Path.cwd()/'../data/attacks/CW',
        #               'name' : 'CW',
        #               'dl' : loaders,
        #               'name_model' : name_model,
        #               'verbose' : True,
        #               'nb_classes' : n_classes,
        #               'confidence': 0,
        #               'c_range': (1e-3, 1e10),
        #               'max_steps': 1000,
        #               'optimizer_lr': 1e-2,
        #               'verbose': True,},
        #      'myDeepFool':{
        #                    'model': model._model,
        #                     'steps' : 50,
        #                     'overshoot' : 0.02,
        #                     'device' : device,
        #                     'path' : Path.cwd()/'../data/attacks/DeepFool',
        #                     'name' : 'DeepFool',
        #                     'dl' : loaders,
        #                     'name_model' : name_model,
        #                     'verbose' : True,
        #                     }
                  }

    atk_dss_dict = {}
    for atk_type, kwargs in atcks.items():
        atk = eval(atk_type)(**kwargs)

        if not atk.atk_path.exists():
            atk.get_ds_attack()
        
        atk_dss_dict[atk_type] = atk._atkds
    
    # define a dimensionality reduction function for each layer
    reduction_fns = {
            'classifier.0': partial(svd_Linear,
                                    reduct_m=model._svds['classifier.0']['Vh'], 
                                    device=device),
            'classifier.3': partial(svd_Linear,
                                    reduct_m=model._svds['classifier.3']['Vh'], 
                                    device=device),
        #     'features.28': partial(svd_Conv2D, 
        #                             reduct_m=model._svds['features.28']['Vh'], 
        #                             layer=model._target_layers['features.28'], 
        #                             device=device),
            }
    
    shapes = {
            'classifier.0': 4096,
            'classifier.3': 4096,
            #'features.28': 300,
            }    
    #--------------------------------
    # Corevectors attacks 
    #--------------------------------
    
    for atk_type, atk_dss in atk_dss_dict.items():
        cvs_path_ = cvs_path/f'{atk_type.replace('my', "")}'
        phs_path_ = phs_path/f'{atk_type.replace('my', "")}'

        corevecs = CoreVectors(
            path = cvs_path_,
            name = cvs_name,
            model = model,
            )

        with corevecs as cv: 
            # copy dataset to coreVect dataset
            cv.get_activations(
                batch_size = bs,
                datasets = atk_dss,
                ds_parser = ftd,
                key_list = list(atk_dss['test'].keys()),
                verbose = verbose
                )        
        
            # defining the corevectors
            cv.get_coreVectors(
                batch_size = bs,
                reduction_fns = reduction_fns,
                shapes = shapes,
                verbose = verbose
                )
            cv_dl = cv.get_dataloaders(verbose=verbose)
    
            i = 0
            print('\nPrinting some corevecs')
            for data in cv_dl['test']:
                print('\nclassifier.0')
                print(data['classifier.0'][34:56,:])
                i += 1
                if i == 1: break

            cv.normalize_corevectors(
                    target_layers = target_layers,
                    from_file=cvs_path/(cvs_name+'.normalization.pt'),
                    verbose=True
                    )
            i = 0
            print('after norm')
            for data in cv_dl['test']:
                print(data['classifier.0'][34:56,:])
                i += 1
                if i == 1: break

        #--------------------------------
        # Peepholes
        #--------------------------------
        n_classes = 100
        n_cluster = 10
        cv_dim = 10
        parser_cv = trim_corevectors
        peep_layers = ['classifier.0', 'classifier.3']

        cls_kwargs = {}#{'batch_size':256} 

        drillers = {}

        for peep_layer in peep_layers:
                parser_kwargs = {'layer': peep_layer, 'cv_dim':cv_dim}

                drillers[peep_layer] = tGMM(
                                        path = drill_path,
                                        name = drill_name+'.'+peep_layer,
                                        nl_classifier = n_cluster,
                                        nl_model = n_classes,
                                        n_features = cv_dim,
                                        parser = parser_cv,
                                        parser_kwargs = parser_kwargs,
                                        batch_size = 256,
                                        device = device
                                        )
                if (drill_path/(drillers[peep_layer]._suffix+'.empp.pt')).exists():
                        print(f'Loading Classifier for {peep_layer}') 
                        drillers[peep_layer].load()
                else:
                        print('The file does not exist')

        peepholes = Peepholes(
                path = phs_path_,
                name = f'{phs_name}.ps_{parser_kwargs['cv_dim']}.nc_{n_cluster}',
                driller = drillers,
                target_layers = peep_layers,
                device = device
                )

        # loading classifiers
        with corevecs as cv:
                cv.load_only(
                        loaders = ['test'],
                        verbose = True
                        ) 
                
                for drill_key, driller in drillers.items():
                        print(driller._suffix)
                        if (drill_path/(driller._suffix+'.empp.pt')).exists():
                                print(f'Loading Classifier for {drill_key}') 
                                driller.load()
                        else:
                              raise RuntimeError('The file does not exist')

        with corevecs as cv, peepholes as ph:
                cv.load_only(
                        loaders = ['test'],
                        verbose = True
                        ) 

                ph.get_peepholes(
                        corevectors = corevecs,
                        batch_size = 256,
                        verbose = verbose
                        )

                ph.get_scores(
                        batch_size = bs,
                        verbose=verbose
                        )

                i = 0
                print('\nPrinting some peeps')
                ph_dl = ph.get_dataloaders(verbose=verbose)
                for data in ph_dl['test']:
                        print('phs\n', data[peep_layer]['peepholes'])
                        print('max\n', data[peep_layer]['score_max'])
                        print('ent\n', data[peep_layer]['score_entropy'])
                        i += 1
                        if i == 3: break
