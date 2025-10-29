# python stuff
import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial
from matplotlib import pyplot as plt

# Attcks
import torchattacks
from peepholelib.adv_atk.attacks_base import ftd
from peepholelib.adv_atk.PGD import myPGD
from peepholelib.adv_atk.BIM import myBIM
from peepholelib.adv_atk.DeepFool import myDeepFool
from peepholelib.adv_atk.CW import myCW

# Our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.datasets.transforms import vgg16_cifar100 as ds_transform 
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.models.svd_fns import linear_svd, conv2d_toeplitz_svd, conv2d_kernel_svd

from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import linear_svd_projection, conv2d_toeplitz_svd_projection, conv2d_kernel_svd_projection

from peepholelib.peepholes.parsers import trim_corevectors, trim_channelwise_corevectors, trim_kernel_corevectors
from peepholelib.peepholes.classifiers.tkmeans import KMeans as tKMeans 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes

from peepholelib.utils.samplers import random_subsampling 
from peepholelib.utils.analyze import conceptogram_protoclass_score as conceptogram_eval
from peepholelib.utils.conceptograms import plot_conceptogram

# torch stuff
import torch
from torchvision.models import vgg16
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
    name_model = 'vgg16'
    pretrained = True
    dataset = 'CIFAR100' 
    seed = 29
    bs = 256
    n_threads = 32

    features_svd_rank = 200 
    classifier_svd_rank = 200
    n_cluster = 200
    features_cv_dim = 196
    classifier_cv_dim = 150

    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
    #model_name = 'SV_model=vit_b_16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau_withInfo.pth'
    
    svds_path = Path('/srv/newpenny/XAI/generated_data/temp_results/toeplitz_conv') #svds_path = '/srv/newpenny/XAI/Peephole-Analysis' #Path.cwd()/'../data'
    svds_name = 'svds' 
    #svds_name = f'svds/{name_model}' 
   
    cvs_path = Path('/srv/newpenny/XAI/generated_data/temp_results/toeplitz_conv/corevectors') #Path.cwd()/f'../data/{name_model}/corevectors' 
    cvs_name = 'corevectors'
   
    drill_path = Path('/srv/newpenny/XAI/generated_data/temp_results/toeplitz_conv/drillers') #Path.cwd()/f'../data/{name_model}/drillers' #Path('/srv/newpenny/XAI/Peephole-Analysis/drillers_on_superclass') #
    drill_name = 'classifier'

    phs_path = Path('/srv/newpenny/XAI/generated_data/temp_results/toeplitz_conv/peepholes') #Path.cwd()/f'../data/{name_model}/peepholes' #Path('/srv/newpenny/XAI/Peephole-Analysis/peepholes_on_superclass') #
    phs_name = 'peepholes'
    
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
    
    nn = vgg16()

    #nn = vit_b_16()
    n_classes = len(ds.get_classes())

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
    
#     target_layers = [ f'encoder.layers.encoder_layer_{i}.mlp.{j}'for i in range(12) for j in [0,3]]

#     target_layers.append('heads.head')
    target_layers = [
            #'features.0',
            #'features.2',
            #'features.5',
            #'features.7',
            'features.10',
            'features.12',
            'features.14',
            'features.17',
            'features.19',
            'features.21',
            'features.24',
            'features.26',
            'features.28',
            'classifier.0',
            'classifier.3',
            'classifier.6',
            ]
    model.set_target_modules(target_modules=target_layers, verbose=verbose)

    #--------------------------------
    # SVDs 
    #--------------------------------

    svd_fns = {}
    for _layer in target_layers:
        if 'features' in _layer:
            svd_fns[_layer] =  partial(
               conv2d_toeplitz_svd, 
               rank = features_svd_rank,
               channel_wise = False,
               device = device,
               ),
        elif 'classifier' in _layer:
            svd_fns[_layer] = partial(
                linear_svd,
                rank = classifier_svd_rank,
                device = device,
                ),
    
    model.get_svds(
            path = svds_path,
            name = svds_name,
            target_modules = target_layers,
            sample_in = ds._dss['train'][0][0],
            svd_fns = svd_fns,
            verbose = verbose
            )

#     for k in model._svds.keys():
#         for kk in model._svds[k].keys():
#             print('svd shapes: ', k, kk, model._svds[k][kk].shape)
       
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
    reduction_fns = {}
    for _layer in target_layers:
        if 'features' in _layer:
            reduction_fns[_layer] =  partial(
               conv2d_toeplitz_svd_projection, 
               svd = model._svds[_layer], 
               layer=model._target_modules[_layer],
               use_s = True, 
               device=device
               )
        elif 'classifier' in _layer:
            reduction_fns[_layer] = partial(
                linear_svd_projection,
                svd = model._svds[_layer],
                use_s = True, 
                device=device
                )
    

#     reduction_fns = { f'encoder.layers.encoder_layer_{i}.mlp.{j}': partial(
#                 svd_Linear_ViT,
#                 reduct_m=model._svds[f'encoder.layers.encoder_layer_{i}.mlp.{j}']['Vh'], 
#                 device=device
#                 ) for i in range(12) for j in [0,3]}
#     reduction_fns['heads.head'] = partial(svd_Linear,
#                                         reduct_m=model._svds['heads.head']['Vh'], 
#                                         device=device
#                                         )
    
    #--------------------------------
    # Corevectors attacks 
    #--------------------------------
    
    for atk_type, atk_dss in atk_dss_dict.items():
        print(atk_dss._dss['test'].keys())
        # for k in atk_dss._dss.keys():
        #      atk_dss._dss[k] = atk_dss._dss[k][0:100] 
        # print(atk_dss._dss['test'].keys())
        
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
                        n_threads = n_threads,
                        save_input = True,
                        save_output = False,
                        verbose = verbose
                        )

                cv.normalize_corevectors(
                        target_layers = target_layers,
                        from_file=cvs_path/(cvs_name+'.normalization.pt'),
                        verbose=True
                        )
                quit()

        #--------------------------------
        # Peepholes
        #--------------------------------
        features_cv_dim = 196
        classifier_cv_dim = 150

        cv_parsers = {}
        for _layer in target_layers:
                if 'features' in _layer:
                        cv_parsers[_layer] =  partial(
                                                trim_corevectors,
                                                module = _layer,
                                                cv_dim = features_cv_dim, 
                                                )
                elif 'classifier' in _layer:
                        cv_parsers[_layer] = partial(
                                                trim_corevectors,
                                                module = _layer,
                                                cv_dim = classifier_cv_dim
                                                )

        # cv_parsers = { f'encoder.layers.encoder_layer_{i}.mlp.{j}': partial(
        #         trim_corevectors,
        #         module = f'encoder.layers.encoder_layer_{i}.mlp.{j}',
        #         cv_dim = cv_dim
        #         ) for i in range(12) for j in [0,3]} 
    
        # cv_parsers['heads.head'] = partial(
        #                 trim_corevectors,
        #                 module = f'heads.head',
        #                 cv_dim = cv_dim
        #                 )

        feature_sizes = {}
        for _layer in target_layers:
                if 'features' in _layer:
                        feature_sizes[_layer] =  features_cv_dim
                elif 'classifier' in _layer:
                        feature_sizes[_layer] = classifier_cv_dim
        feature_sizes['classifier.6'] = n_classes
        

        # feature_sizes = { f'encoder.layers.encoder_layer_{i}.mlp.{j}': cv_dim for i in range(12) for j in [0,3]}

        # feature_sizes['heads.head'] = cv_dim
        
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
               
                # if (drill_path/(drillers[peep_layer]._clas_path)).exists():
                #         print(f'Loading Classifier for {peep_layer}') 
                #         drillers[peep_layer].load()
                # else:
                #         RuntimeError(f'The file {drill_path/(drillers[peep_layer]._clas_path)} does not exist')

        peepholes = Peepholes(
            path = phs_path_,
            name = phs_name,
            device = device
            )

        with corevecs as cv, peepholes as ph:
                cv.load_only(
                        loaders = ['val','test'],
                        verbose = True
                        ) 
                for drill_key, driller in drillers.items():
                        print(driller._clas_path)
                        if (drill_path/(driller._clas_path)).exists():
                                print(f'Loading Classifier for {drill_key}') 
                                driller.load()
                        else:
                              raise RuntimeError(f'The file {drill_path/(driller._clas_path)} does not exist')

                ph.get_peepholes(
                        corevectors = cv,
                        target_modules = target_layers,
                        batch_size = bs,
                        drillers = drillers,
                        n_threads = n_threads,
                        verbose = verbose
                        )

                ph.get_scores(
                        batch_size = bs,
                        verbose=verbose
                        )
        
