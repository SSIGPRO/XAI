import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial
from matplotlib import pyplot as plt
import numpy as np
from contextlib import ExitStack

# Our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.datasets.transforms import vgg16_cifar100 as ds_transform 
from peepholelib.models.model_wrap import ModelWrap 

from peepholelib.adv_atk.attacks_base import ftd

from peepholelib.adv_atk.PGD import myPGD
from peepholelib.adv_atk.BIM import myBIM
from peepholelib.adv_atk.DeepFool import myDeepFool
from peepholelib.adv_atk.CW import myCW

from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import svd_Linear, svd_Conv2D

from peepholelib.peepholes.parsers import trim_corevectors, trim_channelwise_corevectors
from peepholelib.peepholes.classifiers.tkmeans import KMeans as tKMeans 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes

from peepholelib.utils.samplers import random_subsampling 
from peepholelib.utils.analyze import evaluate, evaluate_dists 

# torch stuff
import torch
from torch.utils.data import DataLoader
from torchvision.models import vgg16
from cuda_selector import auto_cuda
from sklearn.manifold import TSNE


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
    dataset = 'CIFAR100' 
    name_model =  'vgg16'
    seed = 29
    bs = 512 
    n_threads = 32

    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
    
    svds_path = Path.cwd()/'../data'
    svds_name = 'svds' 
    
    cvs_path = Path.cwd()/f'../data/{name_model}/corevectors'
    cvs_name = 'corevectors'

    drill_path = Path.cwd()/f'../data/{name_model}/drillers'
    drill_name = 'classifier'

    phs_path = Path.cwd()/f'../data/{name_model}/peepholes'
    phs_name = 'peepholes'
    
    verbose = True 
    
    # Peepholelib
    target_layers = [
            'classifier.6'
            ]

    #--------------------------------
    # Dataset 
    #--------------------------------

    ds = Cifar(
            data_path = ds_path,
            dataset = dataset
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
        
    #--------------------------------
    # SVDs 
    #--------------------------------
    t0 = time()
    model.get_svds(
            path = svds_path,
            name = svds_name,
            target_modules = target_layers,
            sample_in = ds._dss['train'][0][0],
            rank = 10,
            channel_wise = True,
            verbose = verbose
            )
    print('time: ', time()-t0)
    
    '''
    print('\n----------- svds:')
    for k in model._svds.keys():
        for kk in model._svds[k].keys():
            print('svd shapes: ', k, kk, model._svds[k][kk].shape)
        s = model._svds[k]['s']
        if len(s.shape) == 1:
            plt.figure()
            plt.plot(s, '-')
            plt.xlabel('Rank')
            plt.ylabel('EigenVec')
        else:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            _x = torch.linspace(0, s.shape[1]-1, s.shape[1])
            for r in range(s.shape[0]):
                plt.plot(xs=_x, ys=s[r,:], zs=r, zdir='y')
            ax.set_xlabel('Rank')
            ax.set_ylabel('Channel')
            ax.set_zlabel('EigenVec')
        plt.savefig((svds_path/(svds_name+'/'+k+'.png')).as_posix(), dpi=300, bbox_inches='tight')
        plt.close()
    '''
    #--------------------------------
    # CoreVectors 
    #--------------------------------
    #random_subsampling(ds, 0.025)
    
    cv = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            model = model,
            )
#     cv_atk = {}

#     for atk_type, atk_dss in atk_dss_dict.items():
#         print(atk_dss._dss['test'].keys())
        
#         print('--------------------------------')
#         print('atk type: ', atk_type)
#         print('--------------------------------')
#         cvs_path_ = cvs_path/f'{atk_type.replace('my', "")}'

#         cv_atk[atk_type] = CoreVectors(
#                                         path = cvs_path_,
#                                         name = cvs_name,
#                                         model = model,
#                                         )
    
    with ExitStack() as stack:
        stack.enter_context(cv)

        cv.parse_ds(
                batch_size = bs,
                datasets = ds,
                n_threads = n_threads,
                verbose = verbose
                )

        # This occupies a lot of space. Only do if you need it
        # copy dataset to activatons file
        cv.get_activations(
                batch_size = bs,
                n_threads = n_threads,
                save_input = True,
                save_output = False,
                verbose = verbose
                ) 
        Vh = model._svds['classifier.6']['Vh']

        act_data = cv._dss['val']['in_activations']['classifier.6']
        acts_flat = act_data.flatten(start_dim=1)
        n_act = act_data.shape[0]
        ones = torch.ones(n_act, 1)
        _acts = torch.hstack((acts_flat, ones))
        
        _cv = _acts @ Vh.T

        var_cv = torch.var(_cv, dim=0)
        print(var_cv)

        '''
        print('var cv: ', var_cv.shape, var_cv)
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].plot(var_cv/var_cv.max(), '-')
        axs[0].plot(model._svds['classifier.6']['s']/model._svds['classifier.6']['s'].max(), '--')
        axs[0].set_xlabel('Rank')
        axs[0].set_ylabel('Variance')
        axs[0].legend(['CoreVectors', 'SVD'])
        axs[1].scatter(var_cv/var_cv.max(), model._svds['classifier.6']['s']/model._svds['classifier.6']['s'].max())
        axs[1].set_xlabel('CoreVectors')
        axs[1].set_ylabel('SVD')
        axs[1].set_title('CoreVectors vs SVD')
        plt.savefig('confronto.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        '''

        '''
        per_class_var = {}
        for i in range(n_classes):
                act_data = cv._dss['val']['in_activations']['classifier.6'][i == cv._dss['val']['label']]
                acts_flat = act_data.flatten(start_dim=1)
                n_act = act_data.shape[0]
                ones = torch.ones(n_act, 1)
                _acts = torch.hstack((acts_flat, ones))
                
                _cv = _acts @ Vh.T
                per_class_var[i] = torch.var(_cv, dim=0)
        
        fig, axs = plt.subplots(10, 10, figsize=(20,20))

        axs = axs.flatten()
        for i in range(n_classes):
                axs[i].scatter(var_cv/var_cv.max(), per_class_var[i]/per_class_var[i].max())
                axs[i].plot((1, 0), (1, 0), '--', color='red')
                # axs[i].set_xlabel('CoreVectors overall')
                # axs[i].set_ylabel(f'{i} corevectors')
                # axs[i].set_title(f'{i} analysis')

        plt.tight_layout()
        plt.savefig('trying.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        k=5
        per_class_var = torch.zeros((n_classes, Vh.shape[0]))  # already in your code
        topk_indices = torch.zeros((n_classes, k), dtype=torch.long)

        for i in range(n_classes):
                act_data = cv._dss['val']['in_activations']['classifier.6'][i == cv._dss['val']['label']]
                acts_flat = act_data.flatten(start_dim=1)
                n_act = act_data.shape[0]
                ones = torch.ones(n_act, 1)
                _acts = torch.hstack((acts_flat, ones))
                
                _cv = _acts @ Vh.T
                per_class_var[i] = torch.var(_cv, dim=0)     
                _, idx = torch.topk(per_class_var[i], k=k)    
                topk_indices[i] = idx 

                print(f'Class {i}: top-{k} indices = {idx.tolist()}')
        from collections import defaultdict

        groups = defaultdict(list)
        for cls in range(n_classes):
                key = tuple(topk_indices[cls].tolist())
                groups[key].append(cls)

        for key, class_list in groups.items():
                if len(class_list) > 1:
                        print(f"Classes {class_list} all have top-{k} = {key}")
        '''
        