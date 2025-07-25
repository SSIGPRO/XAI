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

from peepholelib.utils.samplers import random_subsampling 

# torch stuff
import torch
from torch.utils.data import DataLoader
from torchvision.models import vgg16
from cuda_selector import auto_cuda
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score

from pyod.models.deep_svdd import DeepSVDD

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
    
    cvs_path = Path('/srv/newpenny/XAI/generated_data/temp_results/toeplitz_conv/corevectors') #Path.cwd()/f'../data/{name_model}/corevectors'
    cvs_name = 'corevectors'
    
    verbose = True 
    
    # Peepholelib
    target_layers = [
        #     'features.26',
        #     'features.28',
            'classifier.0',
            'classifier.3',
            'classifier.6',
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
    # CoreVectors 
    #--------------------------------
    #random_subsampling(ds, 0.025)
    
    cv = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            model = model,
            )
    cv_atk = {}

    for atk_type, atk_dss in atk_dss_dict.items():
        print(atk_dss._dss['test'].keys())
        
        print('--------------------------------')
        print('atk type: ', atk_type)
        print('--------------------------------')
        cvs_path_ = cvs_path/f'{atk_type.replace('my', "")}'

        cv_atk[atk_type] = CoreVectors(
                                        path = cvs_path_,
                                        name = cvs_name,
                                        model = model,
                                        )
    layer = 'classifier.0'
    cv_dim = 50
    with ExitStack() as stack:
        stack.enter_context(cv)

        cv.load_only(
                     loaders = ['val', 'test'],
                     verbose = True
                        )
        data_val = torch.concat([cv._corevds['val'][layer][:,:cv_dim] for layer in (target_layers if target_layers is not None else cv.keys())],dim=1)
        data_test = torch.concat([cv._corevds['test'][layer][:,:cv_dim] for layer in (target_layers if target_layers is not None else cv.keys())],dim=1)
        
        n_features = data_val.shape[1]
        # detector = DeepSVDD(
        #                 n_features=cv_dim,
        #                 use_ae=False,            # one‐class SVDD (no autoencoder branch)
        #                 hidden_neurons=[cv_dim//2, cv_dim//4], # you can tune this to your feature size
        #                 hidden_activation='relu',
        #                 output_activation='tanh',
        #                 optimizer='adam',
        #                 epochs=100,
        #                 batch_size=256,
        #                 dropout_rate=0.1,
        #                 preprocessing=False,
        #                 l2_regularizer=1e-6,
        #                 contamination=0.001,       # expected fraction of “outliers”
        #                 # device=torch.device("cpu"),    # ← no more None
        #                 # dtype=torch.float32            # ← no more None
        #                 )
        
        detector = DeepSVDD(
                n_features=n_features,
                use_ae=True,
                hidden_neurons=[n_features//2, n_features//4, n_features//8],  # More capacity
                hidden_activation='leaky_relu',
                output_activation='relu',
                optimizer='adam',
                # lr=5e-4,  # Explicit learning rate
                epochs=20,
                batch_size=128,  # Smaller batch
                dropout_rate=0.2,
                preprocessing=True,  # Enable preprocessing
                l2_regularizer=1e-6,
                contamination=0.01,  # Higher contaminati
                )
        idx = torch.argwhere(cv._dss['val']['result']==1).squeeze()
        detector.fit(data_val[idx].detach().cpu().numpy()) #cv._corevds['val'][layer].detach().cpu().numpy()[idx]

        # Get scores for the normal validation set (for AUC ground truth)
        normal_val_scores = detector.decision_function(data_val.detach().cpu().numpy())
        # Get scores for the normal test set
        normal_test_scores = detector.decision_function(data_test.detach().cpu().numpy())
    auc_results = {}
    for atk, cv_atk_ in cv_atk.items():
        with ExitStack() as stack:
            stack.enter_context(cv_atk_)
            cv_atk_.load_only(
                              loaders = ['val', 'test'],
                              verbose = True
                              )
            stack.enter_context(cv)

            cv.load_only(
                        loaders = ['val', 'test'],
                        verbose = True
                                )
            
        #     cv_dim = cv_atk_._corevds['val'][layer].shape[1]
            data_atk = torch.concat([cv_atk_._corevds['test'][layer][:,:cv_dim] for layer in (target_layers if target_layers is not None else cv.keys())],dim=1)
            
            # Get scores for the adversarial test set
            idx = torch.argwhere((cv_atk_._dss['test']['attack_success'] == 1) & (cv._dss['test']['result']==1)).squeeze()
            adv_test_scores = detector.decision_function(data_atk.detach().cpu().numpy())[idx]
            normal_test_scores_ = normal_test_scores.copy()[idx]


            # Combine normal and adversarial scores for AUC calculation (using test sets)
            # True labels: 0 for normal, 1 for adversarial
            y_true = np.concatenate((np.zeros(len(normal_test_scores_)), np.ones(len(adv_test_scores))))
            y_scores = np.concatenate((normal_test_scores_, adv_test_scores))

            # Compute AUC
            auc = roc_auc_score(y_true, y_scores)
            auc_results[atk] = auc
            print(f"AUC for {atk.replace('my', '')} attack: {auc:.4f}")

        print("\nAUC Results:")
        for atk_type, auc_score in auc_results.items():
            print(f"{atk_type.replace('my', '')}: {auc_score:.4f}")


            