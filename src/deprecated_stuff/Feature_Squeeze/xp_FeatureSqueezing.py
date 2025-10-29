import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial
from matplotlib import pyplot as plt
import math
import numpy as np

# Our stuff
from peepholelib.adv_atk.PGD import myPGD
from peepholelib.adv_atk.BIM import myBIM
from peepholelib.adv_atk.DeepFool import myDeepFool
from peepholelib.adv_atk.CW import myCW

from peepholelib.datasets.cifar import Cifar
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.models.viz import viz_singular_values

from peepholelib.coreVectors.coreVectors import CoreVectors

from peepholelib.utils.samplers import random_subsampling 

from peepholelib.featureSqueezing.FeatureSqueezingDetector import FeatureSqueezingDetector
from peepholelib.featureSqueezing.preprocessing import NLM_filtering_torch, NLM_filtering_cv, bit_depth_torch, MedianPool2d

# torch stuff
import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights, vit_b_16
from cuda_selector import auto_cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple

from torch_nlm import nlm2d
import cv2
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc


if __name__ == "__main__":
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
    pretrained = True
    dataset = 'CIFAR100' 
    #name_model = 'vgg16'
    name_model = 'ViT'
    seed = 29
    bs = 32

    cvs_path = Path.cwd()/f'../data/{name_model}/corevectors' #Path('/srv/newpenny/XAI/Peephole-Analysis/channel_wise/corevectors') 
    cvs_name = 'coreavg'

    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'SV_model=vit_b_16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau_withInfo.pth'
    #model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth' 
    
    verbose = True 
    num_workers = 4
    
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
    
    #nn = vgg16()

    nn = vit_b_16()
    n_classes = len(ds.get_classes())

    model = ModelWrap(
            model = nn,
            device = device
            )

    model.update_output(
            output_layer = 'heads.head', #'classifier.6', 
            to_n_classes = n_classes,
            overwrite = True 
            )

    model.load_checkpoint(
            name = model_name,
            path = model_dir,
            verbose = verbose
            )

    dss = ds._dss
    #dss = random_subsampling(ds._dss, 0.05)
    dss = {#'train': dss['train'],
            'val': dss['val'],
           'test': dss['test'],
              }
    print(dss)
    loaders = {}
    for key in dss.keys():
        loaders[key] = DataLoader(dss[key], batch_size=bs, shuffle=False) 
    #--------------------------------
    # Attacks 
    #--------------------------------   
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
        
        atk_dss_dict[atk_type] = atk._dss

    median = MedianPool2d(kernel_size=3, stride=1, padding=1)

    mean_t = torch.tensor((0.438, 0.418, 0.377)).view(3,1,1).to(device)
    std_t  = torch.tensor((0.300, 0.287, 0.294)).view(3,1,1).to(device)

    prepro_dict = {
                'median': median,
                'bit_depth': partial(bit_depth_torch, bits=5),
                'nlm': partial(NLM_filtering_torch, kernel_size=11, std=4.0, kernel_size_mean=3, sub_filter_size=32),
                #'nlm_cv': partial(NLM_filtering_cv, mean_t, std_t, h=11, hColor=11, templateWindowSize=3, searchWindowSize=33)
                }
    
    detector = FeatureSqueezingDetector(model=model._model,
                                        prepro_dict=prepro_dict)
    ori_list = []

    print('---------------------')
    print('Computing the scores')
    print('---------------------')

    for data in tqdm(loaders['test']):
        
        inputs, labels = data
        inputs = inputs.to(device)

        output = detector(inputs)

        ori_list.append(output.detach().cpu())

    s_ori = torch.cat(ori_list, dim=0)

    ''' ### AUC on each attack samples
    for atk, atk_dss in atk_dss_dict.items():
        print('--------------')
        print(atk)
        print('--------------')
        loader = DataLoader(atk_dss['test'], batch_size=bs, collate_fn=lambda x:x, shuffle=False, num_workers=4) 

        atk_list = []

        for data in tqdm(loader):            
            
            inputs = data['image'].to(device)

            output = detector(inputs)

            atk_list.append(output.detach().cpu())

        s_atk = torch.cat(atk_list, dim=0)
        labels_pos = torch.ones(len(s_ori))
        labels_neg = torch.zeros(len(s_atk))
        all_scores = torch.cat([s_ori, s_atk], dim=0)
        all_labels = torch.cat([labels_pos, labels_neg], dim=0)

        # Convert tensors to numpy arrays (using .detach().cpu().numpy() if needed)
        all_scores_np = all_scores.detach().cpu().numpy()
        
        all_labels_np = all_labels.detach().cpu().numpy()

        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(all_labels_np, all_scores_np)
        roc_auc = auc(fpr, tpr)

        print("AUC:", roc_auc)

        # Plot ROC curve
        plt.figure()
        fig, axs = plt.subplots(1,2, figsize=(10,10))
        axs[0].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        axs[0].plot([0, 1], [0, 1], 'k--', label='Chance')
        axs[0].set_xlabel('False Positive Rate')
        axs[0].set_ylabel('True Positive Rate')
        axs[0].set_title('Receiver Operating Characteristic')
        axs[0].legend()
        axs[1].hist(s_ori.numpy(), bins=100, label='ori')
        axs[1].hist(s_atk.numpy(), bins=100, label=f'{atk}')
        axs[1].legend()
        fig.savefig(f'../data/ViT/img/AUC/Feature_squeezing_attack={atk}_median.png')
    '''

    for atk, atk_dss in atk_dss_dict.items():
        print('--------------------------------')
        print('atk type: ', atk)
        print('--------------------------------')
        cvs_path_ = cvs_path/f'{atk_type.replace('my', "")}'

        corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )

        corevecs_atk = CoreVectors(
            path = cvs_path_,
            name = cvs_name,
            )
        
        with corevecs as cv, corevecs_atk as cv_atk:
                cv.load_only(
                        loaders = ['test'],
                        verbose = True
                        )
                cv_atk.load_only(
                        loaders = ['test'],
                        verbose = True
                        )
                idx = torch.argwhere((cv._dss['test']['result']==1) & (cv_atk._dss['test']['attack_success']==1))

        loader = DataLoader(atk_dss['test'], batch_size=bs, collate_fn=lambda x:x, shuffle=False, num_workers=4) 

        atk_list = []

        for data in tqdm(loader):            
            
            inputs = data['image'].to(device)

            output = detector(inputs)

            atk_list.append(output.detach().cpu())

        s_atk = torch.cat(atk_list, dim=0)

        s_ori_ = s_ori[idx]
        s_atk = s_atk[idx]
        print(f's_ori: {s_ori_.shape}, s_atk: {s_atk.shape}')

        labels_pos = torch.zeros(len(s_ori_))
        labels_neg = torch.ones(len(s_atk))
        all_scores = torch.cat([s_ori_, s_atk], dim=0)
        all_labels = torch.cat([labels_pos, labels_neg], dim=0)

        # Convert tensors to numpy arrays (using .detach().cpu().numpy() if needed)
        all_scores_np = all_scores.detach().cpu().numpy()
        
        all_labels_np = all_labels.detach().cpu().numpy()

        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(all_labels_np, all_scores_np)
        roc_auc = auc(fpr, tpr)

        print("AUC:", roc_auc)

        # Plot ROC curve
        plt.figure()
        fig, axs = plt.subplots(2,1, figsize=(7,10))
        axs[0].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        axs[0].plot([0, 1], [0, 1], 'k--', label='Chance')
        axs[0].set_xlabel('False Positive Rate')
        axs[0].set_ylabel('True Positive Rate')
        axs[0].set_title('Receiver Operating Characteristic')
        axs[0].legend()
        axs[1].hist(s_ori.numpy(), bins=100, label='ori')
        axs[1].hist(s_atk.numpy(), bins=100, label=f'{atk}', alpha=0.7)
        axs[1].legend()
        fig.savefig(f'../data/{name_model}/img/AUC/Feature_squeezing_attack={atk.replace("my", "")}_combined.png')
