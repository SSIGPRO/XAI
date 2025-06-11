import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial
from matplotlib import pyplot as plt
import numpy as np
from contextlib import ExitStack
from tqdm import tqdm

# Our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.datasets.transforms import vgg16_cifar100 as ds_transform 
from peepholelib.models.model_wrap import ModelWrap 

from peepholelib.adv_atk.attacks_base import ftd

from peepholelib.adv_atk.PGD import myPGD
from peepholelib.adv_atk.BIM import myBIM
from peepholelib.adv_atk.DeepFool import myDeepFool
from peepholelib.adv_atk.CW import myCW

from peepholelib.HeadSqueezing.HeadSqueezingDetector import HeadSqueezingDetector

from peepholelib.utils.samplers import random_subsampling 
from peepholelib.utils.analyze import evaluate, evaluate_dists 

# torch stuff
import torch
from torch.utils.data import DataLoader
from torchvision.models import vgg16
from cuda_selector import auto_cuda
from sklearn.metrics import roc_curve, auc


if __name__ == "__main__":
#     use_cuda = torch.cuda.is_available()
#     device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
#     print(f"Using {device} device")
    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 1
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
    
    #---------------------------
    # Detector
    #---------------------------
    for atk_name, atk in atk_dss_dict.items():
        print(atk_name, atk._dss['test']['attack_success'].sum())
    mode = 'random'
    k=30

    detector = HeadSqueezingDetector(model=model, device=device)

    detector.layer_SVD(
                        path = svds_path,
                        name = svds_name,
                        target_modules = target_layers,
                        sample_in = dss['train'][0],
                        rank = k
                        )   
    s_ori = []
    for data in tqdm(loaders['test']):

        image, _ = data
        output = detector(image=image, mode=mode)
        s_ori.append(output)
                
    s_ori = torch.concat(s_ori).detach().cpu().numpy()
    label_ori = np.ones(len(s_ori))
    
    for atk_name, atk in atk_dss_dict.items():
        print(f'---------------\n {atk_name} \n ----------------')
        loader = DataLoader(atk._dss['test'], batch_size=bs, collate_fn=lambda x:x, shuffle=False)
        s_atk = []
        for data in tqdm(loader):

                image = data['image']
                output = detector(image=image, mode=mode)
                s_atk.append(output)
        s_atk = torch.concat(s_atk).detach().cpu().numpy()
        label_atk = np.zeros(len(s_atk))
        score = np.concatenate((s_ori, s_atk), axis=0)
        labels = np.concatenate((label_ori, label_atk), axis=0)

        fpr, tpr, thresholds = roc_curve(labels, score)
        roc_auc = auc(fpr, tpr)

        print("AUC:", roc_auc)

        # Plot ROC curve
        fig, axs = plt.subplots(1, 2, figsize=(20,20))
        
        axs[0].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        axs[0].plot([0, 1], [0, 1], 'k--', label='Chance')
        axs[0].set_xlabel('False Positive Rate')
        axs[0].set_ylabel('True Positive Rate')
        axs[0].set_title('Receiver Operating Characteristic')
        axs[0].legend()
        axs[1].hist(s_ori, bins=50, label='ori', alpha=0.6)
        axs[1].hist(s_atk, bins=50, label=f'{atk_name}', alpha=0.6)
        axs[1].legend()
        plt.savefig(f'../data/img/AUC/HeadSqueezing_model={name_model}_atk={atk_name}_k={k}_mode={mode}.png')
