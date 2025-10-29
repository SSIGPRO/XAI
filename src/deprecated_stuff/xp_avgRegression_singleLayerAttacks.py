import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import math

# sklearn stuff 
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc

# scipy stuff 
from scipy.stats import entropy

# Our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.avgPooling import ChannelWiseMean_conv, cls_token_ViT, TokenWiseMean_ViT

from peepholelib.peepholes.parsers import get_images 
from peepholelib.peepholes.DeepMahalanobisDistance.DMD import DeepMahalanobisDistance as DMD
from peepholelib.peepholes.peepholes import Peepholes

from peepholelib.utils.samplers import random_subsampling 

# torch stuff
import torch
from torchvision.models import vgg16, VGG16_Weights, vit_b_16
from cuda_selector import auto_cuda

from sklearn.linear_model import LogisticRegressionCV

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    ds_path = '/srv/newpenny/dataset/CIFAR100'

    # model parameters
    #name_model = 'ViT'
    name_model = 'vgg16'
    pretrained = True
    dataset = 'CIFAR100' 
    seed = 29
    bs = 512 

    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
    #model_name = 'SV_model=vit_b_16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau_withInfo.pth'

    cvs_path = Path.cwd()/f'../data/{name_model}/corevectors' #Path('/srv/newpenny/XAI/Peephole-Analysis/corevectors')  
    cvs_name = 'coreavg'

    drill_path = Path.cwd()/f'../data/{name_model}/drillers' #Path('/srv/newpenny/XAI/Peephole-Analysis/channel_wise/drillers_on_class') 
    drill_name = 'DMD'

    phs_path = Path.cwd()/f'../data/{name_model}/peepholes' #Path('/srv/newpenny/XAI/Peephole-Analysis/peepholes_on_class') # 
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
    
    nn = vgg16()
    
#     nn = vit_b_16()
    n_classes = len(ds.get_classes())

    model = ModelWrap(
            model = nn,
            device = device
            )

    model.update_output(
            output_layer = 'classifier.6', #'heads.head'
            to_n_classes = n_classes,
            overwrite = True 
            )

    model.load_checkpoint(
            name = model_name,
            path = model_dir,
            verbose = verbose
            )
    
#     target_layers =['encoder.layers.encoder_layer_11.mlp.3']
    
    target_layers = ['features.28']
          
    model.set_target_modules(target_modules=target_layers, verbose=verbose)

    #--------------------------------
    # Attacks
    #--------------------------------
    attack_list = [
                'PGD',
                'CW',
                'DeepFool',
                'BIM'
                ]

    #--------------------------------
    # Peepholes
    #--------------------------------
    n_classes = 100

    peep_layers = target_layers

    drillers = {}

#     feature_sizes = {'encoder.layers.encoder_layer_11.mlp.3': 768}

    feature_sizes = {'features.28': 512}             

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
                                        magnitude = 0.004,
                                        std_transform = [0.229, 0.224, 0.225],
                                        device = device,
                                        parser_act = cls_token_ViT
                                        )
    for atk in attack_list:
        print('--------------------------------')
        print('atk type: ', atk)
        print('--------------------------------')
        phs_path_ = phs_path/f'{atk}'
        cvs_path_ = cvs_path/f'{atk}'

        corevectors = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )
        
        corevectors_atk = CoreVectors(path = cvs_path_,
                                      name = cvs_name,
                                      )
        
        peepholes = Peepholes(
                path = phs_path,
                name = phs_name,
                driller = drillers,
                target_modules = peep_layers,
                device = device
                )
        peepholes_atk = Peepholes(
                path = phs_path_,
                name = phs_name,
                driller = drillers,
                target_modules = peep_layers,
                device = device
                )
        
        with corevectors as cv, corevectors_atk as cv_atk, peepholes as ph, peepholes_atk as ph_atk:

                cv.load_only(loaders = ['val', 'test'],
                             verbose = True
                             )
                
                cv_atk.load_only(loaders = ['val', 'test'],
                                 verbose = True
                                 )

                ph.load_only(
                        loaders = ['val', 'test'],
                        verbose = True
                        )
                ph_atk.load_only(
                        loaders = ['val', 'test'],
                        verbose = True
                        )
                
                idx = torch.argwhere((cv._dss['test']['result']==1) & (cv_atk._dss['test']['attack_success']==1)).squeeze()

                s_ori = ph._phs['test'][target_layers[0]]['score_max'][idx]
                s_atk = ph_atk._phs['test'][target_layers[0]]['score_max'][idx]

                label_ori = np.zeros(len(s_ori))
                label_atk = np.ones(len(s_atk))

                s = np.concatenate((s_ori, s_atk), axis=0)
                test_label = np.concatenate((label_ori, label_atk), axis=0)

                fpr, tpr, thresholds = roc_curve(test_label, s)
                roc_auc = auc(fpr, tpr)

                print("AUC:", roc_auc)

                plt.figure()
                fig, axs = plt.subplots(2,1, figsize=(7,10))
                axs[0].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                axs[0].plot([0, 1], [0, 1], 'k--', label='Chance')
                axs[0].set_xlabel('False Positive Rate')
                axs[0].set_ylabel('True Positive Rate')
                axs[0].set_title('Receiver Operating Characteristic')
                axs[0].legend()
                axs[1].hist(s_ori, bins=30, label='ori')
                axs[1].hist(s_atk, bins=30, label=f'{atk}', alpha=0.7)
                axs[1].legend()
                fig.savefig(f'../data/{name_model}/img/AUC/SingleLayer_DMD_attack={atk}.png')

                # # Plot ROC curve
                # plt.figure()
                # plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                # plt.plot([0, 1], [0, 1], 'k--', label='Chance')
                # plt.xlabel('False Positive Rate')
                # plt.ylabel('True Positive Rate')
                # plt.title('Receiver Operating Characteristic')
                # plt.legend(loc="lower right")
                # plt.savefig(f'../data/{name_model}/img/AUC/_Regression_attack={atk}_DMD.png')
                
                
