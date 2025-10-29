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
from peepholelib.peepholes.parsers import get_images 
from peepholelib.peepholes.DeepMahalanobisDistance.DMD import DeepMahalanobisDistance as DMD
from peepholelib.peepholes.peepholes import Peepholes

# torch stuff
import torch
from torchvision.models import vgg16, VGG16_Weights
from cuda_selector import auto_cuda

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
    seed = 29
    bs = 512 

    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
    
    cvs_path = Path.cwd()/'../data/corevectors' #Path('/srv/newpenny/XAI/Peephole-Analysis/corevectors')  
    cvs_name = 'coreavg'

    drill_path = Path.cwd()/'../data/drillers' #Path('/srv/newpenny/XAI/Peephole-Analysis/drillers_on_class') 
    drill_name = 'DMD'

    phs_path = Path.cwd()/'../data/peepholes' #Path('/srv/newpenny/XAI/Peephole-Analysis/peepholes_on_class') 
    phs_name = 'peepholes_avg' #_noBackprop

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
           'features.24',
           'features.26',
           'features.28',
            ]
    model.set_target_modules(target_modules=target_layers, verbose=verbose)

    direction = {'save_input':False, 'save_output':True}
    model.add_hooks(**direction, verbose=False) 
    
    dry_img, _ = ds._dss['train'][0]
    dry_img = dry_img.reshape((1,)+dry_img.shape)
    model.dry_run(x=dry_img)

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

    peep_layers = ['features.24','features.26','features.28','output']

    drillers = {}

    shapes = {'features.24': 512,
              'features.26': 512,
              'features.28': 512,
              'output': 100,
              }

    for peep_layer in peep_layers:
                drillers[peep_layer] = DMD(
                                        path = drill_path,
                                        name = drill_name+'.'+peep_layer,
                                        nl_model = n_classes,
                                        n_features = shapes[peep_layer],
                                        parser = get_images,
                                        parser_kwargs = {},
                                        model = model,
                                        layer = peep_layer,
                                        magnitude = 0.004,
                                        std_transform = [0.300, 0.287, 0.294],
                                        device = device,
                                        )
    for atk in attack_list:
        print('--------------------------------')
        print('atk type: ', atk)
        print('--------------------------------')
        phs_path_ = phs_path/f'{atk}'
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
        with peepholes as ph, peepholes_atk as ph_atk:
                ph.load_only(
                        loaders = ['val'],
                        verbose = True
                        )
                ph_atk.load_only(
                        loaders = ['val'],
                        verbose = True
                        )

                for peep_layer in peep_layers:
                        s_ori = torch.max(ph._phs['val'][peep_layer]['peepholes'], dim=1).values
                        s_atk = torch.max(ph_atk._phs['val'][peep_layer]['peepholes'], dim=1).values
                        
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
                        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                        plt.plot([0, 1], [0, 1], 'k--', label='Chance')
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title('Receiver Operating Characteristic')
                        plt.legend(loc="lower right")
                        plt.savefig(f'../data/img/AUC/DMD_attack={atk}_n_classes={n_classes}_{peep_layer}.png')
    