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
from peepholelib.coreVectors.dimReduction.svds import svd_Linear, svd_Conv2D

from peepholelib.peepholes.parsers import trim_corevectors, trim_channelwise_corevectors
from peepholelib.peepholes.classifiers.tkmeans import KMeans as tKMeans 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
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
    pretrained = True
    dataset = 'CIFAR100' 
    seed = 29
    bs = 512 

    drill_path = Path.cwd()/'../data/ViT/drillers' #Path('/srv/newpenny/XAI/Peephole-Analysis/drillers_on_class') 
    drill_name = 'classifier'

    phs_path = Path.cwd()/'../data/ViT/peepholes_100' #Path('/srv/newpenny/XAI/Peephole-Analysis/peepholes_on_class')
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
    n_cluster = 150

    parser_cv = trim_corevectors
    peep_layers = [ f'encoder.layers.encoder_layer_{i}.mlp.{j}' for i in range(12) for j in [0,3]]
    peep_layers.append('heads.head')
#     peep_layers = {
#          'features.24': 100,
#          'features.26': 100,
#          'features.28': 100,
#          'classifier.0': 100, 
#          'classifier.3': 100,
#          'classifier.6': 100, 
#          }
    drillers = {}
    for peep_layer in peep_layers:
        
        parser_kwargs = {'module': peep_layer, 'cv_dim':100, 'label_key':'superclass'}

        drillers[peep_layer] = tGMM(
                                path = drill_path,
                                name = drill_name+'.'+peep_layer,
                                nl_classifier = n_cluster,
                                nl_model = n_classes,
                                n_features = 100,
                                parser = parser_cv,
                                parser_kwargs = parser_kwargs,
                                device = device,
                                batch_size = 512,
                                )
    for atk in attack_list:
        print('--------------------------------')
        print('atk type: ', atk)
        print('--------------------------------')
        phs_path_ = phs_path/f'{atk}'
        peepholes = Peepholes(
                path = phs_path,
                name = f'{phs_name}.nc_{n_cluster}',
                driller = drillers,
                target_modules = peep_layers,
                device = device
                )
        peepholes_atk = Peepholes(
                path = phs_path_,
                name = f'{phs_name}.nc_{n_cluster}',
                driller = drillers,
                target_modules = peep_layers,
                device = device
                )
        with peepholes as ph, peepholes_atk as ph_atk:
                ph.load_only(
                        loaders = ['val','test'],
                        verbose = True
                        )
                ph_atk.load_only(
                        loaders = ['val','test'],
                        verbose = True
                        )
                ### Dataset preparation ###

                train_ori = torch.stack([ph._phs['val'][layer]['score_max'] for layer in peep_layers],dim=1).detach().cpu().numpy()
                train_atk = torch.stack([ph_atk._phs['val'][layer]['score_max'] for layer in peep_layers],dim=1).detach().cpu().numpy()
                
                train_data = np.concatenate((train_ori, train_atk), axis=0)
                print(train_data)
                label_ori = np.zeros(len(train_ori))
                label_atk = np.ones(len(train_atk))
                train_label = np.concatenate((label_ori, label_atk), axis=0)

                val_ori = torch.stack([ph._phs['test'][layer]['score_max'] for layer in peep_layers],dim=1).detach().cpu().numpy()
                val_atk = torch.stack([ph_atk._phs['test'][layer]['score_max'] for layer in peep_layers],dim=1).detach().cpu().numpy()
                val_data = np.concatenate((val_ori, val_atk), axis=0)
                label_ori = np.zeros(len(val_ori))
                label_atk = np.ones(len(val_atk))
                val_label = np.concatenate((label_ori, label_atk), axis=0)

                lr = LogisticRegressionCV(n_jobs=-1).fit(train_data, train_label)
                y_pred = lr.predict_proba(train_data)[:, 1]
                
                y_pred = lr.predict_proba(val_data)[:, 1]
                fpr, tpr, thresholds = roc_curve(val_label, y_pred)
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
                plt.savefig(f'../data/ViT/img/AUC/Regression_attack={atk}_n_classes={n_classes}_n_cluster={n_cluster}_cv_dim=100.png')
                
                
