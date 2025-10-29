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
from contextlib import ExitStack

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
    name_model = 'ViT'
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
    
    attacks_config = {'c0': {'train': ['PGD','BIM','CW'], 'test': 'DeepFool'},
                      'c1': {'train': ['BIM','CW','DeepFool'], 'test': 'PGD'},
                      'c2': {'train': ['CW','DeepFool','PGD'], 'test': 'BIM'},
                      'c3': {'train': ['DeepFool','PGD','BIM'], 'test': 'CW'}}
    
    '''
    attacks_config = {'c0': {'train': 'DeepFool', 'test': ['PGD','BIM','CW']},
                      'c1': {'train': 'PGD', 'test': ['BIM','CW','DeepFool']},
                      'c2': {'train': 'BIM', 'test': ['CW','DeepFool','PGD']},
                      'c3': {'train': 'CW', 'test': ['DeepFool','PGD','BIM']}}
    '''

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
    
        
    '''
    ### Three training attacks and one testing attack
    for config in attacks_config:
                atk_train = attacks_config[config]['train']
                atk_test = attacks_config[config]['test']
                print(f'training: {atk_train} test: {atk_test}')

                peepholes = Peepholes(
                        path = phs_path,
                        name = f'{phs_name}.nc_{n_cluster}',
                        driller = drillers,
                        target_modules = peep_layers,
                        device = device
                        )
                peepholes_atks_train = {} 
                for atk_name in atk_train:
                        phs_path_ = phs_path/f'{atk_name}'

                        peepholes_atks_train[atk_name] = Peepholes(
                                        path = phs_path_,
                                        name = f'{phs_name}.nc_{n_cluster}',
                                        driller = drillers,
                                        target_modules = peep_layers,
                                        device = device
                                        )
                phs_path_ = phs_path/f'{atk_test}'
                
                peepholes_atks_test = Peepholes(
                                path = phs_path_,
                                name = f'{phs_name}.nc_{n_cluster}',
                                driller = drillers,
                                target_modules = peep_layers,
                                device = device
                                )    
                with ExitStack() as stack:
                        # get dataloader for corevectors from the original dataset 
                        stack.enter_context(peepholes)
                        peepholes.load_only(
                                        loaders = ['val', 'test'],
                                        verbose = True
                                        )
                        n_max = int(np.ceil(len(peepholes._phs['val'])))
                        print(f'Number of samples for original dataset: {n_max}')
                        
                        f_ori = torch.stack([peepholes._phs['val'][layer]['score_max'] for layer in peep_layers],dim=1)[:n_max].detach().cpu().numpy()
                        f_atk = {}
                        
                        for i, atk_name in enumerate(atk_train):
                                if verbose: print(f'\n---------\nLoading dataset for attack: {atk_name}')
                                stack.enter_context(peepholes_atks_train[atk_name]) # enter context manager
                                peepholes_atks_train[atk_name].load_only(
                                                                        loaders = ['val'],
                                                                        verbose = verbose 
                                                                        )
                                n_min = int(np.ceil(i*0.3333*len(peepholes_atks_train[atk_name]._phs['val'])))
                                n_max = int(np.ceil((i+1)*0.3333*len(peepholes_atks_train[atk_name]._phs['val'])))
                                print(f'Number of samples for attack {atk_name}: {n_max-n_min}, from {n_min} to {n_max}')
                                
                                f_atk[atk_name] = torch.stack([peepholes_atks_train[atk_name]._phs['val'][layer]['score_max'][n_min:n_max] for layer in peep_layers],dim=1)
                        f_atk = torch.concat([f_atk[atk_name] for atk_name in atk_train], dim=0).detach().cpu().numpy()

                        label_ori = np.zeros(len(f_ori))
                        label_atk = np.ones(len(f_atk))
                        train_data = np.concatenate((f_ori, f_atk), axis=0)
                        train_label = np.concatenate((label_ori, label_atk), axis=0)
                        print(train_data.shape, train_label.shape)

                        stack.enter_context(peepholes_atks_test) # enter context manager
                        peepholes_atks_test.load_only(
                                loaders = ['test'],
                                verbose = verbose 
                                )
                        
                        f_ori = torch.stack([peepholes._phs['test'][layer]['score_max'] for layer in peep_layers],dim=1).detach().cpu().numpy()
                        f_atk = torch.stack([peepholes_atks_test._phs['test'][layer]['score_max'] for layer in peep_layers],dim=1).detach().cpu().numpy()
                        
                        label_ori = np.zeros(len(f_ori))
                        label_atk = np.ones(len(f_atk))

                        test_data = np.concatenate((f_ori, f_atk), axis=0)
                        test_label = np.concatenate((label_ori, label_atk), axis=0)

                        lr = LogisticRegressionCV(n_jobs=-1, max_iter=10000).fit(train_data, train_label)
                        y_pred = lr.predict_proba(train_data)[:, 1]
                        
                        y_pred = lr.predict_proba(test_data)[:, 1]
                        fpr, tpr, thresholds = roc_curve(test_label, y_pred)
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
                        plt.savefig(f'../data/{name_model}/img/AUC/RegressionUnknown_attack={atk_test}_n_classes={n_classes}_n_cluster={n_cluster}_cv_dim=100.png')
    '''

    '''
    ### One training attack and three testing attacks
    for config in attacks_config:
                
                atk_train = attacks_config[config]['train']
                atk_test = attacks_config[config]['test']
                print(f'training: {atk_train} test: {atk_test}')

                peepholes = Peepholes(
                        path = phs_path,
                        name = f'{phs_name}.nc_{n_cluster}',
                        driller = drillers,
                        target_modules = peep_layers,
                        device = device
                        )
                peepholes_atks_test = {} 
                
                phs_path_ = phs_path/f'{atk_train}'

                peepholes_atks_train = Peepholes(
                                                path = phs_path_,
                                                name = f'{phs_name}.nc_{n_cluster}',
                                                driller = drillers,
                                                target_modules = peep_layers,
                                                device = device
                                                )
                for atk_name in atk_test:
                        phs_path_ = phs_path/f'{atk_name}'
                        
                        peepholes_atks_test[atk_name] = Peepholes(
                                                                path = phs_path_,
                                                                name = f'{phs_name}.nc_{n_cluster}',
                                                                driller = drillers,
                                                                target_modules = peep_layers,
                                                                device = device
                                                                )    
                with ExitStack() as stack:
                        # get dataloader for corevectors from the original dataset 
                        stack.enter_context(peepholes)
                        peepholes.load_only(
                                        loaders = ['val', 'test'],
                                        verbose = True
                                        )
                        n_max = int(np.ceil(len(peepholes._phs['val'])))
                        print(f'Number of samples for original dataset: {n_max}')
                        
                        stack.enter_context(peepholes_atks_train) # enter context manager
                        peepholes_atks_train.load_only(
                                loaders = ['val'],
                                verbose = verbose 
                                )
                        
                        f_ori = torch.stack([peepholes._phs['val'][layer]['score_max'] for layer in peep_layers],dim=1).detach().cpu().numpy()
                        f_atk = torch.stack([peepholes_atks_train._phs['val'][layer]['score_max'] for layer in peep_layers],dim=1).detach().cpu().numpy()
                        
                        label_ori = np.zeros(len(f_ori))
                        label_atk = np.ones(len(f_atk))

                        train_data = np.concatenate((f_ori, f_atk), axis=0)
                        train_label = np.concatenate((label_ori, label_atk), axis=0)

                        f_atk = {}

                        for i, atk_name in enumerate(atk_test):

                                if verbose: print(f'\n---------\nLoading dataset for attack: {atk_name}')
                                stack.enter_context(peepholes_atks_test[atk_name]) # enter context manager
                                peepholes_atks_test[atk_name].load_only(
                                                                        loaders = ['test'],
                                                                        verbose = verbose 
                                                                        )
                                n_min = int(np.ceil(i*0.3333*len(peepholes_atks_test[atk_name]._phs['test'])))
                                n_max = int(np.ceil((i+1)*0.3333*len(peepholes_atks_test[atk_name]._phs['test'])))
                                print(f'Number of samples for attack {atk_name}: {n_max-n_min}, from {n_min} to {n_max}')
                                
                                f_atk[atk_name] = torch.stack([peepholes_atks_test[atk_name]._phs['test'][layer]['score_max'][n_min:n_max] for layer in peep_layers],dim=1)
                        
                        f_ori = torch.stack([peepholes._phs['test'][layer]['score_max'] for layer in peep_layers],dim=1).detach().cpu().numpy()
                        f_atk = torch.concat([f_atk[atk_name] for atk_name in atk_test], dim=0).detach().cpu().numpy()

                        label_ori = np.zeros(len(f_ori))
                        label_atk = np.ones(len(f_atk))
                        test_data = np.concatenate((f_ori, f_atk), axis=0)
                        test_label = np.concatenate((label_ori, label_atk), axis=0)

                        lr = LogisticRegressionCV(n_jobs=-1, max_iter=10000).fit(train_data, train_label)
                        y_pred = lr.predict_proba(train_data)[:, 1]
                        
                        y_pred = lr.predict_proba(test_data)[:, 1]
                        fpr, tpr, thresholds = roc_curve(test_label, y_pred)
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
                        plt.savefig(f'../data/{name_model}/img/AUC/RegressionUnknown_trained_on={atk_train}_n_classes={n_classes}_n_cluster={n_cluster}_cv_dim=100.png')
    '''