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
from contextlib import ExitStack

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

from peepholelib.coreVectors.dimReduction.avgPooling import cls_token_ViT

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
    name_model = 'ViT'
    #name_model = 'vgg16'
    pretrained = True
    dataset = 'CIFAR100' 
    seed = 29
    bs = 64 

    model_dir = '/srv/newpenny/XAI/models'
    #model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
    model_name = 'SV_model=vit_b_16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau_withInfo.pth'

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
    
    #nn = vgg16()
    
    nn = vit_b_16()

    n_classes = len(ds.get_classes())

    model = ModelWrap(
            model = nn,
            device = device
            )

    model.update_output(
            output_layer = 'heads.head', #'classifier.6
            to_n_classes = n_classes,
            overwrite = True 
            )

    model.load_checkpoint(
            name = model_name,
            path = model_dir,
            verbose = verbose
            )
    
    target_layers = [ f'encoder.layers.encoder_layer_{i}.mlp.3' for i in range(12)]
    
#     target_layers = [
#            'features.4',
#            'features.9',
#            'features.16',
#            'features.23',
#            'features.30'
#             ]
    
#     target_layers = [
#            'features.3',
#            'features.8',
#            'features.15',
#            'features.22',
#            'features.29'
#             ]

    #target_layers = [
    #       'features.2',
    #       'features.7',
    #       'features.14',
    #       'features.21',
    #       'features.28'
    #        ]
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
    
    
    attacks_config = {}
    for i, test in enumerate(attack_list):
        train = [a for a in attack_list if a != test]
        attacks_config[f'c{i}'] = {'train': train, 'test': test}
    
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
    n_samples = 3333

    peep_layers = target_layers

    drillers = {}

    feature_sizes = { f'encoder.layers.encoder_layer_{i}.mlp.3': 768 for i in range(12)}
    
#     feature_sizes = {

#                 'features.2': 64, 

#                 'features.7': 128, 

#                 'features.14': 256, 

#                 'features.21': 512,

#                 'features.28': 512,

#                 }

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
                                        std_transform = [0.300, 0.287, 0.294],
                                        device = device,
                                        parser_act = cls_token_ViT
                                        )
                
    auc_results = {}

    for config in attacks_config:
        atk_train = attacks_config[config]['train']
        atk_test = attacks_config[config]['test']
        print(f'training: {atk_train} test: {atk_test}')

        corevectors = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )
        peepholes = Peepholes(
                path = phs_path,
                name = phs_name,
                driller = drillers,
                target_modules = peep_layers,
                device = device
                )
        
        corevectors_atks_train = {}
        peepholes_atks_train = {} 

        for atk_name in atk_train:
                cvs_path_ = cvs_path/f'{atk_name}'
                phs_path_ = phs_path/f'{atk_name}'

                corevectors_atks_train[atk_name] = CoreVectors(
                                                path = cvs_path_,
                                                name = cvs_name,
                                                )

                peepholes_atks_train[atk_name] = Peepholes(
                                                path = phs_path_,
                                                name = phs_name,
                                                driller = drillers,
                                                target_modules = peep_layers,
                                                device = device
                                                )
        
        cvs_path_ = cvs_path/f'{atk_test}'
        phs_path_ = phs_path/f'{atk_test}'

        corevectors_atks_test = CoreVectors(
                                        path = cvs_path_,
                                        name = cvs_name,
                                        )
        
        peepholes_atks_test = Peepholes(
                        path = phs_path_,
                        name = phs_name,
                        driller = drillers,
                        target_modules = peep_layers,
                        device = device
                        )
                
        with ExitStack() as stack:
                # get dataloader for corevectors from the original dataset 
        
                stack.enter_context(corevectors)
                corevectors.load_only(
                                loaders = ['val', 'test'],
                                verbose = True
                                )
                
                stack.enter_context(peepholes)
                peepholes.load_only(
                                loaders = ['val', 'test'],
                                verbose = True
                                )
                
                f_ori = torch.stack([peepholes._phs['val'][layer]['score_max'] for layer in peep_layers],dim=1).detach().cpu().numpy()
                f_atk = {}
                
                for i, atk_name in enumerate(atk_train):
                        if verbose: print(f'\n---------\nLoading dataset for attack: {atk_name}')

                        stack.enter_context(corevectors_atks_train[atk_name]) # enter context manager
                        corevectors_atks_train[atk_name].load_only(
                                                                loaders = ['val'],
                                                                verbose = verbose 
                                                                )
                        
                        stack.enter_context(peepholes_atks_train[atk_name]) # enter context manager
                        peepholes_atks_train[atk_name].load_only(
                                                                loaders = ['val'],
                                                                verbose = verbose 
                                                                )
                        
                        f_atk[atk_name] = torch.stack([peepholes_atks_train[atk_name]._phs['val'][layer]['score_max'] for layer in peep_layers],dim=1)
                        idx = torch.argwhere((corevectors._dss['val']['result']==1) & (corevectors_atks_train[atk_name]._dss['val']['attack_success']==1)).squeeze()
                        
                        assert len(idx) >= n_samples, f'Not enough samples for attack {atk_name} in training set. Found {len(idx)} samples, expected at least {n_samples}.'

                        rand = torch.randperm(len(idx))[:n_samples]
                        f_atk[atk_name] = f_atk[atk_name][idx[rand]]
                               
                f_atk = torch.concat([f_atk[atk_name] for atk_name in atk_train], dim=0).detach().cpu().numpy()

                label_ori = np.zeros(len(f_ori))
                label_atk = np.ones(len(f_atk))
                train_data = np.concatenate((f_ori, f_atk), axis=0)
                train_label = np.concatenate((label_ori, label_atk), axis=0)

                stack.enter_context(corevectors_atks_test) # enter context manager
                corevectors_atks_test.load_only(
                                                loaders = ['test'],
                                                verbose = verbose 
                                                )

                stack.enter_context(peepholes_atks_test) # enter context manager
                peepholes_atks_test.load_only(
                        loaders = ['test'],
                        verbose = verbose 
                        )
                
                idx = torch.argwhere((corevectors._dss['test']['result']==1) & (corevectors_atks_test._dss['test']['attack_success']==1)).squeeze()
                f_ori = torch.stack([peepholes._phs['test'][layer]['score_max'] for layer in peep_layers],dim=1)[idx].detach().cpu().numpy()
                f_atk = torch.stack([peepholes_atks_test._phs['test'][layer]['score_max'] for layer in peep_layers],dim=1)[idx].detach().cpu().numpy()

                label_ori = np.zeros(len(f_ori))
                label_atk = np.ones(len(f_atk))

                test_data = np.concatenate((f_ori, f_atk), axis=0)
                test_label = np.concatenate((label_ori, label_atk), axis=0)

                lr = LogisticRegressionCV(n_jobs=-1, max_iter=10000).fit(train_data, train_label) #
                #y_pred = lr.predict_proba(train_data)[:, 1]
                
                y_pred = lr.predict_proba(test_data)[:, 1]

                y_ori = lr.predict_proba(f_ori)[:, 1]
                y_atk = lr.predict_proba(f_atk)[:, 1]

                fpr, tpr, thresholds = roc_curve(test_label, y_pred)
                roc_auc = auc(fpr, tpr)
                auc_results[atk_test] = roc_auc

                print("AUC:", roc_auc)

                plt.figure()
                fig, axs = plt.subplots(2,1, figsize=(7,10))
                axs[0].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                axs[0].plot([0, 1], [0, 1], 'k--', label='Chance')
                axs[0].set_xlabel('False Positive Rate')
                axs[0].set_ylabel('True Positive Rate')
                axs[0].set_title('Receiver Operating Characteristic')
                axs[0].legend()
                axs[1].hist(y_ori, bins=30, label='ori')
                axs[1].hist(y_atk, bins=30, label=f'{atk_test}', alpha=0.7)
                axs[1].legend()
                fig.savefig(f'../data/{name_model}/img/AUC/RegressionUnknown_attack={atk_test}_DMD.png')

                # # Plot ROC curve
                # plt.figure()
                # plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                # plt.plot([0, 1], [0, 1], 'k--', label='Chance')
                # plt.xlabel('False Positive Rate')
                # plt.ylabel('True Positive Rate')
                # plt.title('Receiver Operating Characteristic')
                # plt.legend(loc="lower right")
                # plt.savefig(f'../data/{name_model}/img/AUC/RegressionUnknown_attack={atk_test}_DMD_Conv.png')
                
    '''
    ### One training attack and three testing attacks
    for config in attacks_config:
                
                atk_train = attacks_config[config]['train']
                atk_test = attacks_config[config]['test']
                print(f'training: {atk_train} test: {atk_test}')

                peepholes = Peepholes(
                                path = phs_path,
                                name = phs_name,
                                driller = drillers,
                                target_modules = peep_layers,
                                device = device
                                )
                peepholes_atks_test = {} 
                
                phs_path_ = phs_path/f'{atk_train}'

                peepholes_atks_train = Peepholes(
                                                path = phs_path_,
                                                name = phs_name,
                                                driller = drillers,
                                                target_modules = peep_layers,
                                                device = device
                                                )
                for atk_name in atk_test:
                        phs_path_ = phs_path/f'{atk_name}'
                        
                        peepholes_atks_test[atk_name] = Peepholes(
                                                                path = phs_path_,
                                                                name = phs_name,
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
                        plt.savefig(f'../data/{name_model}/img/AUC/RegressionUnknown_trained_on={atk_train}_DMD.png')
    

    plt.figure(figsize=(10, 6))
    for config, auc_values in auc_results.items():
            print(f'Config: {list_portion}, AUC values: {auc_values}')
            plt.plot(list_portion, auc_values, marker='o', label=attacks_config[config]['test'])

    plt.xlabel('Percentage of data used')
    plt.ylabel('AUC')
    plt.title('AUC across different data percentages')
    plt.legend()
    plt.grid(True)
    plt.xscale('log', base=2)
    plt.xticks(list_portion, labels=[f'{p*100:.1f}%' for p in list_portion])
    plt.savefig(f'prova_.png')

    '''
                                
