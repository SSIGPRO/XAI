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
from peepholelib.peepholes.parsers import trim_corevectors

from peepholelib.peepholes.classifiers.tkmeans import KMeans as tKMeans 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes
#from peepholelib.utils.viz_conceptogram import get_conceptogram_class

from peepholelib.utils.samplers import random_subsampling
from peepholelib.utils.analyze import conceptogram_protoclass_score_attacks, conceptogram_entropy_score_attacks
from peepholelib.utils.conceptograms import plot_conceptogram

# torch stuff
import torch
from torchvision.models import vgg16, VGG16_Weights,vit_b_16
from cuda_selector import auto_cuda

if __name__ == "__main__":
#     use_cuda = torch.cuda.is_available()
#     device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
#     print(f"Using {device} device")
    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 6
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    ds_path = '/srv/newpenny/dataset/CIFAR100'

    # model parameters
    pretrained = True
    dataset = 'CIFAR100' 
#     name_model = 'vgg16'
    name_model = 'ViT'
    seed = 29
    bs = 512 
    cv_dim = 100
    
#     cvs_path = Path('/srv/newpenny/XAI/generated_data/temp_results/toeplitz_conv/corevectors') #Path.cwd()/f'../data/{name_model}/corevectors' 
    cvs_path = Path.cwd()/f'../data/{name_model}/corevectors' #Path('/srv/newpenny/XAI/Peephole-Analysis/channel_wise/corevectors') 
    cvs_name = 'corevectors'
   
#     drill_path = Path('/srv/newpenny/XAI/generated_data/temp_results/toeplitz_conv/drillers') #Path.cwd()/f'../data/{name_model}/drillers' #Path('/srv/newpenny/XAI/Peephole-Analysis/drillers_on_superclass') #
    drill_path = Path.cwd()/f'../data/{name_model}/drillers' #Path('/srv/newpenny/XAI/Peephole-Analysis/channel_wise/drillers_on_class') 
    drill_name = 'classifier'

#     phs_path = Path('/srv/newpenny/XAI/generated_data/temp_results/toeplitz_conv/peepholes') #Path.cwd()/f'../data/{name_model}/peepholes' #Path('/srv/newpenny/XAI/Peephole-Analysis/peepholes_on_superclass') #
    phs_path = Path.cwd()/f'../data/{name_model}/peepholes_{cv_dim}' #Path('/srv/newpenny/XAI/Peephole-Analysis/channel_wise/peepholes_on_class') 
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

#     peep_layers = [
#             #'features.0',
#             #'features.2',
#             #'features.5',
#             #'features.7',
#             'features.10',
#             'features.12',
#             'features.14',
#             'features.17',
#             'features.19',
#             'features.21',
#             'features.24',
#             'features.26',
#             'features.28',
#             'classifier.0',
#             'classifier.3',
#             'classifier.6',
#             ]
    print('peep_layers = ', peep_layers)

    drillers = {}

    feature_sizes = { f'encoder.layers.encoder_layer_{i}.mlp.{j}': cv_dim for i in range(12) for j in [0,3]}
    feature_sizes['heads.head'] = cv_dim

#     features_cv_dim = 100
#     classifier_cv_dim = 150

#     feature_sizes = {}
#     for _layer in peep_layers:
#         if 'features' in _layer:
#                 feature_sizes[_layer] =  features_cv_dim
#         elif 'classifier' in _layer:
#                 feature_sizes[_layer] = classifier_cv_dim
#     feature_sizes['classifier.6'] = n_classes

    for peep_layer in peep_layers:
        
        parser_kwargs = {'module': peep_layer, 'cv_dim':100}

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
        
    auc_dict = {}
    for atk in attack_list:
        auc_dict[atk] = torch.zeros((len(peep_layers),), device=device)
        print('--------------------------------')
        print('atk type: ', atk)
        print('--------------------------------')
        cvs_path_ = cvs_path/f'{atk}'
        phs_path_ = phs_path/f'{atk}'

        corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )
        corevecs_atk = CoreVectors(
            path = cvs_path_,
            name = cvs_name,
            )

        peepholes = Peepholes(
                path = phs_path,
                name = f'{phs_name}.nc_{n_cluster}', # 
                driller = drillers,
                target_modules = peep_layers,
                device = device
                )
        peepholes_atk = Peepholes(
                path = phs_path_,
                name = f'{phs_name}.nc_{n_cluster}', # 
                driller = drillers,
                target_modules = peep_layers,
                device = device
                )
        with corevecs as cv, corevecs_atk as cv_atk, peepholes as ph, peepholes_atk as ph_atk:
        # with peepholes as ph, peepholes_atk as ph_atk:
                ph.load_only(
                        loaders = ['train', 'test'],
                        verbose = True
                        )
                ph_atk.load_only(
                        loaders = ['test'],
                        verbose = True
                        )
                cv.load_only(
                        loaders = ['train', 'test'],
                        verbose = True
                        )
                cv_atk.load_only(
                        loaders = ['test'],
                        verbose = True
                        )
                
                ret = conceptogram_entropy_score_attacks(peepholes_ori=ph,
                                                         peepholes_atk=ph_atk,
                                                         corevectors_ori=cv,
                                                         corevectors_atk=cv_atk,
                                                         target_modules = peep_layers,
                                                         loaders=['test'],
                                                         atk_name=atk,
                                                         plot=True,
                                                         path = f'H_ConceptoVs{atk}_{name_model}.png' #_with_Head
                                                         )
                
                # ret = conceptogram_protoclass_score_attacks(peepholes_ori=ph,
                #                                                peepholes_atk=ph_atk,
                #                                                corevectors_ori=cv,
                #                                                corevectors_atk=cv_atk,
                #                                                target_modules = peep_layers,
                #                                                loaders=['train', 'test'],
                #                                                atk_name=atk,
                #                                                plot=False,
                #                                                path = f'ConceptoVs{atk}_{name_model}_.png' #_with_Head
                #                                                )
                print(f'AUC {atk}: {ret['auc']}')  

                # thr = 0.9

                # mask = (cv._dss['test']['result'] == 1) & (cv_atk._dss['test']['attack_success'] == 1) & (ret['sa']['test'] > thr)

                # # all the indices satisfying the mask
                # idx = torch.argwhere(mask)

                # # extract the corresponding scores (flatten to 1-D)
                # scores = ret['sa']['test'][mask].view(-1)

                # # get sort order (largest first)
                # order = torch.argsort(scores, descending=True)

                # # apply to idx
                # idx = idx[order][-5:].squeeze()
                # print(idx)
                # print(ret['sa']['test'][idx])

                # #idx = torch.argwhere((cv._dss['test']['result'] == 1) & (cv_atk._dss['test']['attack_success'] == 1) & (ret['sa']['test'] > thr))
                
                # ## i have to extract the indices that correspond to a score larger than tr and at the same time satisfy the masking
                # plot_conceptogram( 
                #                 path = Path.cwd()/f'../data/{name_model}/img/concepto/{atk}',
                #                 name = 'concepto_ori',
                #                 corevectors = cv,
                #                 peepholes = ph,
                #                 portion = 'test',
                #                 samples = idx,
                #                 target_modules = peep_layers,
                #                 classes = ds._classes,
                #                 )
                
                # plot_conceptogram(
                #                 path = Path.cwd()/f'../data/{name_model}/img/concepto/{atk}',
                #                 name = 'concepto_atk',
                #                 corevectors = cv_atk,
                #                 peepholes = ph_atk,
                #                 portion = 'test',
                #                 samples = idx,
                #                 target_modules = peep_layers,
                #                 classes = ds._classes,
                #                 )
                              
       
#                 '''
#                 ## visualization of the conceptograms
#                 idx = 120
#                 portion = 'test'
#                 ticks = ticks = [f'encoder_{i}' for i in range(12)]
#                 k_rows = 3
#                 list_classes = ds._classes
#                 path_ori = Path.cwd()/'../data/ViT/img/conceptogram/conceptogram_ori'
#                 # get_conceptogram_class(cv, ph, idx, target_layers, portion, ticks, k_rows, list_classes, path=path_ori)

#                 path_atk = Path.cwd()/f'../data/ViT/img/conceptogram/conceptogram_{atk}'
#                 # get_conceptogram_class(cv_atk, ph_atk, idx, target_layers, portion, ticks, k_rows, list_classes, path=path_atk)

#                 c_ori = ph.get_conceptograms(target_modules=peep_layers)
#                 c_atk = ph_atk.get_conceptograms(target_modules=peep_layers)
#                 '''
#                 '''
#                 scores_ori, _, _, _, _ = conceptogram_cl_score(peepholes=ph,
#                                                                        corevectors=cv,
#                                                                        loaders=['test'],
#                                                                        layers=peep_layers
#                                                                        )
#                 scores_atk, _, _, _, _ = conceptogram_cl_score(peepholes=ph_atk,
#                                                                        corevectors=cv_atk,
#                                                                        loaders=['test'],
#                                                                        layers=peep_layers 
#                                                                        )
                
#                 labels_pos = np.ones(len(scores_ori))
#                 labels_neg = np.zeros(len(scores_atk))
#                 all_scores = np.concatenate([scores_ori, scores_atk], axis=0)
#                 all_labels = np.concatenate([labels_pos, labels_neg], axis=0)

#                 # Compute ROC curve and AUC
#                 fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
#                 roc_auc = auc(fpr, tpr)

#                 print("AUC:", roc_auc)
                
#                 # Plot ROC curve
#                 plt.figure()
#                 plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
#                 plt.plot([0, 1], [0, 1], 'k--', label='Chance')
#                 plt.xlabel('False Positive Rate')
#                 plt.ylabel('True Positive Rate')
#                 plt.title('Receiver Operating Characteristic')
#                 plt.legend(loc="lower right")
#                 plt.savefig(f'prova_{atk}.png')
#                 # plt.savefig(f'../data/ViT/img/AUC/Conceptogram_attack={atk}_n_classes={n_classes}_n_cluster={n_cluster}_cv_dim={cv_dim}.png')
#                 '''

#                 for i in range(len(peep_layers)):
                
#                         layers = peep_layers[i:]
#                         scores_ori, _, _, _, _ = conceptogram_cl_score(peepholes=ph,
#                                                                        corevectors=cv,
#                                                                        loaders=['test'],
#                                                                        layers=layers
#                                                                        )
#                         scores_atk, _, _, _, _ = conceptogram_cl_score(peepholes=ph_atk,
#                                                                         corevectors=cv_atk,
#                                                                         loaders=['test'],
#                                                                         layers=layers 
#                                                                         )
#                         scores_ori = scores_ori[idx]
#                         scores_atk = scores_atk[idx]

#                         labels_pos = np.ones(len(scores_ori))
#                         labels_neg = np.zeros(len(scores_atk))
#                         all_scores = np.concatenate([scores_ori, scores_atk], axis=0)
#                         all_labels = np.concatenate([labels_pos, labels_neg], axis=0)

#                         # Compute ROC curve and AUC
#                         fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
#                         auc_dict[atk][i] = auc(fpr, tpr)
#                 plt.figure(figsize=(10, 10))
#                 plt.plot(peep_layers, auc_dict[atk].cpu().numpy())
#                 plt.xlabel('Peephole Layers')
#                 plt.xticks(rotation=90, fontsize=8)
#                 plt.ylabel('AUC')
#                 plt.title(f'AUC vs Peephole Layers ({atk})')
#                 plt.grid(True)
#                 plt.show()
#                 plt.savefig(f'AUC_H_conceptogram_{atk}_only_AS.png')
#     plt.figure(figsize=(10, 10))
#     for atk in attack_list:
#         plt.plot(peep_layers, auc_dict[atk].cpu().numpy(), label=f'AUC {atk}')
#     plt.xlabel('Peephole Layers')
#     plt.xticks(rotation=90, fontsize=8)
#     plt.ylabel('AUC')
#     plt.title('AUC vs Peephole Layers (All Attacks)')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#     plt.savefig('AUC_H_conceptogram_all_attacks_only_AS.png')