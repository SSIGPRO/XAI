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
import pandas as pd

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

from peepholelib.utils.samplers import random_subsampling 
from peepholelib.utils.analyze import conceptogram_protoclass_score_attacks, conceptogram_entropy_score_attacks

# torch stuff
import torch
from torchvision.models import vgg16, VGG16_Weights, vit_b_16
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
    name_model = 'vgg16'
    # name_model = 'ViT'
    seed = 29
    bs = 512 
    cv_dim = 100
    
    cvs_path = Path('/srv/newpenny/XAI/generated_data/temp_results/toeplitz_conv/corevectors') #Path.cwd()/f'../data/{name_model}/corevectors' 
    # cvs_path = Path.cwd()/f'../data/{name_model}/corevectors' #Path('/srv/newpenny/XAI/Peephole-Analysis/channel_wise/corevectors') 
    cvs_name = 'corevectors'
   
    drill_path = Path('/srv/newpenny/XAI/generated_data/temp_results/toeplitz_conv/drillers') #Path.cwd()/f'../data/{name_model}/drillers' #Path('/srv/newpenny/XAI/Peephole-Analysis/drillers_on_superclass') #
    # drill_path = Path.cwd()/f'../data/{name_model}/drillers' #Path('/srv/newpenny/XAI/Peephole-Analysis/channel_wise/drillers_on_class') 
    drill_name = 'classifier'

    phs_path = Path('/srv/newpenny/XAI/generated_data/temp_results/toeplitz_conv/peepholes') #Path.cwd()/f'../data/{name_model}/peepholes' #Path('/srv/newpenny/XAI/Peephole-Analysis/peepholes_on_superclass') #
    # phs_path = Path.cwd()/f'../data/{name_model}/peepholes_{cv_dim}' #Path('/srv/newpenny/XAI/Peephole-Analysis/channel_wise/peepholes_on_class') 
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
    # CoreVectors 
    #--------------------------------
    attack_list = [
                'BIM',
                'CW',
                'DeepFool',
                'PGD'
                ]

    #--------------------------------
    # Peepholes
    #--------------------------------
    n_classes = 100
    n_cluster = 150

    parser_cv = trim_corevectors

    parser_cv = trim_corevectors
    # peep_layers = [ f'encoder.layers.encoder_layer_{i}.mlp.{j}' for i in range(10) for j in [0,3]]
    # peep_layers.append('heads.head')

    peep_layers = [
            #'features.0',
            #'features.2',
            #'features.5',
            #'features.7',
            'features.10',
            'features.12',
            'features.14',
            'features.17',
            'features.19',
            'features.21',
            'features.24',
            'features.26',
            'features.28',
            'classifier.0',
            'classifier.3',
            'classifier.6',
            ]

    auc_results = {layer: {} for layer in peep_layers}
    drillers = {}

    # feature_sizes = { f'encoder.layers.encoder_layer_{i}.mlp.{j}': cv_dim for i in range(12) for j in [0,3]}
    # feature_sizes['heads.head'] = cv_dim

    features_cv_dim = 100
    classifier_cv_dim = 150

    feature_sizes = {}
    for _layer in peep_layers:
        if 'features' in _layer:
                feature_sizes[_layer] =  features_cv_dim
        elif 'classifier' in _layer:
                feature_sizes[_layer] = classifier_cv_dim
    feature_sizes['classifier.6'] = n_classes
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
    
    for atk in attack_list:
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
                name = f'{phs_name}', #.nc_{n_cluster}
                driller = drillers,
                target_modules = peep_layers,
                device = device
                )
        peepholes_atk = Peepholes(
                path = phs_path_,
                name = f'{phs_name}', #.nc_{n_cluster}
                driller = drillers,
                target_modules = peep_layers,
                device = device
                )
        
        with corevecs as cv, corevecs_atk as cv_atk, peepholes as ph, peepholes_atk as ph_atk:
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
                
                for peep_layer in peep_layers:
                        
                        # ret = conceptogram_entropy_score_attacks(peepholes_ori=ph,
                        #                                  peepholes_atk=ph_atk,
                        #                                  corevectors_ori=cv,
                        #                                  corevectors_atk=cv_atk,
                        #                                  target_modules = [peep_layer],
                        #                                  loaders=['test'],
                        #                                  atk_name=atk,
                        #                                  plot=True,
                        #                                  path = f'../data/{name_model}/img/AUC/H_score={atk}_{peep_layer}_n_cluster={n_cluster}_cv_dim={feature_sizes[peep_layer]}.png'
                        #                                  )

                        ret = conceptogram_protoclass_score_attacks(peepholes_ori=ph,
                                                               peepholes_atk=ph_atk,
                                                               corevectors_ori=cv,
                                                               corevectors_atk=cv_atk,
                                                               target_modules = [peep_layer],
                                                               loaders=['train', 'test'],
                                                               atk_name=atk,
                                                               plot=True,
                                                               path = f'../data/{name_model}/img/AUC/Proto_score={atk}_{peep_layer}_n_cluster={n_cluster}_cv_dim={feature_sizes[peep_layer]}.png'
                                                               )
                        auc_results[peep_layer][atk] = ret['auc']['test']

    df_auc = pd.DataFrame.from_dict(auc_results, orient='index')
    df_auc = df_auc.round(2)

    # Save to CSV
    #df_auc.to_csv('../data/ViT/auc_results_H_only_attack_success.csv')
    df_auc.to_csv(f'../data/{name_model}/auc_results_protoclass.csv')
    df_auc.to_latex(f'../data/{name_model}/auc_results_protoclass.tex')
    
    quit()     
#     peep_layers = {
#          'features.24': 25,
#          'features.26': 25,
#          'features.28': 25,
        #  'classifier.0': 100, 
        #  'classifier.3': 100 
#          }
#     drillers = {}
#     for peep_layer, cv_dim in peep_layers.items():
        
#         parser_kwargs = {'module': peep_layer, 'cv_dim':cv_dim}

#         drillers[peep_layer] = tGMM(
#                                 path = drill_path,
#                                 name = drill_name+'.'+peep_layer,
#                                 nl_classifier = n_cluster,
#                                 nl_model = n_classes,
#                                 n_features = cv_dim,
#                                 parser = parser_cv,
#                                 parser_kwargs = parser_kwargs,
#                                 device = device,
#                                 batch_size = 512,
#                                 )
#     for atk in attack_list:
#         print('--------------------------------')
#         print('atk type: ', atk)
#         print('--------------------------------')
#         phs_path_ = phs_path/f'{atk}'
#         peepholes = Peepholes(
#                 path = phs_path,
#                 name = f'{phs_name}.ps_{parser_kwargs['cv_dim']}.nc_{n_cluster}',
#                 driller = drillers,
#                 target_modules = peep_layers,
#                 device = device
#                 )
#         peepholes_atk = Peepholes(
#                 path = phs_path_,
#                 name = f'{phs_name}.ps_{parser_kwargs['cv_dim']}.nc_{n_cluster}',
#                 driller = drillers,
#                 target_modules = peep_layers,
#                 device = device
#                 )
#         with peepholes as ph, peepholes_atk as ph_atk:
#                 ph.load_only(
#                         loaders = ['val'],
#                         verbose = True
#                         )
#                 ph_atk.load_only(
#                         loaders = ['val'],
#                         verbose = True
#                         )
#                 for peep_layer in peep_layers.keys():
#                         s_ori = ph._phs['val'][peep_layer]['score_max']
#                         s_atk = ph_atk._phs['val'][peep_layer]['score_max']
#                         labels_pos = torch.ones(len(s_ori))
#                         labels_neg = torch.zeros(len(s_atk))
#                         all_scores = torch.cat([s_ori, s_atk], dim=0)
#                         all_labels = torch.cat([labels_pos, labels_neg], dim=0)

#                         # Convert tensors to numpy arrays (using .detach().cpu().numpy() if needed)
#                         all_scores_np = all_scores.detach().cpu().numpy()
                        
#                         all_labels_np = all_labels.detach().cpu().numpy()

#                         # Compute ROC curve and AUC
#                         fpr, tpr, thresholds = roc_curve(all_labels_np, all_scores_np)
#                         roc_auc = auc(fpr, tpr)

#                         print("AUC:", roc_auc)

#                         # Plot ROC curve
#                         plt.figure()
#                         plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
#                         plt.plot([0, 1], [0, 1], 'k--', label='Chance')
#                         plt.xlabel('False Positive Rate')
#                         plt.ylabel('True Positive Rate')
#                         plt.title('Receiver Operating Characteristic')
#                         plt.legend(loc="lower right")
#                         plt.savefig(f'../data/img/AUC/MAX_attack={atk}_n_classes={n_classes}_{peep_layer}_n_cluster={n_cluster}_cv_dim=25.png')

    #--------------------------------
    # Conceptograms
    #--------------------------------
    n_classes = 20
    n_cluster = 40
    SM = torch.nn.Softmax(dim=1)
    idx = 12
    
    ### Original Image###

    parser_cv = trim_corevectors
    peep_layers_clas = {
         'features.24': 100,
         'features.26': 100,
         'features.28': 100,
         'classifier.0': 100, 
         'classifier.3': 100 
         }
    drillers_clas = {}
    for peep_layer, cv_dim in peep_layers_clas.items():
        
        parser_kwargs = {'module': peep_layer, 'cv_dim':cv_dim}

        drillers_clas[peep_layer] = tGMM(
                                path = drill_path,
                                name = drill_name+'.'+peep_layer,
                                nl_classifier = n_cluster,
                                nl_model = n_classes,
                                n_features = cv_dim,
                                parser = parser_cv,
                                parser_kwargs = parser_kwargs,
                                device = device,
                                batch_size = 512,
                                )
#     peep_layers_feat = {
#         #  'features.24': 25,
#         #  'features.26': 27,
#         #  'features.28': 26,
#          'classifier.0': 80, 
#          'classifier.3': 100 
#          }
#     drillers_feat = {}
#     for peep_layer, cv_dim in peep_layers_feat.items():
        
#         parser_kwargs = {'module': peep_layer, 'cv_dim':cv_dim}

#         drillers_feat[peep_layer] = tGMM(
#                                 path = drill_path,
#                                 name = drill_name+'.'+peep_layer,
#                                 nl_classifier = n_cluster,
#                                 nl_model = n_classes,
#                                 n_features = cv_dim,
#                                 parser = parser_cv,
#                                 parser_kwargs = parser_kwargs,
#                                 device = device,
#                                 batch_size = 512,
#                                 )
    peepholes_clas = Peepholes(path = phs_path,
                               name = f'{phs_name}.ps_100.nc_{n_cluster}',
                               driller = drillers_clas,
                               target_modules = peep_layers_clas,
                               device = device
                               )
#     peepholes_feat = Peepholes(path = phs_path,
#                                name = f'{phs_name}.ps_25.nc_{n_cluster}',
#                                driller = drillers_feat,
#                                target_modules = peep_layers_feat,
#                                device = device
#                                )
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )
    superclass_names = [
                "Aquatic Mammals", "Fish", "Flowers", "Food Containers", "Fruit and Vegetables",
                "Household Electrical Devices", "Household Furniture", "Insects", "Large Carnivores", "Large Man-made Outdoor Things",
                "Large Natural Outdoor Scenes", "Large Omnivores and Herbivores", "Medium-sized Mammals", "Non-insect Invertebrates",
                "People", "Reptiles", "Small Mammals", "Trees", "Vehicles 1", "Vehicles 2"
                ]
#     with peepholes_clas as phc, peepholes_feat as phf, corevecs as cv:
    with peepholes_clas as phc, corevecs as cv:
          cv.load_only(loaders = ['val'],
                        verbose = True
                        )
          phc.load_only(loaders = ['val'],
                        verbose = True
                        )
        #   phf.load_only(loaders = ['val'],
        #                 verbose = True
        #   )
          
          img = cv._actds['val']['image'][idx]
          
          
          concepto = torch.stack((phc._phs['val']['features.24']['peepholes'][idx],
                                      phc._phs['val']['features.26']['peepholes'][idx],
                                      phc._phs['val']['features.28']['peepholes'][idx],
                                      phc._phs['val']['classifier.0']['peepholes'][idx],
                                      phc._phs['val']['classifier.3']['peepholes'][idx])).T
        #   concepto = torch.stack((phf._phs['val']['features.24']['peepholes'][idx],
        #                               phf._phs['val']['features.26']['peepholes'][idx],
        #                               phf._phs['val']['features.28']['peepholes'][idx],
        #                               phc._phs['val']['classifier.0']['peepholes'][idx],
        #                               phc._phs['val']['classifier.3']['peepholes'][idx])).T
          true_out = cv._actds["val"]['label'][idx]     # number
          true_class = ds._classes[int(true_out.cpu().numpy())]              # string

          label_out = cv._actds["val"]['pred'][idx]
          label_class = ds._classes[int(label_out.cpu().numpy())]

          #top 5 indices
          row_sums = concepto.sum(axis=1)
          print("Row sums:", row_sums.shape)
          top_5_indices = np.argsort(row_sums)[-5:]#[::-1]  # sort and take last 5 in descending order
          top_5_rows = concepto[top_5_indices]
          print("Indices of top 5 rows:", top_5_indices)
          max_value = torch.argmax(row_sums)


          # plot
          fig, ax = plt.subplots(1, 3, figsize=(8, 6))
          fig.suptitle(f'True label: {true_out} - Pred label: {label_out}')

          ax[0].imshow(img.permute(1, 2, 0).cpu().numpy())
          ax[0].axis("off")

          ax[1].imshow(concepto, aspect='auto', cmap='YlGnBu')
          ax[1].set_yticks([])
          ax[1].set_yticks([max_value], [f'{max_value}, {superclass_names[max_value]}'])
          ax[1].yaxis.tick_right()
          ax[1].set_xlabel('VGG16 Layers')

          ax[2].imshow(SM(cv._actds["val"]["output"])[idx].unsqueeze(dim=1).cpu().numpy(), cmap='YlGnBu')
          ax[2].set_xticks([])
          ax[2].set_yticks([label_out], [f'{label_out}, {ds._classes[int(label_out.detach().cpu().numpy())]}'])
          ax[2].yaxis.set_label_position("right")
          ax[2].yaxis.tick_right()
          ax[2].set_xlabel('Net Out')

          plt.tight_layout()
          plt.show()
          fig.savefig(f'Conceptogram_{idx}.png')

    ### ATTACKS ###
    for atk in attack_list:
        print('--------------------------------')
        print('atk type: ', atk)
        print('--------------------------------')
        phs_path_ = phs_path/f'{atk}'
        parser_cv = trim_corevectors
        peep_layers_clas = {
                 'features.24': 100,
                 'features.26': 100,
                 'features.28': 100,
                'classifier.0': 100, 
                'classifier.3': 100 
                }
        drillers_clas = {}
        for peep_layer, cv_dim in peep_layers_clas.items():
                
                parser_kwargs = {'module': peep_layer, 'cv_dim':cv_dim}

                drillers_clas[peep_layer] = tGMM(
                                        path = drill_path,
                                        name = drill_name+'.'+peep_layer,
                                        nl_classifier = n_cluster,
                                        nl_model = n_classes,
                                        n_features = cv_dim,
                                        parser = parser_cv,
                                        parser_kwargs = parser_kwargs,
                                        device = device,
                                        batch_size = 512,
                                        )
        # peep_layers_feat = {
        #         #  'features.24': 25,
        #         #  'features.26': 25,
        #         #  'features.28': 25,
        #         'classifier.0': 100, 
        #         'classifier.3': 100 
        #         }
        # drillers_feat = {}
        # for peep_layer, cv_dim in peep_layers_feat.items():
                
        #         parser_kwargs = {'module': peep_layer, 'cv_dim':cv_dim}

        #         drillers_feat[peep_layer] = tGMM(
        #                                 path = drill_path,
        #                                 name = drill_name+'.'+peep_layer,
        #                                 nl_classifier = n_cluster,
        #                                 nl_model = n_classes,
        #                                 n_features = cv_dim,
        #                                 parser = parser_cv,
        #                                 parser_kwargs = parser_kwargs,
        #                                 device = device,
        #                                 batch_size = 512,
        #                                 )
        peepholes_clas = Peepholes(path = phs_path_,
                                name = f'{phs_name}.ps_100.nc_{n_cluster}',
                                driller = drillers_clas,
                                target_modules = peep_layers_clas,
                                device = device
                                )
        # peepholes_feat = Peepholes(path = phs_path_,
        #                         name = f'{phs_name}.ps_25.nc_{n_cluster}',
        #                         driller = drillers_feat,
        #                         target_modules = peep_layers_feat,
        #                         device = device
        #                         )
        corevecs = CoreVectors(
                path = cvs_path/f'{atk}',
                name = cvs_name,
                )
        superclass_names = [
                        "Aquatic Mammals", "Fish", "Flowers", "Food Containers", "Fruit and Vegetables",
                        "Household Electrical Devices", "Household Furniture", "Insects", "Large Carnivores", "Large Man-made Outdoor Things",
                        "Large Natural Outdoor Scenes", "Large Omnivores and Herbivores", "Medium-sized Mammals", "Non-insect Invertebrates",
                        "People", "Reptiles", "Small Mammals", "Trees", "Vehicles 1", "Vehicles 2"
                        ]
        # with peepholes_clas as phc, peepholes_feat as phf, corevecs as cv:
        with peepholes_clas as phc, corevecs as cv:
                cv.load_only(loaders = ['val'],
                                verbose = True
                                )
                phc.load_only(loaders = ['val'],
                                verbose = True
                                )
                # phf.load_only(loaders = ['val'],
                #                 verbose = True
                # )
                
                img = cv._actds['val']['image'][idx]

                concepto = torch.stack((phc._phs['val']['features.24']['peepholes'][idx],
                                        phc._phs['val']['features.26']['peepholes'][idx],
                                        phc._phs['val']['features.28']['peepholes'][idx],
                                        phc._phs['val']['classifier.0']['peepholes'][idx],
                                        phc._phs['val']['classifier.3']['peepholes'][idx])).T
                # concepto = torch.stack((phf._phs['val']['features.24']['peepholes'][idx],
                #                         phf._phs['val']['features.26']['peepholes'][idx],
                #                         phf._phs['val']['features.28']['peepholes'][idx],
                #                         phc._phs['val']['classifier.0']['peepholes'][idx],
                #                         phc._phs['val']['classifier.3']['peepholes'][idx])).T
                true_out = cv._actds["val"]['label'][idx]     # number
                true_class = ds._classes[int(true_out.cpu().numpy())]              # string

                label_out = cv._actds["val"]['pred'][idx]
                label_class = ds._classes[int(label_out.cpu().numpy())]

                #top 5 indices
                row_sums = concepto.sum(axis=1)
                print("Row sums:", row_sums.shape)
                top_5_indices = np.argsort(row_sums)[-5:]#[::-1]  # sort and take last 5 in descending order
                top_5_rows = concepto[top_5_indices]
                print("Indices of top 5 rows:", top_5_indices)
                max_value = torch.argmax(row_sums)


                # plot
                fig, ax = plt.subplots(1, 3, figsize=(8, 6))
                fig.suptitle(f'True label: {true_out} - Pred label: {label_out}')

                ax[0].imshow(img.permute(1, 2, 0).cpu().numpy())
                ax[0].axis("off")

                ax[1].imshow(concepto, aspect='auto', cmap='YlGnBu')
                ax[1].set_yticks([])
                ax[1].set_yticks([max_value], [f'{max_value}, {superclass_names[max_value]}'])
                ax[1].yaxis.tick_right()
                ax[1].set_xlabel('VGG16 Layers')

                ax[2].imshow(SM(cv._actds["val"]["output"])[idx].unsqueeze(dim=1).cpu().numpy(), cmap='YlGnBu')
                ax[2].set_xticks([])
                ax[2].set_yticks([label_out], [f'{label_out}, {ds._classes[int(label_out.detach().cpu().numpy())]}'])
                ax[2].yaxis.set_label_position("right")
                ax[2].yaxis.tick_right()
                ax[2].set_xlabel('Net Out')

                plt.tight_layout()
                plt.show()
                fig.savefig(f'{atk}_Conceptogram_{idx}.png')