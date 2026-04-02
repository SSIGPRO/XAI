import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# python stuff
from time import time
from functools import partial
import random
from matplotlib import pyplot as plt
          
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr, spearmanr
# torch stuff
import torch
from cuda_selector import auto_cuda
import torchvision
import torch  

###### Our stuff

# Model
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.models.svd_fns import linear_svd

# datasets
from peepholelib.datasets.cifar100 import Cifar100
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.datasets.functional.parsers import from_dataset
from peepholelib.datasets.functional.samplers import random_subsampling 

# corevecs
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import linear_svd_projection, linear_svd_projection_ViT

# peepholes
from peepholelib.peepholes.parsers import trim_corevectors
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.models.viz import viz_singular_values_2
from peepholelib.utils.viz_empp import *
from peepholelib.scores.protoclass import conceptogram_protoclass_score as proto_score
# from peepholelib.utils.localization import *
# from peepholelib.plots.conceptograms import plot_conceptogram 
# from peepholelib.utils.get_samples import *
# from calculate_layer_importance import localization_delta_auc_lolo as layer_importance, topk_layers_by_delta_auc as topk_layers 
from peepholelib.utils.localization import localization_from_conceptogram
from peepholelib.utils.gini_index import gini_from_conceptogram

def get_st_list(state_dict):
    '''
    Return a clean list of the layers of the model

    Args:
    - state_dict: state dict of the model
    '''
    state_dict_list = list(state_dict)

    # remove .weight and .bias from the strings in the state_dict list
    st_clean = [s.replace(".bias", "").replace(".weight", "") for s in state_dict]
    filtered_layers = [layer for layer in st_clean if 'mlp.0' in layer or 
                                                'mlp.3' in layer or 
                                                'heads' in layer]
    return filtered_layers

def load_all_drillers(**kwargs):
    n_cluster_list = kwargs.get('n_cluster_list', None)
    target_layers = kwargs.get('target_layers', None)
    device = kwargs.get('device', None)
    feature_sizes = kwargs.get('feature_sizes', None)
    cv_parsers = kwargs.get('cv_parsers', None)
    base_drill_path = kwargs.get('drill_path', None) 

    all_drillers = {}
    for n_cluster in n_cluster_list:
        # assuming u have a folder with all the drillers and u name it like drillers_{n_cluster}
        drill_path = base_drill_path / f"drillers_{n_cluster}" 

        drillers = {}
        for peep_layer in target_layers:
            drillers[peep_layer] = tGMM(
                path=drill_path,
                name=f"classifier.{peep_layer}",  
                nl_classifier=n_cluster,
                nl_model=n_classes,
                n_features=feature_sizes[peep_layer],
                parser=cv_parsers[peep_layer],
                device=device
            )

        for drill_key, driller in drillers.items():
            if driller._empp_file.exists():
                print(f'Loading Classifier for {drill_key}')
                driller.load()

        all_drillers[n_cluster] = drillers

    return all_drillers

if __name__ == "__main__":
        use_cuda = torch.cuda.is_available()
        device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
        torch.cuda.empty_cache()

        #device = torch.device("cpu")
        print(f"Using {device} device")

        #--------------------------------
        # Directories definitions
        #--------------------------------
        cifar_path = '/srv/newpenny/dataset/CIFAR100'
        ds_path = Path('/srv/newpenny/XAI/generated_data/TPAMI/parsed_datasets/CIFAR100_ViT')

        # model parameters
        seed = 29
        bs = 512
        n_threads = 1
        model_dir = Path('/srv/newpenny/XAI/models')
        model_name = 'SV_model=vit_b_16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau_withInfo.pth'        
        
        svds_path = Path('/srv/newpenny/XAI/CN/vit_data')
        svds_name = 'svds' 
        
        cvs_path = Path('/srv/newpenny/XAI/CN/vit_data/corevectors')
        cvs_name = 'corevectors'

        drill_path = Path('/srv/newpenny/XAI/CN/vit_data/drillers_all/drillers_100')
        drill_name = 'classifier'

        phs_path =  Path('/srv/newpenny/XAI/CN/vit_data/peepholes_all/peepholes_100')
        phs_name = 'peepholes'

        plots_path = Path.cwd()/'temp_plots/coverage/'
        
        verbose = True 
        
        # Peepholelib
        

        n_cluster = 100

        n_conceptograms = 2 
        
        loaders = ['CIFAR100-train', 'CIFAR100-val', 'CIFAR100-test']

    #--------------------------------
    # Model 
    #--------------------------------
    
        nn = torchvision.models.vit_b_16()
        n_classes = len(Cifar100.get_classes(meta_path = Path(cifar_path)/'cifar-100-python/meta')) 
        target_layers_all = list(dict.fromkeys(get_st_list(nn.state_dict().keys())))
        
        # best
        target_layers_best_c = [
                        'encoder.layers.encoder_layer_7.mlp.0', 'encoder.layers.encoder_layer_8.mlp.0', 'encoder.layers.encoder_layer_8.mlp.3',
                        'encoder.layers.encoder_layer_9.mlp.0', 'encoder.layers.encoder_layer_9.mlp.3', 'encoder.layers.encoder_layer_10.mlp.0',
                        'encoder.layers.encoder_layer_10.mlp.3', 'encoder.layers.encoder_layer_11.mlp.0', 'encoder.layers.encoder_layer_11.mlp.3', 'heads.head'
                ]

        # worst
        target_layers_worst_c = [
                      'encoder.layers.encoder_layer_0.mlp.3','encoder.layers.encoder_layer_1.mlp.0', 'encoder.layers.encoder_layer_1.mlp.3','encoder.layers.encoder_layer_2.mlp.0',
                      'encoder.layers.encoder_layer_2.mlp.3','encoder.layers.encoder_layer_3.mlp.0','encoder.layers.encoder_layer_3.mlp.3', 
                      'encoder.layers.encoder_layer_4.mlp.0','encoder.layers.encoder_layer_4.mlp.3','encoder.layers.encoder_layer_6.mlp.3'
                ]

        # # 10 best auc

        target_layers_best_auc = [
                'encoder.layers.encoder_layer_0.mlp.0', 'encoder.layers.encoder_layer_0.mlp.3', 'encoder.layers.encoder_layer_1.mlp.0', 'encoder.layers.encoder_layer_1.mlp.3',
                'encoder.layers.encoder_layer_9.mlp.0', 'encoder.layers.encoder_layer_10.mlp.0', 'encoder.layers.encoder_layer_10.mlp.3',
                'encoder.layers.encoder_layer_11.mlp.0', 'encoder.layers.encoder_layer_11.mlp.3', 'heads.head'
                ]

        # worst delta auc
        target_layers_worst_auc = [
                        'encoder.layers.encoder_layer_4.mlp.0', 'encoder.layers.encoder_layer_5.mlp.0', 'encoder.layers.encoder_layer_5.mlp.3',
                        'encoder.layers.encoder_layer_6.mlp.0', 'encoder.layers.encoder_layer_6.mlp.3', 'encoder.layers.encoder_layer_7.mlp.0',
                        'encoder.layers.encoder_layer_7.mlp.3', 'encoder.layers.encoder_layer_8.mlp.0', 'encoder.layers.encoder_layer_8.mlp.3',
                        'encoder.layers.encoder_layer_9.mlp.3'
                        ]
      

        tl_config = {
                'All': target_layers_all,
                'Random': random.sample(target_layers_all, 10),
                'Worst ΔAUC': target_layers_worst_auc,
                'Best ΔAUC': target_layers_best_auc,
                'Worst c': target_layers_worst_c,
                'Best c': target_layers_best_c,            
        }

        datasets = ParsedDataset(
                path = ds_path,
                )
        
        classes = Cifar100.get_classes(meta_path = Path(cifar_path)/'cifar-100-python/meta')

        peepholes = Peepholes(
                path = phs_path,
                name = phs_name,
                device = device
                )
       
        with datasets as ds, peepholes as ph: #corevecs as cv,
                
                ds.load_only(
                        loaders = loaders,
                        verbose = verbose
                        )

                label = ds._dss['CIFAR100-test']['label'].long()
                pred = ds._dss['CIFAR100-test']['pred'].long()

                num_classes = len(classes)

                cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
                cm.index_put_((label, pred), torch.ones_like(label), accumulate=True)

                cm_norm = cm / cm.sum(axis=1, keepdims=True)
                
                per_class_acc = torch.diag(cm) / cm.sum(axis=1)

                ph.load_only(
                                loaders = loaders,
                                verbose = verbose 
                                )

                for config, tl in tl_config.items():
                        print(config)

                        _c = torch.stack(
                                        [ph._phs['CIFAR100-test'][layer]['peepholes'] for layer in tl],
                                        dim=1
                                        )

                        # _c2 = _c**2
                        # loc = _c2.sum(dim=(1,2))/(_c.sum(dim=(1,2)))**2

                        # loc_max = 1/(len(target_layers_best_c))
                        # loc_min = 1/(len(target_layers_best_c)*len(classes))
                        
                        # loc = (loc - loc_min)/(loc_max-loc_min) 

                        loc = gini_from_conceptogram(X=_c)  

                        mean_loc_per_class = np.full(num_classes, np.nan, dtype=np.float64)
                        std_loc_per_class  = np.full(num_classes, np.nan)

                        for c in range(num_classes):
                                idx = (label == c)
                                mean_loc_per_class[c] = loc[idx].mean()
                                std_loc_per_class[c]  = loc[idx].std()

                        x = mean_loc_per_class   
                        y = per_class_acc   

                        pear_r, pear_p = pearsonr(x, y)
                        spear_r, spear_p = spearmanr(x, y)

                        print(f"Pearson r  = {pear_r:.4f}")
                quit()
                print(f"Spearman ρ = {spear_r:.4f} (p={spear_p:.3g})")

                plt.figure(figsize=(6, 5))

                # scatter
                plt.scatter(x, y, alpha=0.7)

                # for xi, yi, c in zip(x, y, classes):
                #         class_name = classes[int(c)]
                #         plt.text(
                #                 xi, yi,
                #                 class_name,
                #                 fontsize=7,
                #                 alpha=0.8
                #         )

                # plt.errorbar(
                #         x,
                #         y,
                #         xerr=std_loc_per_class,   
                #         fmt='o',
                #         alpha=0.6,
                #         capsize=2
                #         )

                m, b = np.polyfit(x, y, 1)
                xx = np.linspace(x.min(), x.max(), 100)
                plt.plot(xx, m * xx + b)

                plt.xlabel("Average Localization")
                plt.ylabel("Accuracy")

                plt.text(
                        0.95, 0.05,
                        f"Pearson r = {pear_r:.3f}",
                        transform=plt.gca().transAxes,
                        va="bottom",
                        ha="right",
                        fontsize=10,
                        bbox=dict(boxstyle="round", alpha=0.2)
                        )
                # plt.title(
                # f"LOC vs Accuracy\n"
                # f"Pearson r={pear_r:.3f}, Spearman ρ={spear_r:.3f}"
                # )

                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig('corr_ViT.png')
                