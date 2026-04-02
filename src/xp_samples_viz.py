import sys
from pathlib import Path as Path

from networkx import config
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
        phs_name = 'peepholes'
        
        loaders = ['CIFAR100-train', 'CIFAR100-val', 'CIFAR100-test']
    
        models = {}
        verbose = True

        # ### MobileNet

        # phs_path_mobile = Path.cwd()/'/srv/newpenny/XAI/CN/mobilenet_data/peepholes_all/peepholes_100'
        
        # ds_path_mobile = Path('/srv/newpenny/XAI/CN/mobilenet_data')

        # target_layers_mobile = [
        #         'features.2.conv.0.0','features.3.conv.2','features.5.conv.1.0',
        #         'features.6.conv.1.0','features.8.conv.1.0',
        #         'features.9.conv.1.0','features.17.conv.1.0','features.17.conv.2',
        #         'features.18.0','classifier.1'
        #         ]

        # models['Mobile'] = {
        #         'phs': phs_path_mobile,
        #         'ds': ds_path_mobile,
        #         'tl': target_layers_mobile,
        #         'xticks': torch.linspace(0, 9, steps=4).long()
        #         }

        ### ViT

        # ds_path_vit = Path('/srv/newpenny/XAI/generated_data/TPAMI/parsed_datasets/CIFAR100_ViT')

        # phs_path_vit =  Path('/srv/newpenny/XAI/CN/vit_data/peepholes_all/peepholes_100')

        # target_layers_vit = [
        #                 'encoder.layers.encoder_layer_7.mlp.0', 'encoder.layers.encoder_layer_8.mlp.0', 'encoder.layers.encoder_layer_8.mlp.3',
        #                 'encoder.layers.encoder_layer_9.mlp.0', 'encoder.layers.encoder_layer_9.mlp.3', 'encoder.layers.encoder_layer_10.mlp.0',
        #                 'encoder.layers.encoder_layer_10.mlp.3', 'encoder.layers.encoder_layer_11.mlp.0', 'encoder.layers.encoder_layer_11.mlp.3', 'heads.head'
        #         ]

        # models['ViT'] = {
        #         'phs': phs_path_vit,
        #         'ds': ds_path_vit,
        #         'tl': target_layers_vit,
        #         'xticks': torch.linspace(0, 9, steps=4).long()
        #         }

        ### VGG

        ds_path_vgg = '/srv/newpenny/XAI/generated_data/TPAMI/parsed_datasets/CIFAR100_VGG16'

        phs_path_vgg = Path('/srv/newpenny/XAI/CN/vgg_data/peepholes_all/peepholes_100')

        target_layers_vgg = ['features.26','features.28','classifier.0','classifier.3', 'classifier.6']

        models['vgg'] = {
                'phs': phs_path_vgg,
                'ds': ds_path_vgg,
                'tl': target_layers_vgg,
                'xticks': torch.linspace(0, 4, steps=3).long()
                }
        
        for model, config in models.items():

                model_path = Path.cwd() / f"{model}_samples_viz"
                model_path.mkdir(parents=True, exist_ok=True)

                datasets = ParsedDataset(
                        path = config['ds'],
                        )
                        
                peepholes = Peepholes(
                        path = config['phs'],
                        name = phs_name,
                        device = device
                        )
        
                classes = Cifar100.get_classes(meta_path = Path(cifar_path)/'cifar-100-python/meta')
       
                with datasets as ds, peepholes as ph: 
                        
                        ds.load_only(
                                loaders = loaders,
                                verbose = verbose
                                )

                        label = ds._dss['CIFAR100-test']['label']
                        pred = ds._dss['CIFAR100-test']['pred']
                        result = ds._dss['CIFAR100-test']['result']

                        ph.load_only(
                                loaders = loaders,
                                verbose = verbose 
                                )

                        sm = torch.nn.Softmax(dim=1)
                        _p = sm(ds._dss['CIFAR100-test']['output'])

                        for c, class_name in classes.items():

                                class_path = model_path / f"{class_name}"
                                class_path.mkdir(parents=True, exist_ok=True)
                        
                                for correct in [0, 1]:

                                        sub_path = class_path / ("correct" if correct else "incorrect")
                                        sub_path.mkdir(parents=True, exist_ok=True)

                                        color = "green" if correct else "red"
                                
                                        idxs = torch.where((label == c) & (result == correct))[0].tolist()[:20]

                                        for idx in idxs:

                                                fig = plt.figure(figsize=(5,16))
                                                gs = gridspec.GridSpec(
                                                        2, 1,
                                                        height_ratios=[0.5,1],  # image smaller than matrices
                                                        wspace=0.2
                                                        )

                                                # ── Denormalize image
                                                img = ds._dss['CIFAR100-test']['image'][idx]

                                                mean = torch.tensor([0.438, 0.418, 0.377]).view(3,1,1)
                                                std = torch.tensor([0.300, 0.287, 0.294]).view(3,1,1)

                                                img = (img * std + mean).clamp(0, 1)

                                                # ── Image subplot (first column)
                                                ax_img = fig.add_subplot(gs[0, 0])
                                                ax_img.imshow(img.cpu().permute(1,2,0))
                                                # ax_img.axis("off")
                                                ax_img.set_title(classes[int(ds._dss['CIFAR100-test']['label'][idx])].capitalize())

                                                ax_c = fig.add_subplot(gs[1,0])

                                                _conceptograms = torch.stack(
                                                        [ph._phs['CIFAR100-test'][layer]['peepholes'] for layer in config['tl']],
                                                        dim=1
                                                )
                                                p = _p[idx]
                                                        
                                                #_c = _conceptograms[idx]
                                                _c = torch.cat((_conceptograms[idx], p.unsqueeze(dim=0)), dim=0)

                                                _c_display = _c.T
                                        
                                                ax_c.imshow(
                                                        1 - _c_display[:, :-1], 
                                                        aspect="auto",
                                                        vmin=0.0,
                                                        vmax=1.0,
                                                        cmap="bone",
                                                        extent=[0, _c.shape[0]-1, _c.shape[1], 0]
                                                        )

                                                ax_c.imshow(
                                                        _c_display[:, -1:],
                                                        aspect="auto",
                                                        vmin=0.0,
                                                        vmax=1.0,
                                                        cmap="YlOrRd", 
                                                        extent=[_c.shape[0]-1, _c.shape[0], _c.shape[1], 0]
                                                        )

                                                _, idx_topk = torch.topk(_c.sum(dim=0), 3, sorted=True)
                                                classes_topk = [classes[i] for i in idx_topk.tolist()]
                                                tick_labels = [f'{cls.capitalize()}' for i, cls in enumerate(classes_topk)]
                                                ax_c.set_yticks(idx_topk, tick_labels, fontsize=15)
                                                ax_c.yaxis.tick_right()

                                                xticks = torch.linspace(0, len(config['tl'])-1, steps=4).long()

                                                ax_c.set_xticks(xticks)

                                                _c2 = _conceptograms**2
                                                loc = _c2.sum(dim=(1,2))/len(config['tl'])

                                                ax_c.set_title(f'σ: {loc[idx]:.2f}\n conf: {p.max().numpy():.2f}', fontsize=15)
                                                
                                                ax_img.set_xticks([])
                                                ax_img.set_yticks([])

                                                ax_img.set_frame_on(True)
                                                for spine in ax_img.spines.values():
                                                        spine.set_visible(True)
                                                        spine.set_edgecolor(color)
                                                        spine.set_linewidth(4)


                                                plt.tight_layout()
                                                fig.savefig(sub_path / f"sample_{idx}.png", bbox_inches="tight")