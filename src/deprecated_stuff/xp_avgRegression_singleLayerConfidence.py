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
import seaborn as sb
import pandas as pd
from math import floor

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
from torch.nn.functional import softmax as sm
from torcheval.metrics import BinaryAUROC as AUC

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
#     name_model = 'ViT'
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
    
    #target_layers =['encoder.layers.encoder_layer_11.mlp.3']
    
    target_layers = ['features.28']
          
    model.set_target_modules(target_modules=target_layers, verbose=verbose)

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
    loaders = ['train', 'test', 'val']   
    drop_max = 100
    fig, axs = plt.subplots(2, len(loaders), sharex='row', sharey='row', figsize=(4*len(loaders), 4))

    with corevectors as cv, peepholes as ph:

        cv.load_only(loaders = loaders,
                        verbose = True
                        )

        ph.load_only(
                loaders = loaders,
                verbose = True
                )
        
        for loader_n, ds_key in enumerate(loaders):
                scores = ph._phs[ds_key][target_layers[0]]['score_max']
                
                ns = scores.shape[0]
                results = cv._dss[ds_key]['result']
                labels = (cv._dss[ds_key]['label']).int()
                confs = sm(cv._dss[ds_key]['output'], dim=-1).max(dim=-1).values

                # compute AUC for score
                s_auc = AUC().update(scores, results.int()).compute().item()

                if verbose: print(f'AUC for {ds_key} split: {s_auc:.4f}')
                
                s_oks = scores[results == True]
                s_kos = scores[results == False]
                m_oks = confs[results == True]
                m_kos = confs[results == False]
                
                # compute AUC for model 
                m_auc = AUC().update(confs, results.int()).compute().item()
                
                df = pd.DataFrame({
                        'Value': torch.hstack((s_oks, s_kos)), #, m_oks, m_kos
                        'Score': \
                                ['DMD: OK' for i in range(len(s_oks))] + \
                                ['DMD: KO' for i in range(len(s_kos))] #+ \
                                # ['Model: OK' for i in range(len(m_oks))] + \
                                # ['Model: KO' for i in range(len(m_kos))]

                        })
                colors = ['xkcd:cobalt', 'xkcd:bluish green'] #, 'xkcd:bluish green', 'xkcd:bluish green'

                # effective plotting
                ax = axs[0][loader_n] 
                # p = sb.kdeplot(
                #         data = df,
                #         ax = ax,
                #         x = 'Value',
                #         hue = 'Score',
                #         common_norm = False,
                #         palette = colors,
                #         clip = [0., 1.],
                #         alpha = 0.8,
                #         legend = loader_n == 0,
                #         )
                ax.hist(s_oks, bins=30, alpha=0.7, label='DMD: OK')
                ax.hist(s_kos, bins=30, alpha=0.7, label='DMD: KO')
                # p = sb.histplot(
                #                 data=df,
                #                 x='Value',
                #                 hue='Score',
                #                 multiple='dodge',      # 'layer' to overlay, 'dodge' to side-by-side, 'stack' to stack
                #                 bins=5,               # adjust number of bins as needed
                #                 alpha=0.7,             # transparency
                #                 palette=colors,
                #                 edgecolor='black',     # optional: black edges for clarity
                #                 ax=ax,
                #                 legend=(loader_n == 0)
                #                 )

                # lines = ['--', '-', '--', '-']
                # # set up linestyles
                # for ls, line in zip(lines, p.lines):
                #         line.set_linestyle(ls)
                
                # # set legend linestyle
                # if loader_n == 0:
                #         handles = p.legend_.legend_handles[::-1]
                #         for ls, h in zip(lines, handles):
                #                 h.set_ls(ls)

                ax.set_xlabel('Score')
                ax.set_ylabel('%')
                ax.title.set_text(f'{ds_key}\nDMD AUC={s_auc:.4f}\nModel AUC={m_auc:.4f}')

                # plot dropping-out accuracy plot
                _, s_idx = scores.sort()
                _, m_idx = confs.sort()
                s_acc = torch.zeros(drop_max+1)
                m_acc = torch.zeros(drop_max+1)
                for drop_perc in range(drop_max+1):
                        n_drop = floor((drop_perc/100)*ns)
                        s_acc[drop_perc] = 100*(results[s_idx[n_drop:]]).sum()/(ns-n_drop)
                        m_acc[drop_perc] = 100*(results[m_idx[n_drop:]]).sum()/(ns-n_drop)
                
                colors = ['xkcd:cobalt', 'xkcd:bluish green']
                ax = axs[1][loader_n]
                df = pd.DataFrame({
                        'Values': torch.hstack((s_acc, m_acc)),
                        'Score': \
                                ['DMD' for i in range(drop_max+1)] + \
                                ['Model confidece' for i in range(drop_max+1)]
                        })

                sb.lineplot(
                        data = df,
                        ax = ax,
                        x = torch.linspace(0, drop_max, drop_max+1).repeat(2),
                        y = 'Values',
                        hue = 'Score',
                        palette = colors,
                        alpha = 0.8,
                        legend = loader_n == 0,
                        )
                ax.set_xlabel('% dropped')
                ax.set_ylabel('Accuracy (%)')

        
        plt.savefig(f'DMD_singleLayer_Confidence_{name_model}.png', dpi=300, bbox_inches='tight')
        plt.close()
                        

                
                
                
                
