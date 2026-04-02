import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# import cuml
# cuml.accel.install()

# python stuff
from time import time
from functools import partial
import random
from matplotlib import pyplot as plt
plt.rc('font', size=10)          
import matplotlib.gridspec as gridspec
from math import ceil

# torch stuff
import torch
import torchvision
from cuda_selector import auto_cuda

###### Our stuff

# Model
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.models.svd_fns import linear_svd, conv2d_toeplitz_svd

# datasets
from peepholelib.datasets.cifar100 import Cifar100
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.datasets.functional.parsers import from_dataset
from peepholelib.datasets.functional.samplers import random_subsampling 

# corevecs
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import linear_svd_projection, conv2d_toeplitz_svd_projection

# peepholes
from peepholelib.peepholes.parsers import trim_corevectors
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes
# from peepholelib.models.viz import viz_singular_values_2
from peepholelib.utils.viz_empp import *
from peepholelib.utils.localization import localization_from_conceptogram

if __name__ == "__main__":
        use_cuda = torch.cuda.is_available()
        device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
        #device = torch.device('cuda:1') 
        torch.cuda.empty_cache()

        #device = torch.device("cpu")
        print(f"Using {device} device")

        #--------------------------------
        # Directories definitions
        #--------------------------------
        cifar_path = '/srv/newpenny/dataset/CIFAR100'
        sm = torch.nn.Softmax(dim=0)

        # model parameters
        seed = 29
        bs = 512
        n_threads = 1
        phs_name = 'peepholes'
        verbose = True 
        loaders = [
                'CIFAR100-train',
                'CIFAR100-val',
                'CIFAR100-test',
                ]
        classes = Cifar100.get_classes(meta_path = Path(cifar_path)/'cifar-100-python/meta')
        mean = torch.tensor([0.438, 0.418, 0.377]).view(3,1,1)
        std = torch.tensor([0.300, 0.287, 0.294]).view(3,1,1)

        models = {}

        ### MobileNet

        phs_path_mobile = Path.cwd()/'/srv/newpenny/XAI/CN/mobilenet_data/peepholes_all/peepholes_100'
        
        ds_path_mobile = Path('/srv/newpenny/XAI/CN/mobilenet_data')

        target_layers_mobile = [
                'features.2.conv.0.0','features.3.conv.2','features.5.conv.1.0',
                'features.6.conv.1.0','features.8.conv.1.0',
                'features.9.conv.1.0','features.17.conv.1.0','features.17.conv.2',
                'features.18.0','classifier.1'
                ]

        models['Mobile'] = {
                'phs': phs_path_mobile,
                'ds': ds_path_mobile,
                'tl': target_layers_mobile,
                'xticks': torch.linspace(0, 9, steps=4).long()
                }

        ### ViT

        ds_path_vit = Path('/srv/newpenny/XAI/generated_data/TPAMI/parsed_datasets/CIFAR100_ViT')

        phs_path_vit =  Path('/srv/newpenny/XAI/CN/vit_data/peepholes_all/peepholes_100')

        target_layers_vit = [
                        'encoder.layers.encoder_layer_7.mlp.0', 'encoder.layers.encoder_layer_8.mlp.0', 'encoder.layers.encoder_layer_8.mlp.3',
                        'encoder.layers.encoder_layer_9.mlp.0', 'encoder.layers.encoder_layer_9.mlp.3', 'encoder.layers.encoder_layer_10.mlp.0',
                        'encoder.layers.encoder_layer_10.mlp.3', 'encoder.layers.encoder_layer_11.mlp.0', 'encoder.layers.encoder_layer_11.mlp.3', 'heads.head'
                ]

        models['ViT'] = {
                'phs': phs_path_vit,
                'ds': ds_path_vit,
                'tl': target_layers_vit,
                'xticks': torch.linspace(0, 9, steps=4).long()
                }

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

                datasets = ParsedDataset(
                        path = config['ds'],
                        )
                        
                peepholes = Peepholes(
                        path = config['phs'],
                        name = phs_name,
                        device = device
                        )

                with datasets as ds, peepholes as ph:

                        ds.load_only(
                                loaders = loaders,
                                verbose = verbose
                                )

                        ph.load_only(
                                loaders = loaders,
                                verbose = verbose 
                                )

                        _conceptograms = torch.stack(
                                        [ph._phs['CIFAR100-test'][layer]['peepholes'] for layer in config['tl']],
                                        dim=1
                                )
                        
                        loc = (_conceptograms**2).sum(dim=(1,2))/len(config['tl'])

                        r = ds._dss['CIFAR100-test']['result']

                        for i in range(100):

                                fig = plt.figure(figsize=(12,8))

                                gs = gridspec.GridSpec(
                                        2, 2,
                                        height_ratios=[1.5, 6],
                                        width_ratios=[1, 1],
                                        hspace=0.22,
                                        wspace=0.0
                                        )

                                axs_img = [fig.add_subplot(gs[0, j]) for j in range(2)]
                                
                                axs_mat = [fig.add_subplot(gs[1, j]) for j in range(2)]

                                idxs_max = ((ds._dss['CIFAR100-test']['label'] == i)&(r == 1)).nonzero(as_tuple=True)[0]
                                #idxs_min = ((ds._dss['CIFAR100-test']['label'] == i)&(r == 1)).nonzero(as_tuple=True)[0]

                                max_idx = torch.argmax(loc[idxs_max])
                                min_idx = torch.argmin(loc[idxs_max])

                                for j, idx in enumerate([idxs_max[min_idx], idxs_max[max_idx]]):

                                        img = ds._dss['CIFAR100-test']['image'][idx]
                                        p = sm(ds._dss['CIFAR100-test']['output'][idx])
                                        
                                        _c = torch.cat((_conceptograms[idx], p.unsqueeze(dim=0)), dim=0)
                                        _r = r[idx]

                                        img = (img * std + mean).clamp(0, 1)

                                        axs_img[j].imshow(img.cpu().permute(1,2,0), aspect='auto')
                                        axs_img[j].set_yticks([])
                                        axs_img[j].set_xticks([])
                                        
                                        axs_img[j].set_frame_on(True)
                                        label = classes[int(ds._dss['CIFAR100-test']['label'][idx])].capitalize()
                                        pred = classes[int(ds._dss['CIFAR100-test']['pred'][idx])].capitalize()
                                        conf = sm(ds._dss['CIFAR100-test']['output'][idx]).max().item() * 100

                                        color = "green" if _r == 1 else "red"
                                        for spine in axs_img[j].spines.values():
                                                spine.set_edgecolor(color)
                                                spine.set_linewidth(4)

                                        axs_img[j].text(
                                                        0.5, -0.15,
                                                        f"Conf: {conf:.1f}%\nσ: {loc[idx]:.2f}", #Conf: {conf:.1f}%\nLabel: {label}\nPred: {pred}\n
                                                        transform=axs_img[j].transAxes,
                                                        va="top",
                                                        ha="center",
                                                        fontsize=12
                                                        ) 
                                        axs_img[j].set_title(f'{label}: {idx}', fontsize=14)                               

                                        axs_mat[j].imshow(
                                                        1 - _c.T,
                                                        aspect="auto",
                                                        vmin=0.0,
                                                        vmax=1.0,
                                                        cmap="bone"
                                                )

                                        xticks = config['xticks']
                                        axs_mat[j].set_xticks(xticks)

                                        _, idx_topk = torch.topk(_c.sum(dim=0), 3, sorted=True)
                                        classes_topk = [classes[i] for i in idx_topk.tolist()]
                                        tick_labels = [f'{cls.capitalize()}' for i, cls in enumerate(classes_topk)]
                                        axs_mat[j].set_yticks(idx_topk, tick_labels, fontsize=12)
                                        axs_mat[j].yaxis.tick_right()

                                        axs_mat[j].set_box_aspect(4.0) 
                                        axs_img[j].set_box_aspect(1.0)

                                        axs_mat[j].margins(x=0, y=0)
                                        # axs_img[i].margins(x=0, y=0)
                                fig.tight_layout()
                                models_path = Path.cwd()/f'../model={model}_correct'
                                models_path.mkdir(parents=True, exist_ok=True)
                                fig.savefig(models_path/f'corner_case_{model}_{i}.png', dpi=300, bbox_inches="tight")