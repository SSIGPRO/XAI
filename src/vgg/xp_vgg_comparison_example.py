import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# python stuff
import random
from matplotlib import pyplot as plt
          
import matplotlib.gridspec as gridspec
import numpy as np

# torch stuff
import torch
from cuda_selector import auto_cuda
import torchvision
import torch  

# datasets
from peepholelib.datasets.cifar100 import Cifar100
from peepholelib.datasets.parsedDataset import ParsedDataset 

# peepholes
from peepholelib.peepholes.peepholes import Peepholes

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
        ds_path = '/srv/newpenny/XAI/generated_data/TPAMI/parsed_datasets/CIFAR100_VGG16'

        # model parameters
        seed = 29
        bs = 512
        n_threads = 1

        model_dir = Path('/srv/newpenny/XAI/models')
        model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
        
        svds_path = Path('/srv/newpenny/XAI/CN/vgg_data')
        svds_name = 'svds'  
        
        cvs_path = Path('/srv/newpenny/XAI/CN/vgg_data/corevectors')
        cvs_name = 'corevectors'

        drill_path = Path('/srv/newpenny/XAI/CN/vgg_data/drillers_all/drillers_100')
        drill_name = 'classifier'

        phs_path =  Path('/srv/newpenny/XAI/CN/vgg_data/peepholes_all/peepholes_100')
        phs_name = 'peepholes'

        plots_path = Path.cwd()/'temp_plots/coverage/'
        
        verbose = True 

        features_cv_dim = 100

        
        target_layers_all = ['features.0', 'features.2', 'features.5', 'features.7', 'features.10', 'features.12', 'features.14', 'features.17', 'features.19', 'features.21',
                                'features.24','features.26','features.28','classifier.0','classifier.3', 
                                'classifier.6',
                        ]

        # best (0.8)
        target_layers_best_c = ['features.26','features.28','classifier.0','classifier.3', 'classifier.6']

        # worst
        target_layers_worst_c = ['features.0', 'features.2', 'features.5', 'features.7', 'features.10']

        # best auc
        target_layers_best_auc = ['features.0', 'features.7' ,'classifier.0','classifier.3', 'classifier.6']

        #worst auc
        target_layers_worst_auc= ['features.19', 'features.21', 'features.24', 'features.26', 'features.28']

        loaders = ['CIFAR100-train', 'CIFAR100-val', 'CIFAR100-test']

        plots_path = Path.cwd()/'temp_plots/coverage/'

      

        tl_config = {
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

                for idx in [4063]:

                        n_cols = len(tl_config)

                        fig = plt.figure(figsize=(10,8))
                        gs = gridspec.GridSpec(
                                2, n_cols,
                                width_ratios=[3]*(n_cols),  
                                height_ratios=[1,3],
                                hspace=0.2,
                                wspace=0.3
                                )


                        img = ds._dss['CIFAR100-test']['image'][idx]

                        mean = torch.tensor([0.438, 0.418, 0.377]).view(3,1,1)
                        std = torch.tensor([0.300, 0.287, 0.294]).view(3,1,1)

                        img = (img * std + mean).clamp(0, 1)


                        ax_img = fig.add_subplot(gs[0,:])
                        # ax_img = fig.add_subplot(gs[0, 0])
                        ax_img.imshow(img.cpu().permute(1,2,0))
                        ax_img.axis("off")
                        ax_img.set_title(classes[int(ds._dss['CIFAR100-test']['label'][idx])].capitalize())

                        # ── Conceptograms (remaining columns)
                        axs = [fig.add_subplot(gs[1, i]) for i in range(n_cols)]

                        for i, (config, tl) in enumerate(tl_config.items()):
                                if config == 'Random':
                                        tl = [l for l in target_layers_all if l in tl]
                                
                                _p = sm(ds._dss['CIFAR100-test']['output'])
                                p = _p[idx]


                                _conceptograms = torch.stack(
                                        [ph._phs['CIFAR100-test'][layer]['peepholes'] for layer in tl],
                                        dim=1
                                )
                                
                                _c = torch.cat((_conceptograms[idx], p.unsqueeze(dim=0)), dim=0)
                                
                                
                                _c_display = _c.T

                                axs[i].imshow(
                                        1 - _c_display[:, :-1],
                                        aspect="auto",
                                        vmin=0.0,
                                        vmax=1.0,
                                        cmap="bone",
                                        extent=[0, _c.shape[0]-1, _c.shape[1], 0]
                                        )

                                axs[i].imshow(
                                        _c_display[:, -1:],  
                                        aspect="auto",
                                        vmin=0.0,
                                        vmax=1.0,
                                        cmap="YlOrRd", 
                                        extent=[_c.shape[0]-1, _c.shape[0], _c.shape[1], 0]
                                        )

                                if config == 'Best c':
                                        _, idx_topk = torch.topk(_c.sum(dim=0), 3, sorted=True)
                                        classes_topk = [classes[i] for i in idx_topk.tolist()]
                                        tick_labels = [f'{cls.capitalize()}' for i, cls in enumerate(classes_topk)]
                                        axs[i].set_yticks(idx_topk+0.5, tick_labels, fontsize=15)
                                        axs[i].yaxis.tick_right()
                                else: axs[i].set_yticks([])

                                xticks = torch.linspace(0, len(tl)-1, steps=4).long()

                                axs[i].set_xticks(xticks)

                                axs[i].set_xlabel(config, fontsize=15)

                                _c2 = _conceptograms**2
                                loc = _c2.sum(dim=(1,2))/len(tl)

                                axs[i].set_title(f'σ: {loc[idx]:.2f}', fontsize=15)

                                plt.tight_layout()
                        fig.savefig(f'comparison_example.png', bbox_inches="tight")