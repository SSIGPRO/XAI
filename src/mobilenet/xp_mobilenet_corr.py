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
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr, spearmanr

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
from peepholelib.utils.gini_index import gini_from_conceptogram
# from peepholelib.utils.viz_corevecs import plot_tsne, plot_tsne_CUDA
# from peepholelib.utils.localization import *
# from peepholelib.utils.get_samples import *
# from peepholelib.scores.protoclass import conceptogram_protoclass_score as proto_score
# from peepholelib.plots.conceptograms import plot_conceptogram 

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
        ds_path = Path('/srv/newpenny/XAI/CN/mobilenet_data')

        # model parameters
        seed = 29
        bs = 512
        n_threads = 1

        model_dir = '/srv/newpenny/XAI/models'
        model_name = 'CN_model=mobilenet_v2_dataset=CIFAR100_optim=Adam_scheduler=RoP_lr=0.001_factor=0.1_patience=5.pth'
        
        svds_path = '/srv/newpenny/XAI/CN/mobilenet_data'
        svds_name = 'svds' 
        
        cvs_path = Path.cwd()/'/srv/newpenny/XAI/CN/mobilenet_data/corevectors'
        cvs_name = 'corevectors'

        drill_path = Path.cwd()/'/srv/newpenny/XAI/CN/mobilenet_data/drillers_all/drillers_100'
        drill_name = 'classifier'

        phs_path = Path.cwd()/'/srv/newpenny/XAI/CN/mobilenet_data/peepholes_all/peepholes_100'
        phs_name = 'peepholes'

        plots_path = Path.cwd()/'temp_plots/coverage/'
        
        verbose = True 
        
        target_layers_all = [ 
                'features.1.conv.0.0', 'features.1.conv.1','features.2.conv.0.0',
                'features.2.conv.1.0','features.2.conv.2',
                'features.3.conv.0.0', 'features.3.conv.1.0', 'features.3.conv.2',
                'features.4.conv.0.0', 'features.4.conv.1.0', 'features.4.conv.2',
                'features.5.conv.0.0', 'features.5.conv.1.0', 'features.5.conv.2',
                'features.6.conv.0.0','features.6.conv.1.0', 'features.6.conv.2',
                'features.7.conv.0.0', 'features.7.conv.1.0','features.7.conv.2',
                'features.8.conv.0.0', 'features.8.conv.1.0', 'features.8.conv.2',
                'features.9.conv.0.0', 'features.9.conv.1.0', 'features.9.conv.2',  
                'features.10.conv.0.0', 'features.10.conv.1.0', 'features.10.conv.2',
                'features.11.conv.0.0', 'features.11.conv.1.0', 'features.11.conv.2',
                'features.12.conv.0.0', 'features.12.conv.1.0',  'features.12.conv.2',
                'features.13.conv.0.0', 'features.13.conv.1.0', 'features.13.conv.2',
                'features.14.conv.0.0', 'features.14.conv.1.0', 'features.14.conv.2',
                'features.15.conv.0.0', 'features.15.conv.1.0', 'features.15.conv.2',
                'features.16.conv.0.0', 'features.16.conv.1.0', 'features.16.conv.2', 
                'features.17.conv.0.0', 'features.17.conv.1.0', 'features.17.conv.2',
                'features.18.0', 'classifier.1',
               ]

        #worst c
        target_layers_worst_c = [
                'features.1.conv.0.0','features.4.conv.1.0','features.4.conv.2', 
                'features.7.conv.2','features.8.conv.0.0',
                'features.8.conv.2','features.9.conv.0.0', 'features.11.conv.1.0',
                'features.12.conv.1.0','features.13.conv.2'
                 ]

        # best auc
        target_layers_best_auc = [
                'features.1.conv.1','features.2.conv.0.0', 'features.3.conv.1.0', 
                'features.5.conv.1.0', 'features.6.conv.1.0','features.8.conv.1.0',
                'features.17.conv.1.0', 'features.17.conv.2', 'features.18.0', 'classifier.1'
                ]

        # worst auc
        target_layers_worst_auc = [
                'features.11.conv.1.0', 'features.11.conv.2', 'features.14.conv.1.0',
                'features.14.conv.2', 'features.15.conv.0.0',  'features.15.conv.1.0',
                'features.15.conv.2', 'features.16.conv.0.0', 'features.16.conv.1.0', 'features.16.conv.2'
                ]

        # #best fr95
        # target_layers=[
        #         'features.14.conv.2', 'features.17.conv.0.0', 'features.17.conv.2', 'features.15.conv.2', 'features.11.conv.2','features.17.conv.1.0', 
        #         'features.14.conv.1.0','features.15.conv.0.0','features.18.0', 'classifier.1'
        # ]
        #best coverage (threshold =0.7-0.89)
        target_layers_best_c = [
                'features.2.conv.0.0','features.3.conv.2','features.5.conv.1.0',
                'features.6.conv.1.0','features.8.conv.1.0',
                'features.9.conv.1.0','features.17.conv.1.0','features.17.conv.2',
                'features.18.0','classifier.1'
                ]

        #best coverage (threshold =0.95)
        # target_layers = ['features.2.conv.0.0','features.3.conv.0.0','features.3.conv.1.0','features.3.conv.2','features.5.conv.1.0',
        # 'features.6.conv.1.0','features.8.conv.1.0','features.9.conv.1.0','features.17.conv.2','classifier.1']

        tl_config = {
                #'All': target_layers_all,
                'Random': random.sample(target_layers_all, 10),
                'Best c': target_layers_best_c,
                'Worst c': target_layers_worst_c,
                'Best ΔAUC': target_layers_best_auc,
                'Worst ΔAUC': target_layers_worst_auc,
        }

        loaders = [
                'CIFAR100-train',
                'CIFAR100-val',
                'CIFAR100-test',
                ]

        datasets = ParsedDataset(
                path = ds_path,
                )
                
        classes = Cifar100.get_classes(meta_path = Path(cifar_path)/'cifar-100-python/meta')
        
        peepholes = Peepholes(
                path = phs_path,
                name = phs_name,
                device = device
                )

        with datasets as ds, peepholes as ph: #, corevecs as cv

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

                        #_c2 = _c**2
                        # loc = _c2.sum(dim=(1,2))/(_c.sum(dim=(1,2)))**2

                        # loc_max = 1/(len(target_layers_best_c))
                        # loc_min = 1/(len(target_layers_best_c)*len(classes))
                        
                        # loc = (loc - loc_min)/(loc_max-loc_min) 

                        loc = gini_from_conceptogram(X=_c)
                        print(loc.shape)

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
                        #print(f"Spearman ρ = {spear_r:.4f} (p={spear_p:.3g})")
                quit()

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

                # plt.xlabel("Mean LOC (per class)")
                # plt.ylabel("Accuracy (per class)")
                # plt.title(
                # f"LOC vs Accuracy\n"
                # f"Pearson r={pear_r:.3f}, Spearman ρ={spear_r:.3f}"
                # )

                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig('corr_mobile.png')