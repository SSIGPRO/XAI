import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# python stuff
import random
from matplotlib import pyplot as plt        
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import pearsonr

# torch stuff
import torch
import torchvision
from cuda_selector import auto_cuda

# datasets
from peepholelib.datasets.cifar100 import Cifar100
from peepholelib.datasets.parsedDataset import ParsedDataset 

# peepholes
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.utils.viz_empp import *

if __name__ == "__main__":
        use_cuda = torch.cuda.is_available()
        device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")

        print(f"Using {device} device")

        #--------------------------------
        # Directories definitions
        #--------------------------------
        cifar_path = '/srv/newpenny/dataset/CIFAR100'
        sm = torch.nn.Softmax(dim=1)

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
        
        target_layers_mobile = {

                'All': [ 
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
                ],

                'Random': [],
                
                'Best c': [
                        'features.2.conv.0.0','features.3.conv.2','features.5.conv.1.0',
                        'features.6.conv.1.0','features.8.conv.1.0',
                        'features.9.conv.1.0','features.17.conv.1.0','features.17.conv.2',
                        'features.18.0','classifier.1'],

                'Worst c':  [
                                'features.1.conv.0.0','features.4.conv.1.0','features.4.conv.2', 
                                'features.7.conv.2','features.8.conv.0.0',
                                'features.8.conv.2','features.9.conv.0.0', 'features.11.conv.1.0',
                                'features.12.conv.1.0','features.13.conv.2'
                                ],

                'Best ΔAUC': [
                'features.1.conv.1','features.2.conv.0.0', 'features.3.conv.1.0', 
                'features.5.conv.1.0', 'features.6.conv.1.0','features.8.conv.1.0',
                'features.17.conv.1.0', 'features.17.conv.2', 'features.18.0', 'classifier.1'],

                'Worst ΔAUC': [
                'features.11.conv.1.0', 'features.11.conv.2', 'features.14.conv.1.0',
                'features.14.conv.2', 'features.15.conv.0.0',  'features.15.conv.1.0',
                'features.15.conv.2', 'features.16.conv.0.0', 'features.16.conv.1.0', 'features.16.conv.2'
                ]
        }

        models['MobileNetV2'] = {
                'phs': phs_path_mobile,
                'ds': ds_path_mobile,
                'tl': target_layers_mobile,
                'numLayers': 10,
                'xlim': 0.35
                }

        ### ViT

        ds_path_vit = Path('/srv/newpenny/XAI/generated_data/TPAMI/parsed_datasets/CIFAR100_ViT')

        phs_path_vit =  Path('/srv/newpenny/XAI/CN/vit_data/peepholes_all/peepholes_100')

        tls = [
        f'encoder.layers.encoder_layer_{i}.mlp.{j}' for i in range(12) for j in [0,3]
        ]
        tls.append('heads.head')
        
        target_layers_vit = {

                'All': tls,

                'Random': [],
                
                'Best c': [
                        'encoder.layers.encoder_layer_7.mlp.0', 'encoder.layers.encoder_layer_8.mlp.0', 'encoder.layers.encoder_layer_8.mlp.3',
                        'encoder.layers.encoder_layer_9.mlp.0', 'encoder.layers.encoder_layer_9.mlp.3', 'encoder.layers.encoder_layer_10.mlp.0',
                        'encoder.layers.encoder_layer_10.mlp.3', 'encoder.layers.encoder_layer_11.mlp.0', 'encoder.layers.encoder_layer_11.mlp.3', 'heads.head'],

                'Worst c': [
                      'encoder.layers.encoder_layer_0.mlp.3','encoder.layers.encoder_layer_1.mlp.0', 'encoder.layers.encoder_layer_1.mlp.3','encoder.layers.encoder_layer_2.mlp.0',
                      'encoder.layers.encoder_layer_2.mlp.3','encoder.layers.encoder_layer_3.mlp.0','encoder.layers.encoder_layer_3.mlp.3', 
                      'encoder.layers.encoder_layer_4.mlp.0','encoder.layers.encoder_layer_4.mlp.3','encoder.layers.encoder_layer_6.mlp.3'
                ],

                'Best ΔAUC': [
                        'encoder.layers.encoder_layer_0.mlp.0', 'encoder.layers.encoder_layer_0.mlp.3', 'encoder.layers.encoder_layer_1.mlp.0', 'encoder.layers.encoder_layer_1.mlp.3',
                        'encoder.layers.encoder_layer_9.mlp.0', 'encoder.layers.encoder_layer_10.mlp.0', 'encoder.layers.encoder_layer_10.mlp.3',
                        'encoder.layers.encoder_layer_11.mlp.0', 'encoder.layers.encoder_layer_11.mlp.3', 'heads.head'],

                'Worst ΔAUC': [
                        'encoder.layers.encoder_layer_4.mlp.0', 'encoder.layers.encoder_layer_5.mlp.0', 'encoder.layers.encoder_layer_5.mlp.3',
                        'encoder.layers.encoder_layer_6.mlp.0', 'encoder.layers.encoder_layer_6.mlp.3', 'encoder.layers.encoder_layer_7.mlp.0',
                        'encoder.layers.encoder_layer_7.mlp.3', 'encoder.layers.encoder_layer_8.mlp.0', 'encoder.layers.encoder_layer_8.mlp.3',
                        'encoder.layers.encoder_layer_9.mlp.3'
                        ]
        }

        models['ViTB16'] = {
                'phs': phs_path_vit,
                'ds': ds_path_vit,
                'tl': target_layers_vit,
                'numLayers': 10,
                'xlim': 0.35
                }

        ### VGG

        ds_path_vgg = '/srv/newpenny/XAI/generated_data/TPAMI/parsed_datasets/CIFAR100_VGG16'

        phs_path_vgg = Path('/srv/newpenny/XAI/CN/vgg_data/peepholes_all/peepholes_100')

        target_layers_vgg = {
                'All': ['features.0', 'features.2', 'features.5', 'features.7', 'features.10', 'features.12', 'features.14', 'features.17', 'features.19', 'features.21',
                                'features.24','features.26','features.28','classifier.0','classifier.3', 
                                'classifier.6',],

                'Random': [],
                
                'Best c': ['features.26','features.28','classifier.0','classifier.3', 'classifier.6'],

                'Worst c': ['features.0', 'features.2', 'features.5', 'features.7', 'features.10'],

                'Best ΔAUC': ['features.0', 'features.10', 'features.17','classifier.3', 'classifier.6'],

                'Worst ΔAUC': ['features.19', 'features.21', 'features.24', 'features.26', 'features.28']
        }

        models['VGG16'] = {
                'phs': phs_path_vgg,
                'ds': ds_path_vgg,
                'tl': target_layers_vgg,
                'numLayers': 5,
                'xlim': 0.35
                }
        
        fig, axs = plt.subplots(1,3, figsize=(18,6), sharey=True)
        num_classes = len(classes)
        fontsize = 17

        for i, (model, config) in enumerate(models.items()):

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
                        print('--------------------')
                        print(f'Model: {model}')
                        print('--------------------')
                        
                        label = ds._dss['CIFAR100-test']['label'].long()
                        pred = ds._dss['CIFAR100-test']['pred'].long()

                        num_classes = len(classes)

                        cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
                        cm.index_put_((label, pred), torch.ones_like(label), accumulate=True)

                        cm_norm = cm / cm.sum(axis=1, keepdims=True)
                        
                        per_class_acc = torch.diag(cm) / cm.sum(axis=1)

        #                 ### Table 1 computation

        #                 for config_name, target_layers_selection in config['tl'].items():

        #                         if config_name != 'Random':
        #                                 _c = torch.stack(
        #                                                 [ph._phs['CIFAR100-test'][layer]['peepholes'] for layer in target_layers_selection],
        #                                                 dim=1
        #                                         )
                                        
        #                                 _c2 = _c**2
        #                                 loc = _c2.sum(dim=(1,2))/(_c.sum(dim=(1,2)))**2

        #                                 loc_max = 1/(len(target_layers_selection))
        #                                 loc_min = 1/(len(target_layers_selection)*len(classes))
                                        
        #                                 loc = (loc - loc_min)/(loc_max-loc_min)

        #                                 # loc = gini_from_conceptogram(X=_c)
        #                                 conf, _ = sm(ds._dss['CIFAR100-test']['output']).max(dim=1)

        #                                 mean_loc_per_class = np.full(num_classes, np.nan, dtype=np.float64)
        #                                 std_loc_per_class  = np.full(num_classes, np.nan)

        #                                 for c in range(num_classes):
        #                                         idx = (label == c)
        #                                         mean_loc_per_class[c] = loc[idx].mean()
        #                                         std_loc_per_class[c]  = loc[idx].std()

                                        
        #                                 mean_conf_per_class = np.full(num_classes, np.nan, dtype=np.float64)

        #                                 for c in range(num_classes):
        #                                         idx = (label == c)
        #                                         mean_conf_per_class[c] = conf[idx].mean()

        #                                 x = mean_loc_per_class   
        #                                 y = mean_conf_per_class   

        #                                 pear_r, pear_p = pearsonr(x, y)
        #                         else:
        #                                 pearson_mean = []
        #                                 loc_mean = []
        #                                 for i in range(20):
        #                                         target_layers_random = random.sample(config['tl']['All'], config['numLayers']) 

        #                                         _c = torch.stack(
        #                                                         [ph._phs['CIFAR100-test'][layer]['peepholes'] for layer in target_layers_random],
        #                                                         dim=1
        #                                                 )
                                        
        #                                         _c2 = _c**2
        #                                         loc = _c2.sum(dim=(1,2))/config['numLayers']

        #                                         conf, _ = sm(ds._dss['CIFAR100-test']['output']).max(dim=1)

        #                                         mean_loc_per_class = np.full(num_classes, np.nan, dtype=np.float64)

        #                                         for c in range(num_classes):
        #                                                 idx = (label == c)
        #                                                 mean_loc_per_class[c] = loc[idx].mean()

        #                                         mean_conf_per_class = np.full(num_classes, np.nan, dtype=np.float64)

        #                                         for c in range(num_classes):
        #                                                 idx = (label == c)
        #                                                 mean_conf_per_class[c] = conf[idx].mean()

        #                                         x = mean_loc_per_class   
        #                                         y = mean_conf_per_class   

        #                                         pear_r, pear_p = pearsonr(x, y)

        #                                         pearson_mean.append(pear_r)
        #                                         loc_mean.append(loc)


        #                                 pear_r = np.array(pearson_mean).mean()
        #                                 pear_r_std = np.array(pearson_mean).std()

        #                                 loc_mean = np.array(loc_mean).mean()
        #                                 loc_std = np.array(loc_mean).std()
        #                                 print(f'{model} {config_name}: std Pearson r  = {pear_r_std:.4f} (mean {pear_r:.4f})')
        #                                 print(f'{model} {config_name}: std Localization  = {loc_std:.4f} (mean {loc_mean:.4f})')

                        #### Figure 3 Viz

                        # for config, target_layers_selection in config['tl'].items():

                        _c = torch.stack(
                                        [ph._phs['CIFAR100-test'][layer]['peepholes'] for layer in config['tl']['Best c']],
                                        dim=1
                                )
                        
                        _c2 = _c**2
                        loc = _c2.sum(dim=(1,2))/(_c.sum(dim=(1,2)))**2

                        loc_max = 1/(config['numLayers'])
                        loc_min = 1/(config['numLayers']*len(classes))
                        
                        loc = (loc - loc_min)/(loc_max-loc_min)

                        # loc = gini_from_conceptogram(X=_c)
                        conf, _ = sm(ds._dss['CIFAR100-test']['output']).max(dim=1)

                        mean_loc_per_class = np.full(num_classes, np.nan, dtype=np.float64)
                        std_loc_per_class  = np.full(num_classes, np.nan)

                        for c in range(num_classes):
                                idx = (label == c)
                                mean_loc_per_class[c] = loc[idx].mean()
                                std_loc_per_class[c]  = loc[idx].std()

                                # axs[i].text(
                                #         x[c], y[c],
                                #         classes[c],
                                #         fontsize=fontsize-4,
                                #         alpha=0.8
                                # )

                        mean_conf_per_class = np.full(num_classes, np.nan, dtype=np.float64)

                        for c in range(num_classes):
                                idx = (label == c)
                                mean_conf_per_class[c] = conf[idx].mean()

                        x = mean_loc_per_class   
                        y = mean_conf_per_class   
                        print(x)

                        pear_r, _ = pearsonr(x, y)

                        print(f"Pearson r  = {pear_r:.4f}")
                       
                        axs[i].scatter(x, y, alpha=0.7, label=f"{config}")

                        for c in range(num_classes):

                                

                                if (x[c] < config['xlim']) :

                                        axs[i].text(
                                                x[c], y[c],
                                                classes[c],
                                                fontsize=fontsize-4,
                                                alpha=0.8
                                        )

                        m, b = np.polyfit(x, y, 1)
                        xx = np.linspace(x.min(), x.max(), 100)
                        axs[i].plot(xx, m * xx + b)

                        axs[i].set_xlim(0.08, 0.35)

                        axs[i].set_xlabel("Average Localization", fontsize=fontsize)
                        if i == 0 : axs[i].set_ylabel("Average Model Confidence", fontsize=fontsize)

                        axs[i].tick_params(axis='both', which='major', labelsize=fontsize)

                        axs[i].grid(True, alpha=0.3)
                        axs[i].set_title(model, fontsize=fontsize)

        # handles, labels = axs[0].get_legend_handles_labels()

        # fig.legend(
        #         handles,
        #         labels,
        #         loc="lower center",
        #         ncol=len(labels),
        #         fontsize=fontsize,
        #         frameon=True,
        #         bbox_to_anchor=(0.52, -0.01)
        #         )

        fig.tight_layout(rect=[0, 0.09, 1, 1])
        fig.savefig('correlation_conf_loc_zoomed_low.png')

                        