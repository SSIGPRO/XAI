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

        samples_mobile = {
                'HCLL': 899,
                'HCHL': 131,
                'LCLL': 127,
                'LCHL': 0
        }

        models['Mobile'] = {
                'phs': phs_path_mobile,
                'ds': ds_path_mobile,
                'tl': target_layers_mobile,
                'samples': samples_mobile
                }

        ### ViT

        ds_path_vit = Path('/srv/newpenny/XAI/generated_data/TPAMI/parsed_datasets/CIFAR100_ViT')

        phs_path_vit =  Path('/srv/newpenny/XAI/CN/vit_data/peepholes_all/peepholes_100')

        target_layers_vit = [
                        'encoder.layers.encoder_layer_7.mlp.0', 'encoder.layers.encoder_layer_8.mlp.0', 'encoder.layers.encoder_layer_8.mlp.3',
                        'encoder.layers.encoder_layer_9.mlp.0', 'encoder.layers.encoder_layer_9.mlp.3', 'encoder.layers.encoder_layer_10.mlp.0',
                        'encoder.layers.encoder_layer_10.mlp.3', 'encoder.layers.encoder_layer_11.mlp.0', 'encoder.layers.encoder_layer_11.mlp.3', 'heads.head'
                ]

        samples_vit = {
                'HCLL': 871,
                'HCHL': 19,
                'LCLL': 695,
                'LCHL': 2568
        }

        models['ViT'] = {
                'phs': phs_path_vit,
                'ds': ds_path_vit,
                'tl': target_layers_vit,
                'samples': samples_vit
                }

        ### VGG

        ds_path_vgg = '/srv/newpenny/XAI/generated_data/TPAMI/parsed_datasets/CIFAR100_VGG16'

        phs_path_vgg = Path('/srv/newpenny/XAI/CN/vgg_data/peepholes_all/peepholes_100')

        target_layers_vgg = ['features.26','features.28','classifier.0','classifier.3', 'classifier.6']

        samples_vgg =  {
                'HCLL': 41,
                'HCHL': 312,
                'LCLL': 33,
                'LCHL': 7405
        }

        models['vgg'] = {
                'phs': phs_path_vgg,
                'ds': ds_path_vgg,
                'tl': target_layers_vgg,
                'samples': samples_vgg
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

                        _r = ds._dss['CIFAR100-test']['result']

                        fig = plt.figure(figsize=(12,8))

                        gs = gridspec.GridSpec(
                                2, 4,
                                height_ratios=[1.2, 5],
                                width_ratios=[2, 2, 2, 2],
                                hspace=0.15,
                                wspace=0.2
                                )

                        axs_img = [fig.add_subplot(gs[0, j]) for j in range(4)]
                        
                        axs_mat = [fig.add_subplot(gs[1, j]) for j in range(4)]

                        for i, (corner, idx) in enumerate(config['samples'].items()):

                                img = ds._dss['CIFAR100-test']['image'][idx]
                                img = (img * std + mean).clamp(0, 1)

                                axs_img[i].imshow(img.cpu().permute(1,2,0))
                                # axs_img[i].axis("off")
                                axs_img[i].set_title(corner)
                                axs_img[i].set_frame_on(True)

                                color = "green" if _r[idx] == 1 else "red"
                                for spine in axs_img[i].spines.values():
                                        spine.set_edgecolor(color)
                                        spine.set_linewidth(5)

                                _c = _conceptograms[idx]

                                axs_mat[i].imshow(
                                                1 - _c.T,
                                                aspect="auto",
                                                vmin=0.0,
                                                vmax=1.0,
                                                cmap="bone"
                                        )

                                xticks = torch.linspace(0, len(config['tl'])-1, steps=4).long()
                                axs_mat[i].set_xticks(xticks)
                                axs_mat[i].set_yticks([])
                        
                        fig.savefig('prova.png')
                        quit()




                        


                for idx in samples:

                        n_cols = len(tl_config)

                        

                        # ── Denormalize image
                        img = ds._dss['CIFAR100-test']['image'][idx]

                        mean = torch.tensor([0.438, 0.418, 0.377]).view(3,1,1)
                        std = torch.tensor([0.300, 0.287, 0.294]).view(3,1,1)

                        img = (img * std + mean).clamp(0, 1)

                        # ── Image subplot (first column)
                        ax_img = fig.add_subplot(gs[0, 0])
                        ax_img.imshow(img.cpu().permute(1,2,0))
                        ax_img.axis("off")
                        ax_img.set_title(classes[int(ds._dss['CIFAR100-test']['label'][idx])].capitalize())

                        # ── Conceptograms (remaining columns)
                        axs = [fig.add_subplot(gs[0, i+1]) for i in range(n_cols)]

                        for i, (config, tl) in enumerate(tl_config.items()):
                                if config == 'Random':
                                        tl = [l for l in target_layers_all if l in tl]

                                _conceptograms = torch.stack(
                                        [ph._phs['CIFAR100-test'][layer]['peepholes'] for layer in tl],
                                        dim=1
                                )
                                _c = _conceptograms[idx]

                                axs[i].imshow(
                                        1 - _c.T,
                                        aspect='auto',
                                        vmin=0.0,
                                        vmax=1.0,
                                        cmap='bone'
                                )

                                xticks = torch.linspace(0, len(tl)-1, steps=4).long()
                                axs[i].set_xticks(xticks)
                                axs[i].set_yticks([])
                                axs[i].set_title(config)

                                plt.tight_layout()
                        fig.savefig(f'comparison_{idx}.png', bbox_inches="tight")

                        # n_cols = len(tl_config)

                        # fig = plt.figure(figsize=(8,5))
                        # gs = gridspec.GridSpec(
                        #         2, n_cols,
                        #         height_ratios=[1, 4],   # top image smaller than plots
                        #         hspace=0.2
                        #         )

                        # img = ds._dss['CIFAR100-test']['image'][idx]  # (3, H, W)

                        # mean = torch.tensor([0.438, 0.418, 0.377]).view(3,1,1)
                        # std = torch.tensor([0.300, 0.287, 0.294]).view(3,1,1)

                        # img_denorm = img * std + mean
                        # img = img_denorm.clamp(0, 1)
                        # print(img.shape)

                        # # ── Top image (spans all columns)
                        # ax_top = fig.add_subplot(gs[0, :])
                        # ax_top.imshow(img.detach().cpu().numpy().transpose(1,2,0))   
                        # ax_top.axis("off")
                        # ax_top.set_title(classes[int(ds._dss['CIFAR100-test']['label'][idx])].capitalize())

                        # # ── Bottom plots
                        # axs = [fig.add_subplot(gs[1, i]) for i in range(n_cols)]

                        # for i, (config, tl) in enumerate(tl_config.items()):
                        #         if config == 'Random':
                        #                 tl = [l for l in target_layers_all if l in tl]

                        #         _conceptograms = torch.stack(
                        #                         [ph._phs['CIFAR100-test'][layer]['peepholes'] for layer in tl],
                        #                         dim=1
                        #                 )
                        #         _c = _conceptograms[idx]

                        #         axs[i].imshow(
                        #                 1 - _c.T,
                        #                 aspect='auto',
                        #                 vmin=0.0,
                        #                 vmax=1.0,
                        #                 cmap='bone'
                        #         )

                        #         xticks = torch.linspace(0, len(tl)-1, steps=4).long()
                        #         axs[i].set_yticks([])
                        #         axs[i].set_xticks(xticks)
                        #         axs[i].set_title(config)
                        # plt.tight_layout()

                        # fig.savefig(f'comparison_{idx}.png', bbox_inches="tight")

                quit()

                # corrs = localization_pmax_correlations(
                #         phs=ph,
                #         ds=ds,
                #         ds_key="CIFAR100-test",
                #         target_modules=target_layers,
                #         save_dir="/home/claranunesbarrancos/repos/XAI/src/temp_plots/localization" ,  
                #         file_name="conf_vs_localization_mobilenet.png"
                #         )

                # print(corrs)
                # quit()

                
                # drillers = {}
                # for peep_layer in target_layers:
                #         drillers[peep_layer] = tGMM(
                #                 path=drill_path,
                #                 name=f"classifier.{peep_layer}",  
                #                 label_key = 'label',
                #                 nl_classifier=100,
                #                 nl_model=n_classes,
                #                 n_features=feature_sizes[peep_layer],
                #                 parser=cv_parsers[peep_layer],
                #                 device=device
                #         )

                # for drill_key, driller in drillers.items():
                #         if driller._empp_file.exists():
                #                 print(f'Loading Classifier for {drill_key}')
                #                 driller.load()
                #         else:
                #                 print(f'No Classifier found for {drill_key} at {driller._empp_file}')
                # coverage = empp_coverage_scores(drillers=drillers, threshold=0.8, plot=False)
                # for drill_key, driller in drillers.items():
                       
                #         if drill_key == 'features.6.conv.1.0' or drill_key == 'features.8.conv.2' or drill_key == 'classifier.1':

                #                 plt.imshow(1-driller._empp.detach().cpu().numpy(), cmap='bone')
                #                 plt.title(f'c={coverage[drill_key]}')
                #                 plt.xticks([]) 
                #                 plt.yticks([]) 
                #                 # plt.tight_layout()
                #                 plt.savefig(f'Ep_{drill_key}.png', dpi=300, bbox_inches='tight')

                # quit()
                

                # correct = get_filtered_samples(ds=ds,
                #         split='CIFAR100-test',
                #         #correct=False,
                #         conf_range=[0,30],
                #         localization_range = [0.05, 0.06],
                #         phs = ph,
                #         target_modules = target_layers # best config
                #         )
                # quit()
                # scores, protoclasses = proto_score(
                #         datasets = ds,
                #         peepholes = ph,
                #         proto_key = 'CIFAR100-test',
                #         score_name = 'LACS',
                #         target_modules = target_layers,
                #         verbose = verbose,
                #         )

                # plot_conceptogram(path = Path.cwd()/'temp_plots/conceptos/mobilenet',
                #         name='low_local_not_conf', 
                #         datasets=ds,
                #         peepholes=ph,
                #         loaders=['CIFAR100-test'],
                #         target_modules=target_layers,
                #         samples=[1674],
                #         classes =Cifar100.get_classes(meta_path = Path(cifar_path)/'cifar-100-python/meta'),
                #         scores=scores,
                #         )
                # quit()

                # avg_scores = {}

                # for ds_key in scores:
                #         avg_scores[ds_key] = scores[ds_key]['LACS'].mean()
                # print(avg_scores)

                # quit()
                # out =localization_from_peepholes(phs=ph, ds=ds, ds_key="CIFAR100-test", target_modules=target_layers, plot = True,
                # save_dir = plots_path)
                # results = ds._dss["CIFAR100-test"]["result"]

                # means = localization_means(Ls=out["Ls"], results=results)
                # print(means)
                #plot_empp_posteriors(drillers=drillers, save_dir=drill_path)
                #coverage = empp_coverage_scores(drillers=drillers, threshold=0.9, plot=False, save_path='/home/claranunesbarrancos/repos/XAI/src/clustering_xp/temp_plots', file_name='coverage_mobilenet_06.png')
                #empp_relative_coverage_scores(drillers=ph._drillers, threshold=0.8, plot=True, save_path='/home/claranunesbarrancos/repos/XAI/src/clustering_xp/temp_plots', file_name='relative_cluster_coverage_vgg_550clusters.png')
                # compare_relative_coverage_all_clusters( all_drillers = drillers_dict,
                #         threshold=0.8, plot= True, save_path=plots_path, filename='relative_coverage_all_clusters_mobilenet.png')

                # localization_runs = []
                # localization_metric_runs = []

                # for i in range(20):
                #         random_layers = random.sample(target_layers, 10)

                #         out = localization_from_peepholes(
                #                 phs=ph,
                #                 ds=ds,
                #                 ds_key="CIFAR100-test",
                #                 target_modules=random_layers,
                #                 plot=False,
                #                 verbose=False,
                #         )

                #         results = ds._dss["CIFAR100-test"]["result"]
                #         means = localization_means(Ls=out["Ls"], results=results)
                #         localization_runs.append(means)

                #         localization_metric_runs.append({
                #                 "auc": out["auc"],
                #                 "fpr95": out["fpr95"],
                #                 "threshold_tpr95": out["threshold_tpr95"],
                #                 "L_avg": out["L_avg"],
                #         })

                

                # # --- aggregate localization means (exclude counts from averaging) ---
                # avg_localization = {}
                # for key in localization_runs[0]:
                #         if key.startswith("n_"):
                #                 continue
                #         xs = torch.stack([torch.as_tensor(run[key]).float() for run in localization_runs])
                #         avg_localization[key] = torch.nanmean(xs)

                # # keep counts from first run (they should be identical across runs)
                # for k in ["n_all", "n_correct", "n_incorrect"]:
                #         avg_localization[k] = localization_runs[0][k]

                # # --- aggregate auc/fpr95 metrics ---
                # avg_loc_metrics = {}
                # for key in localization_metric_runs[0]:
                #         xs = torch.stack([torch.as_tensor(run[key]).float() for run in localization_metric_runs])
                #         avg_loc_metrics[key] = torch.nanmean(xs)


                # print("\nAverage localization means over random layers:")
                # print(avg_localization)

                # print("\nAverage localization AUC/FPR95 over random layers:")
                # print(avg_loc_metrics)
