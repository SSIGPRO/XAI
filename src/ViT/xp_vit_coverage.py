import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# python stuff
from time import time
from functools import partial
import random
from matplotlib import pyplot as plt
plt.rc('font', size=10)          
import matplotlib.gridspec as gridspec

# torch stuff
import torch
from cuda_selector import auto_cuda
import torchvision

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
                #'All': target_layers_all,
                'Random': random.sample(target_layers_all, 10),
                'Worst ΔAUC': target_layers_worst_auc,
                'Best ΔAUC': target_layers_best_auc,
                'Worst c': target_layers_worst_c,
                'Best c': target_layers_best_c,            
        }

        # print(f'Target layers: {target_layers}')        

        # model = ModelWrap(
        #         model = nn,
        #         device = device
        #         )
                                                
        # model.update_output(
        #         output_layer = 'heads.head', 
        #         to_n_classes = n_classes,
        #         overwrite = True 
        #         )
                                                
        # model.load_checkpoint(
        #         name = model_name,
        #         path = model_dir,
        #         verbose = verbose
        #         )
                                                
        # model.set_target_modules(
        #         target_modules = target_layers,
        #         verbose = verbose
        #         )

        datasets = ParsedDataset(
                path = ds_path,
                )
        
        classes = Cifar100.get_classes(meta_path = Path(cifar_path)/'cifar-100-python/meta')

    #--------------------------------
    # CoreVectors 
    #--------------------------------
        # corevecs = CoreVectors(
        #         path = cvs_path,
        #         name = cvs_name,
        #         model = model,
        #         )

    #--------------------------------
    # Peepholes
    #--------------------------------

        # cv_parsers = {}
        # feature_sizes = {}
        # for layer in target_layers:

        #         if layer == "heads.head":
        #                 features_cv_dim = 100
        #         else:
        #                 features_cv_dim = 200
        #         cv_parsers[layer] = partial(trim_corevectors,
        #                 module = layer,
        #                 cv_dim = features_cv_dim)
        #         feature_sizes[layer] = features_cv_dim


        # drillers_dict = load_all_drillers(
        #     n_cluster_list = [50,100,200],  
        #     target_layers = target_layers,
        #     drill_path = drill_path,
        #     device = device,
        #     feature_sizes = feature_sizes,
        #     cv_parsers = cv_parsers
        #     )

        # compare_relative_coverage_all_clusters(all_drillers = drillers_dict, threshold=0.8, plot= True, 
        # save_path=plots_path, filename='relative_coverage_rand_proj_vit.png')
        # quit()
        
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

                # cv.load_only(
                #         loaders = loaders,
                #         verbose = verbose 
                #         ) 
                ph.load_only(
                        loaders = loaders,
                        verbose = verbose 
                        )
                sm = torch.nn.Softmax(dim=1)

                samples = torch.randint(high=len(ds._dss['CIFAR100-test']), size=(5,)).tolist()

                for idx in [4063]:

                        n_cols = len(tl_config)

                        fig = plt.figure(figsize=(10,2))
                        gs = gridspec.GridSpec(
                                1, n_cols + 1,
                                width_ratios=[1.2] + [3]*n_cols,  # image smaller than matrices
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
                        ax_img.axis("off")
                        ax_img.set_title(classes[int(ds._dss['CIFAR100-test']['label'][idx])].capitalize())

                        # ── Conceptograms (remaining columns)
                        axs = [fig.add_subplot(gs[0, i+1]) for i in range(n_cols)]

                        for i, (config, tl) in enumerate(tl_config.items()):
                                if config == 'Random':
                                        tl = [l for l in target_layers_all if l in tl]
                                
                                _p = sm(ds._dss['CIFAR100-test']['output'])
                                p = _p[idx]

                                top10 = torch.topk(p, k=20).indices
                                top10_sorted, _ = torch.sort(top10)
                                print(top10_sorted)

                                _conceptograms = torch.stack(
                                        [ph._phs['CIFAR100-test'][layer]['peepholes'] for layer in tl],
                                        dim=1
                                )
                                
                                #_c = _conceptograms[idx]
                                _c = torch.cat((_conceptograms[idx], p.unsqueeze(dim=0)), dim=0)
                                _c_sub = _c[:,top10_sorted]

                                axs[i].imshow(
                                        1 - _c_sub.T,
                                        aspect='auto',
                                        vmin=0.0,
                                        vmax=1.0,
                                        cmap='bone'
                                )

                                # if config == 'Best c':
                                #         _, idx_topk = torch.topk(_c.sum(dim=0), 3, sorted=True)
                                #         classes_topk = [classes[i] for i in idx_topk.tolist()]
                                #         tick_labels = [f'{cls.capitalize()}' for i, cls in enumerate(classes_topk)]
                                #         axs[i].set_yticks(idx_topk, tick_labels)
                                #         axs[i].yaxis.tick_right()
                                # else: axs[i].set_yticks([])

                                # xticks = torch.linspace(0, len(tl)-1, steps=4).long()
                                # axs[i].set_xticks(xticks)
                                axs[i].set_xticks([])
                                axs[i].set_yticks([])
                                axs[i].set_title(config)

                                plt.tight_layout()
                        fig.savefig(f'comparison_{idx}.png', bbox_inches="tight")
                quit()
                # corrs = localization_pmax_correlations(
                #         phs=ph,
                #         ds=ds,
                #         ds_key="CIFAR100-test",
                #         target_modules=target_layers,
                #         save_dir="/home/claranunesbarrancos/repos/XAI/src/temp_plots/localization" ,  
                #         file_name="conf_vs_localization_vit.png"
                #         )

                # print(corrs)
                # quit()
                # deltas = layer_importance(ds=ds, phs=peepholes,
                #         loader = "CIFAR100-test",
                #         target_modules=target_layers, 
                #         )
                # quit()
                # correct = get_filtered_samples(ds=ds,
                # split='CIFAR100-test',
                # #correct=False,
                # conf_range=[0,40],
                # localization_range = [0.06, 0.1],
                # phs = ph,
                # target_modules = target_layers # best config
                # )
                # quit()

                scores, protoclasses = proto_score(
                        datasets = ds,
                        peepholes = ph,
                        proto_key = 'CIFAR100-test',
                        score_name = 'LACS',
                        target_modules = target_layers,
                        verbose = verbose,
                        )

                plot_conceptogram(path = Path.cwd()/'temp_plots/conceptos/vit',
                        name='low_local_not_conf', 
                        datasets=ds,
                        peepholes=ph,
                        loaders=['CIFAR100-test'],
                        target_modules=target_layers,
                        samples=[695],
                        classes = Cifar100.get_classes(meta_path = Path(cifar_path)/'cifar-100-python/meta'),
                        scores=scores,
                )
                quit()

                # out =localization_from_peepholes(phs=ph, ds=ds, ds_key="CIFAR100-test", target_modules=target_layers, plot = True,
                # save_dir = plots_path)
                # results = ds._dss["CIFAR100-test"]["result"]

                # means = localization_means(Ls=out["Ls"], results=results)
                # print(means)

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
                # coverage = empp_coverage_scores(drillers=drillers, threshold=0.95, plot=False, save_path='/home/claranunesbarrancos/repos/XAI/src/clustering_xp/temp_plots', file_name='coverage_vgg_550clusters.png')
               # empp_relative_coverage_scores(drillers=ph._drillers, threshold=0.8, plot=True, save_path='/home/claranunesbarrancos/repos/XAI/src/clustering_xp/temp_plots', file_name='relative_cluster_coverage_vgg_550clusters.png')
        
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