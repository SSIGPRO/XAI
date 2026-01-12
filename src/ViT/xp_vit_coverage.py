import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# python stuff
from time import time
from functools import partial
import random

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
from calculate_layer_importance import layer_importance_lolo_deltas_per_loader_okko as layer_importance, topk_layers_per_loader 
from peepholelib.utils.localization import *


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
        ds_path = Path.cwd()/'../data/datasets'

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
        # target_layers = list(dict.fromkeys(get_st_list(nn.state_dict().keys())))
        # #best
        # target_layers = ['encoder.layers.encoder_layer_7.mlp.0', 'encoder.layers.encoder_layer_8.mlp.0', 'encoder.layers.encoder_layer_8.mlp.3',
        # 'encoder.layers.encoder_layer_9.mlp.0', 'encoder.layers.encoder_layer_9.mlp.3', 'encoder.layers.encoder_layer_10.mlp.0',
        # 'encoder.layers.encoder_layer_10.mlp.3', 'encoder.layers.encoder_layer_11.mlp.0', 'encoder.layers.encoder_layer_11.mlp.3', 'heads.head']

        # # #worst
        # target_layers = ['encoder.layers.encoder_layer_0.mlp.3','encoder.layers.encoder_layer_1.mlp.0', 'encoder.layers.encoder_layer_1.mlp.3','encoder.layers.encoder_layer_2.mlp.0',
        # 'encoder.layers.encoder_layer_2.mlp.3','encoder.layers.encoder_layer_3.mlp.0','encoder.layers.encoder_layer_3.mlp.3', 
        # 'encoder.layers.encoder_layer_4.mlp.0','encoder.layers.encoder_layer_4.mlp.3','encoder.layers.encoder_layer_6.mlp.3']

        # 10 best auc
        #target_layers = ['encoder.layers.encoder_layer_3.mlp.3', 'encoder.layers.encoder_layer_2.mlp.3', 'encoder.layers.encoder_layer_1.mlp.3',
        #'encoder.layers.encoder_layer_0.mlp.0', 'encoder.layers.encoder_layer_0.mlp.3',
        # 'encoder.layers.encoder_layer_10.mlp.0',
        # 'encoder.layers.encoder_layer_10.mlp.3', 'encoder.layers.encoder_layer_11.mlp.0', 'encoder.layers.encoder_layer_11.mlp.3', 'heads.head']

        #best frp95
        # target_layers = ['encoder.layers.encoder_layer_2.mlp.3', 'encoder.layers.encoder_layer_5.mlp.0', 'encoder.layers.encoder_layer_1.mlp.3',
        # 'encoder.layers.encoder_layer_8.mlp.3', 'encoder.layers.encoder_layer_9.mlp.3','encoder.layers.encoder_layer_10.mlp.0',
        # 'encoder.layers.encoder_layer_10.mlp.3', 'encoder.layers.encoder_layer_11.mlp.0', 'encoder.layers.encoder_layer_11.mlp.3', 'heads.head']

        target_layers = ['heads.head',
        'encoder.layers.encoder_layer_11.mlp.0',
        'encoder.layers.encoder_layer_11.mlp.3',
        'encoder.layers.encoder_layer_10.mlp.3',
        'encoder.layers.encoder_layer_10.mlp.0',
        'encoder.layers.encoder_layer_9.mlp.0',
        'encoder.layers.encoder_layer_9.mlp.3',
        'encoder.layers.encoder_layer_8.mlp.3',
        'encoder.layers.encoder_layer_8.mlp.0',
        'encoder.layers.encoder_layer_7.mlp.0'
        ]



        print(f'Target layers: {target_layers}')        

        model = ModelWrap(
                model = nn,
                device = device
                )
                                                
        model.update_output(
                output_layer = 'heads.head', 
                to_n_classes = n_classes,
                overwrite = True 
                )
                                                
        model.load_checkpoint(
                name = model_name,
                path = model_dir,
                verbose = verbose
                )
                                                
        model.set_target_modules(
                target_modules = target_layers,
                verbose = verbose
                )

        datasets = ParsedDataset(
                path = ds_path,
                )

    #--------------------------------
    # CoreVectors 
    #--------------------------------
        corevecs = CoreVectors(
                path = cvs_path,
                name = cvs_name,
                model = model,
                )

    #--------------------------------
    # Peepholes
    #--------------------------------

        cv_parsers = {}
        feature_sizes = {}
        for layer in target_layers:

                if layer == "heads.head":
                        features_cv_dim = 100
                else:
                        features_cv_dim = 200
                cv_parsers[layer] = partial(trim_corevectors,
                        module = layer,
                        cv_dim = features_cv_dim)
                feature_sizes[layer] = features_cv_dim


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
       
        with datasets as ds, corevecs as cv, peepholes as ph:
                ds.load_only(
                        loaders = loaders,
                        verbose = verbose
                        )

                cv.load_only(
                        loaders = loaders,
                        verbose = verbose 
                        ) 
                ph.load_only(
                        loaders = loaders,
                        verbose = verbose 
                        )
                corrs = localization_pmax_correlations(
                        phs=ph,
                        ds=ds,
                        ds_key="CIFAR100-test",
                        target_modules=target_layers,
                        save_dir="/home/claranunesbarrancos/repos/XAI/src/temp_plots/localization" ,  
                        file_name="conf_vs_localization_vit.png"
                        )

                print(corrs)
                quit()
                # deltas = layer_importance(score_fn=proto_score,
                #         datasets=ds, peepholes=peepholes,
                #         target_modules=target_layers, loaders=loaders,
                #         score_name="LACS", proto_key="CIFAR100-train",
                #         batch_size=bs, verbose=True,
                #         )
                # topk = topk_layers_per_loader(deltas, k=5,
                #         mode="fpr95",     # or "fpr95" or "joint"
                #         )
                # quit()

                scores, protoclasses = proto_score(
                        datasets = ds,
                        peepholes = ph,
                        proto_key = 'CIFAR100-test',
                        score_name = 'LACS',
                        target_modules = target_layers,
                        verbose = verbose,
                        )

                avg_scores = {}

                for ds_key in scores:
                        avg_scores[ds_key] = scores[ds_key]['LACS'].mean()
                print(avg_scores)

                out =localization_from_peepholes(phs=ph, ds=ds, ds_key="CIFAR100-test", target_modules=target_layers, plot = True,
                save_dir = plots_path)
                results = ds._dss["CIFAR100-test"]["result"]

                means = localization_means(Ls=out["Ls"], results=results)
                print(means)

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
        
                # proto_scores_runs = []
                # localization_runs = []
                # localization_metric_runs = []

                # for i in range(20):
                #         random_layers = random.sample(target_layers, 10)

                #         scores, protoclasses = proto_score(
                #                 datasets=ds,
                #                 peepholes=ph,
                #                 proto_key='CIFAR100-test',
                #                 score_name='LACS',
                #                 target_modules=random_layers,
                #                 verbose=verbose,
                #         )

                #         avg_scores = {}
                #         for ds_key in scores:
                #                 avg_scores[ds_key] = scores[ds_key]['LACS'].mean()
                #         proto_scores_runs.append(avg_scores)

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

                # # --- aggregate protoscore ---
                # avg_proto_scores = {}
                # for key in proto_scores_runs[0]:
                #         xs = torch.stack([torch.as_tensor(run[key]).float() for run in proto_scores_runs])
                #         avg_proto_scores[key] = xs.mean()

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

                # print("Average ProtoScores over random layers:")
                # print(avg_proto_scores)

                # print("\nAverage localization means over random layers:")
                # print(avg_localization)

                # print("\nAverage localization AUC/FPR95 over random layers:")
                # print(avg_loc_metrics)