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
import matplotlib as mpl

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
# from peepholelib.utils.viz_corevecs import plot_tsne, plot_tsne_CUDA
# from peepholelib.utils.localization import *
# from peepholelib.utils.get_samples import *
# from peepholelib.scores.protoclass import conceptogram_protoclass_score as proto_score
# from peepholelib.plots.conceptograms import plot_conceptogram 

# mpl.use("pgf")
# mpl.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     "pgf.preamble": r"\usepackage{mathpazo}",
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

plt.rc('font', size=10)          # default text


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
                label_key = 'label',
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
        
        target_layers = [ 
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

        #worst ones
        # target_layers = ['features.1.conv.0.0','features.4.conv.1.0','features.4.conv.2', 'features.7.conv.2','features.8.conv.0.0',
        #         'features.8.conv.2','features.9.conv.0.0', 'features.11.conv.1.0',
        #         'features.12.conv.1.0','features.13.conv.2' ]

        # best auc
        # target_layers = ['features.1.conv.1','features.2.conv.0.0', 'features.3.conv.1.0', 'features.5.conv.1.0', 'features.6.conv.1.0','features.8.conv.1.0',
        # 'features.17.conv.1.0', 'features.17.conv.2', 'features.18.0', 'classifier.1']

        # worst auc
        # target_layers = ['features.11.conv.1.0', 'features.11.conv.2', 'features.14.conv.1.0', 'features.14.conv.2', 'features.15.conv.0.0',  'features.15.conv.1.0',
        # 'features.15.conv.2', 'features.16.conv.0.0', 'features.16.conv.1.0', 'features.16.conv.2']

        # #best fr95
        # target_layers=[
        #         'features.14.conv.2', 'features.17.conv.0.0', 'features.17.conv.2', 'features.15.conv.2', 'features.11.conv.2','features.17.conv.1.0', 
        #         'features.14.conv.1.0','features.15.conv.0.0','features.18.0', 'classifier.1'
        # ]
        #best coverage (threshold =0.7-0.89)
        target_layers = ['features.2.conv.0.0','features.3.conv.2','features.5.conv.1.0','features.6.conv.1.0','features.8.conv.1.0',
        'features.9.conv.1.0','features.17.conv.1.0','features.17.conv.2','features.18.0','classifier.1']

        #best coverage (threshold =0.95)
        # target_layers = ['features.2.conv.0.0','features.3.conv.0.0','features.3.conv.1.0','features.3.conv.2','features.5.conv.1.0',
        # 'features.6.conv.1.0','features.8.conv.1.0','features.9.conv.1.0','features.17.conv.2','classifier.1']

        loaders = [
        'CIFAR100-train',
        'CIFAR100-val',
        'CIFAR100-test',
         ]

    #--------------------------------
    # Model 
    #--------------------------------
    
        nn = torchvision.models.mobilenet_v2(pretrained=True)

        n_classes = len(Cifar100.get_classes(meta_path = Path(cifar_path)/'cifar-100-python/meta')) 

        model = ModelWrap(
                model = nn,
                device = device
                )
                                                
        model.update_output(
                output_layer = 'classifier.1', 
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

                if layer == "classifier.1":
                        features_cv_dim = 100
                else:
                        features_cv_dim = 300
                cv_parsers[layer] = partial(trim_corevectors,
                        module = layer,
                        cv_dim = features_cv_dim)
                feature_sizes[layer] = features_cv_dim

        # drillers_dict = load_all_drillers(
        #     n_cluster_list = [10, 50, 100, 150, 200, 250, 300, 400, 600],  
        #     target_layers = target_layers,
        #     drill_path = drill_path,
        #     device = device,
        #     feature_sizes = feature_sizes,
        #     cv_parsers = cv_parsers
        #     )

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
                # layer = 'features.9.conv.1.0'
                # X = cv._corevds['CIFAR100-train'][layer]
                # X_reduced = X[:, :10]
                        
                # X_np = X_reduced.cpu().numpy()

                # plot_tsne_CUDA(corevector = cv,
                #         ds = ds,
                #         save_path = Path('/home/claranunesbarrancos/repos/XAI/src/temp_plots/corevectors'),
                #         layer = layer,
                #         file_name = "features6conv10_mobilenet_supersuperclass_tsne",
                #         n_classes = 10,
                #         )

                #quit()
                        
                ph.load_only(
                        loaders = loaders,
                        verbose = verbose 
                        )

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

                
                drillers = {}
                for peep_layer in target_layers:
                        drillers[peep_layer] = tGMM(
                                path=drill_path,
                                name=f"classifier.{peep_layer}",  
                                label_key = 'label',
                                nl_classifier=100,
                                nl_model=n_classes,
                                n_features=feature_sizes[peep_layer],
                                parser=cv_parsers[peep_layer],
                                device=device
                        )

                for drill_key, driller in drillers.items():
                        if driller._empp_file.exists():
                                print(f'Loading Classifier for {drill_key}')
                                driller.load()
                        else:
                                print(f'No Classifier found for {drill_key} at {driller._empp_file}')
                coverage = empp_coverage_scores(drillers=drillers, threshold=0.8, plot=False)
                for drill_key, driller in drillers.items():
                       
                        if drill_key == 'features.6.conv.1.0' or drill_key == 'features.8.conv.2' or drill_key == 'classifier.1':

                                plt.imshow(1-driller._empp.detach().cpu().numpy(), cmap='bone')
                                plt.title(f'c={coverage[drill_key]}')
                                plt.xticks([]) 
                                plt.yticks([]) 
                                # plt.tight_layout()
                                plt.savefig(f'Ep_{drill_key}.png', dpi=300, bbox_inches='tight')

                quit()
                

                correct = get_filtered_samples(ds=ds,
                        split='CIFAR100-test',
                        #correct=False,
                        conf_range=[0,30],
                        localization_range = [0.05, 0.06],
                        phs = ph,
                        target_modules = target_layers # best config
                        )
                quit()
                scores, protoclasses = proto_score(
                        datasets = ds,
                        peepholes = ph,
                        proto_key = 'CIFAR100-test',
                        score_name = 'LACS',
                        target_modules = target_layers,
                        verbose = verbose,
                        )

                plot_conceptogram(path = Path.cwd()/'temp_plots/conceptos/mobilenet',
                        name='low_local_not_conf', 
                        datasets=ds,
                        peepholes=ph,
                        loaders=['CIFAR100-test'],
                        target_modules=target_layers,
                        samples=[1674],
                        classes =Cifar100.get_classes(meta_path = Path(cifar_path)/'cifar-100-python/meta'),
                        scores=scores,
                        )
                quit()

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
