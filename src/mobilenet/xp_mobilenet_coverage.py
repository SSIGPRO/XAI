import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial

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
from peepholelib.datasets.functional.transforms import vgg16_cifar100 as ds_transform 
from peepholelib.datasets.functional.samplers import random_subsampling 

# corevecs
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import linear_svd_projection, conv2d_toeplitz_svd_projection

# peepholes
from peepholelib.peepholes.parsers import trim_corevectors
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.models.viz import viz_singular_values_2
from peepholelib.utils.viz_empp import *
from peepholelib.utils.viz_tsne import plot_tsne

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
        #device = torch.device('cuda:1') 
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

        model_dir = '/srv/newpenny/XAI/models'
        model_name = 'CN_model=mobilenet_v2_dataset=CIFAR100_optim=Adam_scheduler=RoP_lr=0.001_factor=0.1_patience=5.pth'
        
        svds_path = '/srv/newpenny/XAI/CN/data'
        svds_name = 'svds' 
        
        cvs_path = Path.cwd()/'/srv/newpenny/XAI/CN/data/corevectors'
        cvs_name = 'corevectors'

        drill_path = Path.cwd()/'/srv/newpenny/XAI/CN/data/drillers_all'
        drill_name = 'classifier'

        phs_path = Path.cwd()/'/srv/newpenny/XAI/CN/data/peepholes'
        phs_name = 'peepholes'

        plots_path = Path.cwd()/'temp_plots'
        
        verbose = True 
        
        target_layers = [ 'features.1.conv.0.0', 'features.1.conv.1','features.2.conv.0.0','features.2.conv.1.0','features.2.conv.2',
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

        
        loaders = ['train', 'val', 'test']

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

        # datasets = ParsedDataset(
        #         path = ds_path,
        #         )

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
        features_cv_dim = 300

        cv_parsers = {
                'features.1.conv.0.0': partial(trim_corevectors,
                        module = 'features.1.conv.0.0',
                        cv_dim = features_cv_dim),
                'features.1.conv.1': partial(trim_corevectors,
                        module = 'features.1.conv.1',
                        cv_dim = features_cv_dim),
                'features.2.conv.0.0': partial(trim_corevectors,
                        module = 'features.2.conv.0.0',
                        cv_dim = features_cv_dim),
                'features.2.conv.1.0': partial(trim_corevectors,
                        module = 'features.2.conv.1.0',
                        cv_dim = features_cv_dim),
                'features.2.conv.2': partial(trim_corevectors,
                        module = 'features.2.conv.2',
                        cv_dim = features_cv_dim),
                'features.3.conv.0.0': partial(trim_corevectors,
                        module = 'features.3.conv.0.0', 
                        cv_dim = features_cv_dim ),
                'features.3.conv.1.0': partial(trim_corevectors,
                        module = 'features.3.conv.1.0',
                        cv_dim = features_cv_dim),
                'features.3.conv.2': partial(trim_corevectors,
                        module = 'features.3.conv.2',
                        cv_dim = features_cv_dim),
                'features.4.conv.0.0': partial(trim_corevectors,
                        module = 'features.4.conv.0.0',
                        cv_dim = features_cv_dim),
                'features.4.conv.1.0': partial(trim_corevectors,
                        module = 'features.4.conv.1.0',
                        cv_dim = features_cv_dim),
                'features.4.conv.2': partial(trim_corevectors,
                        module = 'features.4.conv.2',
                        cv_dim = features_cv_dim),
                'features.5.conv.0.0': partial(trim_corevectors,
                        module = 'features.5.conv.0.0', 
                        cv_dim = features_cv_dim),
                'features.5.conv.1.0': partial(trim_corevectors,
                        module = 'features.5.conv.1.0',
                        cv_dim = features_cv_dim),
                'features.5.conv.2': partial(trim_corevectors,
                        module = 'features.5.conv.2',
                        cv_dim = features_cv_dim),      
                'features.6.conv.0.0': partial(trim_corevectors,
                        module = 'features.6.conv.0.0',
                        cv_dim = features_cv_dim),
                'features.6.conv.1.0': partial(trim_corevectors,
                        module = 'features.6.conv.1.0',     
                        cv_dim = features_cv_dim),
                'features.6.conv.2': partial(trim_corevectors,
                        module = 'features.6.conv.2',
                        cv_dim = features_cv_dim),
                'features.7.conv.0.0': partial(trim_corevectors,
                        module = 'features.7.conv.0.0',
                        cv_dim = features_cv_dim),
                'features.7.conv.1.0': partial(trim_corevectors,
                        module = 'features.7.conv.1.0',
                        cv_dim = features_cv_dim),
                'features.7.conv.2': partial(trim_corevectors,
                        module = 'features.7.conv.2',
                        cv_dim = features_cv_dim),
                'features.8.conv.0.0': partial(trim_corevectors,
                        module = 'features.8.conv.0.0',
                        cv_dim = features_cv_dim),
                'features.8.conv.1.0': partial(trim_corevectors,
                        module = 'features.8.conv.1.0',
                        cv_dim = features_cv_dim),
                'features.8.conv.2': partial(trim_corevectors,
                        module = 'features.8.conv.2',
                        cv_dim = features_cv_dim),
                'features.9.conv.0.0': partial(trim_corevectors,
                        module = 'features.9.conv.0.0',
                        cv_dim = features_cv_dim),
                'features.9.conv.1.0': partial(trim_corevectors,
                        module = 'features.9.conv.1.0',
                        cv_dim = features_cv_dim),
                'features.9.conv.2': partial(trim_corevectors,
                        module = 'features.9.conv.2',
                        cv_dim = features_cv_dim),
                'features.10.conv.0.0': partial(trim_corevectors,
                        module = 'features.10.conv.0.0',
                        cv_dim = features_cv_dim),
                'features.10.conv.1.0': partial(trim_corevectors,
                        module = 'features.10.conv.1.0',
                        cv_dim = features_cv_dim),
                'features.10.conv.2': partial(trim_corevectors,
                        module = 'features.10.conv.2',
                        cv_dim = features_cv_dim),
                'features.11.conv.0.0': partial(trim_corevectors,
                        module = 'features.11.conv.0.0',
                        cv_dim = features_cv_dim),
                'features.11.conv.1.0': partial(trim_corevectors,
                        module = 'features.11.conv.1.0',
                        cv_dim = features_cv_dim),
                'features.11.conv.2': partial(trim_corevectors,
                        module = 'features.11.conv.2',
                        cv_dim = features_cv_dim),
                'features.12.conv.0.0': partial(trim_corevectors,
                        module = 'features.12.conv.0.0',
                        cv_dim = features_cv_dim),
                'features.12.conv.1.0': partial(trim_corevectors,
                        module = 'features.12.conv.1.0',            
                        cv_dim = features_cv_dim),
                'features.12.conv.2': partial(trim_corevectors,
                        module = 'features.12.conv.2',
                        cv_dim = features_cv_dim),
                'features.13.conv.0.0': partial(trim_corevectors,
                        module = 'features.13.conv.0.0',
                        cv_dim = features_cv_dim),
                'features.13.conv.1.0': partial(trim_corevectors,
                        module = 'features.13.conv.1.0',
                        cv_dim = features_cv_dim),
                'features.13.conv.2': partial(trim_corevectors,
                        module = 'features.13.conv.2',
                        cv_dim = features_cv_dim),
                'features.14.conv.0.0': partial(trim_corevectors,
                        module = 'features.14.conv.0.0',
                        cv_dim = features_cv_dim),
                'features.14.conv.1.0': partial(trim_corevectors,
                        module = 'features.14.conv.1.0',
                        cv_dim = features_cv_dim),
                'features.14.conv.2': partial(trim_corevectors,
                        module = 'features.14.conv.2',
                        cv_dim = features_cv_dim),
                'features.15.conv.0.0': partial(trim_corevectors,
                        module = 'features.15.conv.0.0',
                        cv_dim = features_cv_dim),
                'features.15.conv.1.0': partial(trim_corevectors,
                        module = 'features.15.conv.1.0',
                        cv_dim = features_cv_dim),
                'features.15.conv.2': partial(trim_corevectors,
                        module = 'features.15.conv.2',
                        cv_dim = features_cv_dim),
                'features.16.conv.0.0': partial(trim_corevectors,
                        module = 'features.16.conv.0.0',
                        cv_dim = features_cv_dim),
                'features.16.conv.1.0': partial(trim_corevectors,
                        module = 'features.16.conv.1.0',
                        cv_dim = features_cv_dim),
                'features.16.conv.2': partial(trim_corevectors,
                        module = 'features.16.conv.2',
                        cv_dim = features_cv_dim),
                'features.17.conv.0.0': partial(trim_corevectors,
                        module = 'features.17.conv.0.0',
                        cv_dim = features_cv_dim),
                'features.17.conv.1.0': partial(trim_corevectors,
                        module = 'features.17.conv.1.0',
                        cv_dim = features_cv_dim),
                'features.17.conv.2': partial(trim_corevectors,
                        module = 'features.17.conv.2',
                        cv_dim = features_cv_dim),
                'features.18.0': partial(trim_corevectors,
                        module = 'features.18.0',
                        cv_dim = features_cv_dim),
                'classifier.1': partial(trim_corevectors,
                        module = 'classifier.1',
                        cv_dim = features_cv_dim),
                }

        feature_sizes = {
                'features.1.conv.0.0': features_cv_dim, 'features.1.conv.1': features_cv_dim,
                'features.2.conv.0.0': features_cv_dim, 'features.2.conv.1.0': features_cv_dim, 'features.2.conv.2': features_cv_dim,
                'features.3.conv.0.0': features_cv_dim, 'features.3.conv.1.0': features_cv_dim, 'features.3.conv.2': features_cv_dim,
                'features.4.conv.0.0': features_cv_dim, 'features.4.conv.1.0': features_cv_dim, 'features.4.conv.2': features_cv_dim,
                'features.5.conv.0.0': features_cv_dim, 'features.5.conv.1.0': features_cv_dim, 'features.5.conv.2': features_cv_dim,      
                'features.6.conv.0.0': features_cv_dim, 'features.6.conv.1.0': features_cv_dim, 'features.6.conv.2': features_cv_dim,
                'features.7.conv.0.0': features_cv_dim, 'features.7.conv.1.0': features_cv_dim, 'features.7.conv.2': features_cv_dim,
                'features.8.conv.0.0': features_cv_dim, 'features.8.conv.1.0': features_cv_dim, 'features.8.conv.2': features_cv_dim,
                'features.9.conv.0.0': features_cv_dim, 'features.9.conv.1.0': features_cv_dim, 'features.9.conv.2': features_cv_dim,
                'features.10.conv.0.0': features_cv_dim, 'features.10.conv.1.0': features_cv_dim, 'features.10.conv.2': features_cv_dim,
                'features.11.conv.0.0': features_cv_dim, 'features.11.conv.1.0': features_cv_dim, 'features.11.conv.2': features_cv_dim,
                'features.12.conv.0.0': features_cv_dim, 'features.12.conv.1.0': features_cv_dim, 'features.12.conv.2': features_cv_dim,
                'features.13.conv.0.0': features_cv_dim, 'features.13.conv.1.0': features_cv_dim, 'features.13.conv.2': features_cv_dim,
                'features.14.conv.0.0': features_cv_dim, 'features.14.conv.1.0': features_cv_dim, 'features.14.conv.2': features_cv_dim,
                'features.15.conv.0.0': features_cv_dim, 'features.15.conv.1.0': features_cv_dim, 'features.15.conv.2': features_cv_dim,
                'features.16.conv.0.0': features_cv_dim, 'features.16.conv.1.0': features_cv_dim, 'features.16.conv.2': features_cv_dim,
                'features.17.conv.0.0': features_cv_dim, 'features.17.conv.1.0': features_cv_dim, 'features.17.conv.2': features_cv_dim,
                'features.18.0': features_cv_dim, 'classifier.1': 100,                
                }

        drillers_dict = load_all_drillers(
            n_cluster_list = [100, 150, 200, 250, 300, 400, 600],  
            target_layers = target_layers,
            drill_path = drill_path,
            device = device,
            feature_sizes = feature_sizes,
            cv_parsers = cv_parsers
            )

        peepholes = Peepholes(
                path = phs_path,
                name = phs_name,
                device = device
                )
       
        with corevecs as cv, peepholes as ph:
                # ds.load_only(
                #         loaders = loaders,
                #         verbose = verbose
                #         )

                cv.load_only(
                        loaders = loaders,
                        verbose = verbose 
                        ) 
                layer = "features.13.conv.1.0"
                X = cv._corevds['train'][layer]
                X_reduced = X[:, :10]
                        
                X_np = X_reduced.cpu().numpy()

                plot_tsne(X_np = X_np, 
                        save_path = plots_path,
                        file_name = "features13conv10_mobilenet_tsne")
                quit()
                #coverage = empp_coverage_scores(drillers=ph._drillers, threshold=0.8, plot=True, save_path='/home/claranunesbarrancos/repos/XAI/src/clustering_xp/temp_plots', file_name='coverage_vgg_550clusters.png')
                #empp_relative_coverage_scores(drillers=ph._drillers, threshold=0.8, plot=True, save_path='/home/claranunesbarrancos/repos/XAI/src/clustering_xp/temp_plots', file_name='relative_cluster_coverage_vgg_550clusters.png')
                compare_relative_coverage_all_clusters( all_drillers = drillers_dict,
                        threshold=0.8, plot= True, save_path=plots_path, file_name='relative_coverage_all_clusters_mobilenet.png')


