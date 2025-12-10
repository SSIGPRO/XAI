import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial

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
from peepholelib.datasets.functional.transforms import vgg16_cifar100 as ds_transform 
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

        model_dir = '/srv/newpenny/XAI/models'
        model_name = 'SV_model=vit_b_16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau_withInfo.pth'        
        
        svds_path = '/srv/newpenny/XAI/CN/vit_data'
        svds_name = 'svds' 
        
        cvs_path = Path.cwd()/'/srv/newpenny/XAI/CN/vit_data/corevectors'
        cvs_name = 'corevectors'

        drill_path = Path.cwd()/'/srv/newpenny/XAI/CN/vit_data/drillers_all'
        drill_name = 'classifier'

        plots_path = Path.cwd()/'temp_plots/coverage/'
        
        verbose = True 
        
        # Peepholelib
        

        n_cluster = 550

        n_conceptograms = 2 
        
        loaders = ['CIFAR100-train', 'CIFAR100-val', 'CIFAR100-test']

    #--------------------------------
    # Model 
    #--------------------------------
    
        nn = torchvision.models.vit_b_16()
        n_classes = len(Cifar100.get_classes(meta_path = Path(cifar_path)/'cifar-100-python/meta')) 
        target_layers = get_st_list(nn.state_dict().keys())
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
    # SVDs 
    #--------------------------------

        svd_fns = {}

        for layer in target_layers:
                svd_fns[layer] = partial(linear_svd,
                        layer = layer,
                        rank = 200,
                        device=device
                )


        with datasets as ds:
                ds.load_only(
                        loaders = loaders,
                        verbose = verbose
                        )

                model.get_svds(
                        path = svds_path,
                        name = svds_name,
                        target_modules = target_layers,
                        sample_in = ds._dss['CIFAR100-train']['image'][0],
                        svd_fns = svd_fns,
                        verbose = verbose
                        )
                viz_singular_values_2(model, svds_path)
    #--------------------------------
    # CoreVectors 
    #--------------------------------
        corevecs = CoreVectors(
                path = cvs_path,
                name = cvs_name,
                model = model,
                )
        
        # define a dimensionality reduction function for each layer
        reduction_fns = {}
        for layer in target_layers:
                if layer == "heads.head":
                        fn = linear_svd_projection  
                else:
                        fn = linear_svd_projection_ViT

                reduction_fns[layer] = partial(fn,
                        svd=model._svds[layer],
                        use_s=True,
                        device=device
                )

        with datasets as ds, corevecs as cv: 
                ds.load_only(
                        loaders = loaders,
                        verbose = verbose
                        )

                # computing the corevectors
                cv.get_coreVectors(
                        datasets = ds,
                        reduction_fns = reduction_fns,
                        save_input = True,
                        save_output = False,
                        batch_size = bs,
                        n_threads = n_threads,
                        verbose = verbose
                        )

                if not (cvs_path/(cvs_name+'.normalization.pt')).exists():
                        cv.normalize_corevectors(
                                wrt = 'CIFAR100-train',
                                to_file = cvs_path/(cvs_name+'.normalization.pt'),
                                batch_size = bs,
                                n_threads = n_threads,
                                verbose=verbose
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


        drillers_dict = load_all_drillers(
            n_cluster_list = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550],  
            target_layers = target_layers,
            drill_path = drill_path,
            device = device,
            feature_sizes = feature_sizes,
            cv_parsers = cv_parsers
            )

        # peepholes = Peepholes(
        #         path = phs_path,
        #         name = phs_name,
        #         device = device
        #         )
       
        with corevecs as cv:

                cv.load_only(
                        loaders = loaders,
                        verbose = verbose 
                        ) 
                # layer = "features.13.conv.1.0"
                # X = cv._corevds['train'][layer]
                # X_reduced = X[:, :10]
                        
                # X_np = X_reduced.cpu().numpy()

                # plot_tsne(X_np = X_np, 
                #         save_path = plots_path,
                #         file_name = "features13conv10_mobilenet_tsne")
                # quit()
                #coverage = empp_coverage_scores(drillers=ph._drillers, threshold=0.8, plot=True, save_path='/home/claranunesbarrancos/repos/XAI/src/clustering_xp/temp_plots', file_name='coverage_vgg_550clusters.png')
                #empp_relative_coverage_scores(drillers=ph._drillers, threshold=0.8, plot=True, save_path='/home/claranunesbarrancos/repos/XAI/src/clustering_xp/temp_plots', file_name='relative_cluster_coverage_vgg_550clusters.png')
                compare_relative_coverage_all_clusters(all_drillers = drillers_dict,
                        threshold=0.8, plot= True, save_path=plots_path, file_name='relative_coverage_all_clusters_vit.png')


