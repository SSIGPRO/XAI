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
        ds_path = '/srv/newpenny/XAI/CN/data/corevectors'

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

        drill_path = Path.cwd()/'/srv/newpenny/XAI/CN/data/drillers_all/drillers_50'
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
        n_cluster = 50

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
    # SVDs 
    #--------------------------------
        svd_fns = {}

        for layer in target_layers:
            if 'classifier' in layer:
                svd_type = linear_svd
            else:
                svd_type = conv2d_toeplitz_svd

            svd_fns[layer] = partial(svd_type,
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
                        sample_in = ds._dss['train']['image'][0],
                        svd_fns = svd_fns,
                        verbose = verbose
                        )

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
                if layer == "classifier.1":
                        fn = linear_svd_projection  
                else:
                        fn = conv2d_toeplitz_svd_projection

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

                if layer == "classifier.1":
                        features_cv_dim = 100
                else:
                        features_cv_dim = 300
                cv_parsers[layer] = partial(trim_corevectors,
                        module = layer,
                        cv_dim = features_cv_dim)
                feature_sizes[layer] = features_cv_dim

        drillers = {}
        for peep_layer in target_layers:
                drillers[peep_layer] = tGMM(
                        path = drill_path,
                        name = drill_name+'.'+peep_layer,
                        nl_classifier = n_cluster,
                        nl_model = n_classes,
                        n_features = feature_sizes[peep_layer],
                        parser = cv_parsers[peep_layer],
                        device = device
                        )


        peepholes = Peepholes(
                path = phs_path,
                name = phs_name,
                device = device
                )
       
        # fitting classifiers
        with datasets as ds, corevecs as cv:
                ds.load_only(
                        loaders = loaders,
                        verbose = verbose
                        )

                cv.load_only(
                        loaders = loaders,
                        verbose = verbose 
                        ) 

                for drill_key, driller in drillers.items():
                        if (driller._empp_file).exists():
                                print(f'Loading Classifier for {drill_key}') 
                                driller.load()
                        else:
                                t0 = time()
                                print(f'Fitting classifier for {drill_key}')
                                driller.fit(
                                        corevectors = cv,
                                        loader = 'train',
                                        verbose=verbose
                                        )
                                print(f'Fitting time for {drill_key}  = ', time()-t0)

                                driller.compute_empirical_posteriors(
                                        datasets = ds,
                                        corevectors = cv,
                                        loader = 'train',
                                        batch_size = bs,
                                        verbose=verbose
                                        )
                        
                                # save classifiers
                                print(f'Saving classifier for {drill_key}')
                                driller.save()