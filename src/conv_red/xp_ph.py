import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial

# torch stuff
import torch
from torchvision.models import vgg16
from cuda_selector import auto_cuda

###### Our stuff

# Model
from peepholelib.models.model_wrap import ModelWrap 

# datasets
from peepholelib.datasets.cifar100 import Cifar100
from peepholelib.datasets.parsedDataset import ParsedDataset 

# corevecs
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds.linear_svd import LinearSVD
from peepholelib.coreVectors.dimReduction.svds.conv2d_toeplitz_svd import Conv2dToeplitzSVD
from peepholelib.coreVectors.dimReduction.svds.conv2d_kernel_svd import Conv2dKernelSVD
from peepholelib.coreVectors.dimReduction.svds.conv2d_avg_kernel_svd import Conv2dAvgKernelSVD

# peepholes
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    cifar_path = '/srv/newpenny/dataset/CIFAR100'
    ds_path = Path.cwd()/'../data/datasets'

    # model parameters
    bs = 512 
    n_threads = 1

    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
     
    svds_path = Path.cwd()/'../data/svds'
    
    cvs_path = Path.cwd()/'../data/corevectors'
    cvs_name = 'corevectors'

    drill_path = Path.cwd()/'../data/drillers'
    drill_name = 'classifier'

    phs_path = Path.cwd()/'../data/peepholes'
    phs_name = 'peepholes'
    
    verbose = True 
    
    # Peepholelib
    target_layers = [
            #'features.26',
            'features.28',
            #'classifier.0',
            ]
    
    cv_dims = {
            #'features.26': 30,
            'features.28': 50,
            #'classifier.0': 30,
            }

    svd_rank = 300
    n_cluster = 50 
    
    loaders = [
            'CIFAR100-train',
            'CIFAR100-val',
            'CIFAR100-test',  
            #'CIFAR100-C-val-c0',
            #'CIFAR100-C-test-c0' 
            ]

    #--------------------------------
    # Model 
    #--------------------------------
    
    nn = vgg16()
    n_classes = len(Cifar100.get_classes(meta_path = Path(cifar_path)/'cifar-100-python/meta')) 

    model = ModelWrap(
            model = nn,
            target_modules = target_layers,
            device = device
            )
                                            
    model.update_output(
            output_layer = 'classifier.6', 
            to_n_classes = n_classes,
            overwrite = True 
            )
                                            
    model.load_checkpoint(
            name = model_name,
            path = model_dir,
            verbose = verbose
            )
                                            
    #--------------------------------
    # Datasets 
    #--------------------------------
    
    # Assuming we have a parsed dataset in ds_path
    datasets = ParsedDataset(
            path = ds_path,
            )

    #--------------------------------
    # SVDs 
    #--------------------------------
    t0 = time()
    with datasets as ds:
        ds.load_only(
                loaders = ['CIFAR100-train'],
                verbose = verbose
                )
        sample_in = ds._dss['CIFAR100-train']['image'][0]

        svds = {
                #'features.26': Conv2dToeplitzSVD(
                #    path = svds_path,
                #    layer = 'features.26',
                #    model = model,
                #    rank = svd_rank,
                #    sample_in = sample_in,
                #    ),
                'features.28': Conv2dAvgKernelSVD(
                    path = svds_path,
                    layer = 'features.28',
                    model = model,
                    rank = svd_rank,
                    ),
                #'classifier.0': LinearSVD(
                #    path = svds_path,
                #    layer = 'classifier.0',
                #    model = model,
                #    rank = svd_rank,
                #    verbose = verbose
                #    ),
                }
    print('time: ', time()-t0)

    #--------------------------------
    # CoreVectors 
    #--------------------------------
    
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            model = model,
            )
    
    with datasets as ds, corevecs as cv: 
        ds.load_only(
                loaders = loaders,
                verbose = verbose
                )

        # computing the corevectors
        cv.get_coreVectors(
                datasets = ds,
                reducers = svds,
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
                    #from_file = cvs_path/(cvs_name+'.normalization.pt'),
                    #loaders = ['CIFAR100-val', 'CIFAR100-test'],
                    batch_size = bs,
                    n_threads = n_threads,
                    verbose=verbose
                    )

    #--------------------------------
    # Peepholes
    #--------------------------------
    feature_sizes = {
            #'features.26': cv_dims['features.26'],
            # for channel_wise corevectors, the size is out_size * cv_dim
            # TODO: get 196 from somewhere
            'features.28': cv_dims['features.28'],
            #'classifier.0': cv_dims['classifier.0'],
            }

    drillers = {}
    for peep_layer in target_layers:
        cv_parser = partial(
                    svds[peep_layer].parser,
                    cv_dim = cv_dims[peep_layer]
                    )

        drillers[peep_layer] = tGMM(
                path = drill_path,
                name = f'{drill_name}.GMM.{peep_layer}.{n_classes}.{feature_sizes[peep_layer]}.{n_cluster}',
                target_module = peep_layer,
                nl_classifier = n_cluster,
                nl_model = n_classes,
                n_features = feature_sizes[peep_layer],
                parser = cv_parser,
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
                        loader = 'CIFAR100-train',
                        verbose=verbose
                        )
                print(f'Fitting time for {drill_key}  = ', time()-t0)

                driller.compute_empirical_posteriors(
                        datasets = ds,
                        corevectors = cv,
                        loader = 'CIFAR100-train',
                        batch_size = bs,
                        verbose=verbose
                        )
        
                # save classifiers
                print(f'Saving classifier for {drill_key}')
                driller.save()

    with datasets as ds, corevecs as cv, peepholes as ph:
        ds.load_only(
                loaders = loaders,
                verbose = verbose
                )

        cv.load_only(
                loaders = loaders,
                verbose = verbose 
                ) 

        ph.get_peepholes(
                datasets = ds,
                corevectors = cv,
                target_modules = target_layers,
                batch_size = bs,
                drillers = drillers,
                n_threads = n_threads,
                verbose = verbose
                )
