import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time

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
from peepholelib.datasets.functional.transforms import TransformWrap 
from peepholelib.datasets.functional.transforms import vgg16_transform as ds_transform 

# corevecs
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds.linear_svd import LinearSVD
from peepholelib.coreVectors.dimReduction.svds.conv2d_avg_kernel_svd import Conv2dAvgKernelSVD

# peepholes
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    cifar_path = '/srv/newpenny/dataset/CIFAR100'
    ds_path = Path.cwd()/'../data/datasets'

    model_path = '/srv/newpenny/XAI/models'
    model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
     
    svds_path = Path.cwd()/'../data/svds'
    
    cvs_path = Path.cwd()/'../data/corevectors'
    cvs_name = 'corevectors'

    drill_path = Path.cwd()/'../data/drillers'
    drill_name = 'GMM'

    phs_path = Path.cwd()/'../data/peepholes'
    phs_name = 'phs_macs'
    
    # model / driller parameters
    n_classes = 100
    bs = 512 
    svd_rank = 300
    n_cluster = 300 
    verbose = True 
    
    # Peepholelib
    target_layers = [f'features.{i}' for i in [7, 14, 21, 28]]
    
    cv_dims = {
            'features.7': 128,
            'features.14': 256,
            'features.21': 300,
            'features.28': 300,
            }
    cv_names = {l: cvs_name for l in target_layers}
    ph_names = {l: phs_name for l in target_layers}

    loaders = [
            'CIFAR100-train',
            'CIFAR100-val',
            'CIFAR100-test',  
            'CIFAR100-C-val-c0',
            'CIFAR100-C-test-c0',
            'MNIST-val',
            'MNIST-test',
            'Textures-val',
            'Textures-test',
            ]

    _transforms = {
            k: TransformWrap(transform=ds_transform, input_key='image') for k in loaders 
            }

    _inference_names = {
            k: ['vgg'] for k in loaders
            }
    _inference_names['CIFAR100-val'] += ['BIM', 'PGD']
    _inference_names['CIFAR100-test'] += ['BIM', 'PGD']

    #--------------------------------
    # Model 
    #--------------------------------
    model = ModelWrap(
            model = vgg16(),
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
            path = model_path,
            verbose = verbose
            )
                                            
    #--------------------------------
    # Datasets 
    #--------------------------------
    datasets = ParsedDataset(path = ds_path)

    #--------------------------------
    # SVDs
    #--------------------------------
    svds = {l: Conv2dAvgKernelSVD(
        path = svds_path,
        layer = l,
        model = model,
        rank = svd_rank,
        cv_dim = cv_dims[l],
        ) for l in target_layers
        }

    corevecs = CoreVectors(
            path = cvs_path,
            model = model,
            )

    #--------------------------------
    # Peepholes
    #--------------------------------
    drillers = {}
    for peep_layer in target_layers:
        drillers[peep_layer] = tGMM(
                path = drill_path,
                name = f'{drill_name}.{peep_layer}.{n_classes}.{cv_dims[peep_layer]}.{n_cluster}',
                target_module = peep_layer,
                nl_classifier = n_cluster,
                nl_model = n_classes,
                n_features = cv_dims[peep_layer],
                cls_kwargs = {
                    'covariance_regularization': 1e-5,
                    'convergence_tolerance': 1e-3
                    },
                reducer = svds[peep_layer],
                device = device
                )

    peepholes = Peepholes(
            path = phs_path,
            device = device
            )

    # fit MRC signatures
    with datasets as ds, corevecs as cv:
        ds.load_only(
                loaders = loaders,
                transforms = _transforms,
                inference_names = _inference_names,
                verbose = verbose
                )

        cv.load_only(
                loaders = list(ds._dss.keys()),
                names = cv_names,
                verbose = verbose 
                ) 
        
        for drill_key, driller in drillers.items():
            if not driller.load():
                t0 = time()
                print(f'Fitting classifier for {drill_key}')
                driller.fit(
                        datasets = ds,
                        corevectors = cv,
                        loader = 'CIFAR100-train-vgg',
                        verbose=verbose
                        )
                print(f'  fit time: {time()-t0:.1f}s')
                driller.save()

    with datasets as ds, corevecs as cv, peepholes as ph:
        ds.load_only(
                loaders = loaders,
                transforms = _transforms,
                inference_names = _inference_names,
                verbose = verbose
                )

        cv.load_only(
                loaders = list(ds._dss.keys()),
                names = cv_names,
                verbose = verbose 
                ) 

        ph.get_peepholes(
                datasets = ds,
                corevectors = cv,
                target_modules = target_layers,
                batch_size = bs,
                drillers = drillers,
                names = ph_names,
                verbose = verbose
                )
