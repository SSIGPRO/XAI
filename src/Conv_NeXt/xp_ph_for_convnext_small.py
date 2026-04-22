import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial

# torch stuff
import torch
from torchvision.models import convnext_small
from cuda_selector import auto_cuda

###### Our stuff

# Model
from peepholelib.models.model_wrap import ModelWrap 

# datasets
from peepholelib.datasets.cifar100 import Cifar100
from peepholelib.datasets.parsedDataset import ParsedDataset
from peepholelib.datasets.functional.transforms import TransformWrap 
from peepholelib.datasets.functional.transforms import convnext_small_transform as ds_transform 

# corevecs
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds.linear_svd import LinearSVD
from peepholelib.coreVectors.dimReduction.svds.conv2d_toeplitz_svd import Conv2dToeplitzSVD

# # peepholes
# from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
# from peepholelib.peepholes.peepholes import Peepholes

if __name__ == "__main__":
#     use_cuda = torch.cuda.is_available()
#     device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
#     print(f"Using {device} device")
    gpu_id = 0 # physical GPU index
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > gpu_id
    device = torch.device(f"cuda:{gpu_id}" if use_cuda else "cpu")
    device = torch.device('cpu')  # Force CPU usage
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    cifar_path = '/srv/newpenny/dataset/CIFAR100'
    ds_path = '/home/arshakumari/repos/XAI/data/datasets'  #Path.cwd()/'../data/datasets'

    # model parameters
    bs = 10#512 
    n_threads = 1

    model_path = '/srv/newpenny/XAI/models'
    model_name = 'convnext_cifar100_clean_sd.pt'
     
    svds_path = Path.cwd()/'../data/svds'
    
    cvs_path = Path.cwd()/'../data/corevectors'
    cvs_name = 'corevectors'

    drill_path = Path.cwd()/'../data/drillers'
    drill_name = 'classifier'

    phs_path = Path.cwd()/'../data/peepholes'
    phs_name = 'peepholes'
    
    verbose = True 
    
    # Peepholelib
    target_layers = []

# Add Stem
    target_layers.append('features.0.0')

# Stage configuration: (feature_index, number_of_blocks)
    stages = [
        (1, 3),   # Stage 1
        (3, 3),   # Stage 2
        (5, 27),  # Stage 3
        (7, 3),   # Stage 4
]

# Layers inside each CNBlock
    inner_layers = [0, 3, 5]

# Generate all layers
    for stage, num_blocks in stages:
        for block in range(num_blocks):
            for inner in inner_layers:
                target_layers.append(f'features.{stage}.{block}.block.{inner}')

# Add classifier
    target_layers.append('classifier.2')
#----Trget_layers_code_part_end-------------
#----CV_dims_code_part_start----------------
    cv_dims = {}

# Stem
    cv_dims['features.0.0'] = 32

# Stage configuration
    stages = [
        (1, 3),   # Stage 1
        (3, 3),   # Stage 2
        (5, 27),  # Stage 3
        (7, 3),   # Stage 4
]

# Inner layer → dimension mapping
    inner_dim_map = {
        0: 96,   # Conv
        3: 192,  # Linear
        5: 192   # Linear
}
# Generate automatically
    for stage, num_blocks in stages:
        for block in range(num_blocks):
            for inner, dim in inner_dim_map.items():
                layer_name = f'features.{stage}.{block}.block.{inner}'
                cv_dims[layer_name] = dim
# Classifier
    cv_dims['classifier.2'] = 96

#------cv_dims_code_part_end-------------

    svd_rank = 300
    n_cluster = 4 
    
    loaders = [
            'CIFAR100-train',
            'CIFAR100-val',
            'CIFAR100-test',  
            'CIFAR100-C-val-c0',
            'CIFAR100-C-test-c0' 
            ]

    _transforms = {
            k: TransformWrap(transform=ds_transform, input_key='image') for k in loaders 
            }

    _inference_names = {
            k: ['convnext_small'] for k in loaders
            }

    n_classes = len(Cifar100.get_classes(meta_path = Path(cifar_path)/'cifar-100-python/meta')) 

    #--------------------------------
    # Model 
    #--------------------------------

    model = ModelWrap(
            model = convnext_small(),
            target_modules = target_layers,
            device = device
            )
                                            
    model.update_output(
            output_layer = 'classifier.2', 
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
                transforms = _transforms,
                inference_names = _inference_names,
                verbose = verbose
                )
        sample_in = ds._dss['CIFAR100-train-convnext_small'][0]['image']

#--------svds_code_part_start----------------
    svds = {}

    # Stem (Conv layer)
    svds['features.0.0'] = Conv2dToeplitzSVD(
        path=svds_path,
        layer='features.0.0',
        model=model,
        rank=svd_rank,
        cv_dim=cv_dims['features.0.0'],
        sample_in=sample_in,
    )

    # Stage configuration
    stages = [
        (1, 3),
        (3, 3),
        (5, 27),
        (7, 3),
    ]

    # Generate layers
    for stage, num_blocks in stages:
        for block in range(num_blocks):
            for inner in [0, 3, 5]:
                layer_name = f'features.{stage}.{block}.block.{inner}'
                if inner == 0:
                    svds[layer_name] = Conv2dToeplitzSVD(
                        path=svds_path,
                        layer=layer_name,
                        model=model,
                        rank=svd_rank,
                        cv_dim=cv_dims[layer_name],
                        sample_in=sample_in,
                    )
                else:
                    svds[layer_name] = LinearSVD(
                        path=svds_path,
                        layer=layer_name,
                        model=model,
                        rank=svd_rank,
                        cv_dim=cv_dims[layer_name],
                        sample_in=sample_in,
                    )

    # Classifier (Linear)
    svds['classifier.2'] = LinearSVD(
        path=svds_path,
        layer='classifier.2',
        model=model,
        rank=svd_rank,
        cv_dim=cv_dims['classifier.2'],
        verbose=verbose
    )
#--------svds_code_part_end------------------
    print('time: ', time()-t0)
    quit()
    # #--------------------------------
    # # CoreVectors 
    # #--------------------------------
    
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            model = model,
            )
    
    with datasets as ds, corevecs as cv: 
        ds.load_only(
                loaders = loaders,
                transforms = _transforms,
                inference_names = _inference_names,
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
                wrt = 'CIFAR100-train-convnext_small',
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
#     drillers = {}
#     for peep_layer in target_layers:
#         drillers[peep_layer] = tGMM(
#                 path = drill_path,
#                 name = f'{drill_name}.GMM.{peep_layer}.{n_classes}.{cv_dims[peep_layer]}.{n_cluster}',
#                 target_module = peep_layer,
#                 nl_classifier = n_cluster,
#                 nl_model = n_classes,
#                 n_features = cv_dims[peep_layer],
#                 cls_kwargs = {
#                     'covariance_regularization': 1e-4,
#                     'convergence_tolerance': 1e-2
#                     },
#                 reducer = svds[peep_layer],
#                 device = device
#                 )

#     peepholes = Peepholes(
#             path = phs_path,
#             name = phs_name,
#             device = device
#             )

#     # fitting classifiers
#     with datasets as ds, corevecs as cv:
#         ds.load_only(
#                 loaders = loaders,
#                 transforms = _transforms,
#                 inference_names = _inference_names,
#                 verbose = verbose
#                 )

#         cv.load_only(
#                 loaders = list(ds._dss.keys()),
#                 verbose = verbose 
#                 ) 

#         for drill_key, driller in drillers.items():
#             if not driller.load():
#                 t0 = time()
#                 print(f'Fitting classifier for {drill_key}')
#                 driller.fit(
#                         datasets = ds,
#                         corevectors = cv,
#                         loader = 'CIFAR100-train-vgg',
#                         verbose=verbose
#                         )
#                 print(f'Fitting time for {drill_key}  = ', time()-t0)

#                 # save classifiers
#                 print(f'Saving classifier for {drill_key}')
#                 driller.save()

#     with datasets as ds, corevecs as cv, peepholes as ph:
#         ds.load_only(
#                 loaders = loaders,
#                 transforms = _transforms,
#                 inference_names = _inference_names,
#                 verbose = verbose
#                 )

#         cv.load_only(
#                 loaders = list(ds._dss.keys()),
#                 verbose = verbose 
#                 ) 

#         ph.get_peepholes(
#                 datasets = ds,
#                 corevectors = cv,
#                 target_modules = target_layers,
#                 batch_size = bs,
#                 drillers = drillers,
#                 n_threads = n_threads,
#                 verbose = verbose
#                 )
