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
from peepholelib.coreVectors.dimReduction.svds.conv2d_toeplitz_svd import Conv2dToeplitzSVD

# peepholes
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.scores.protoclass import conceptogram_protoclass_score as proto_score
from peepholelib.plots.conceptograms import *
if __name__ == "__main__":
#     use_cuda = torch.cuda.is_available()
#     device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
#     print(f"Using {device} device")
    gpu_id = 2 # physical GPU index
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > gpu_id
    device = torch.device(f"cuda:{gpu_id}" if use_cuda else "cpu")
    #device = torch.device('cpu')  # Force GPU usage
    print(f"Using {device} device")    # Directories definitions
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
    #----Target_layers_start-----
    target_layers = []
    target_layers.append('features.0.0')

        # Stage configuration
    stages = [
        (1, 3),   # Stage 1
        (3, 3),
        (5, 27),
        (7, 3),
        ]

        # Feature layers (inner layer = 0)
    for stage, num_blocks in stages:
        for block in range(num_blocks):
                target_layers.append(f'features.{stage}.{block}.block.0')
#     target_layers.append('classifier.2')   # Linear layer
#     print('Target layers:', target_layers)
#     quit()
    
    #----Target_layers_code_part_end-------------

    #----CV_dims_code_part_start----------------
    cv_dims = {}

    # Only inner layer 0
    inner_dim = 96  # dimension for layer 0

    for _l in target_layers:
        cv_dims[_l] = inner_dim
        
    
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

        # -----------------------------
        # ✅ Stem (Conv layer)
        # -----------------------------
        svds['features.0.0'] = Conv2dToeplitzSVD(
            path=svds_path,
            layer='features.0.0',
            model=model,
            rank=svd_rank,
            cv_dim=cv_dims['features.0.0'], 
            sample_in=sample_in,
        )

        # -----------------------------
        # Define stages
        # -----------------------------
        stages = [
            (1, 3),   # Stage 1
            (3, 3),   # Stage 2
            (5, 27),  # Stage 3
            (7, 3)    # Stage 4
        ]

        # -----------------------------
        # ✅ Conv layers (block.0)
        # -----------------------------
        for stage, num_blocks in stages:
            for block in range(num_blocks):
                layer_name = f'features.{stage}.{block}.block.0'

                svds[layer_name] = Conv2dToeplitzSVD(
                    path=svds_path,
                    layer=layer_name,
                    model=model,
                    rank=svd_rank,
                    cv_dim=cv_dims[layer_name],
                    sample_in=sample_in,
                    
                )

    #---------End of this part---------------
    print('time: ', time()-t0)
    
#   #--------svds_code_part_end------------------
#     # #--------------------------------
#     # # CoreVectors 
#     # #--------------------------------
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
   
    #--------------------------------
    # Peepholes
    #--------------------------------
    drillers = {}
    for peep_layer in target_layers:
        drillers[peep_layer] = tGMM(
                path = drill_path,
                name = f'{drill_name}.GMM.{peep_layer}.{n_classes}.{cv_dims[peep_layer]}.{n_cluster}',
                target_module = peep_layer,
                nl_classifier = n_cluster,
                nl_model = n_classes,
                n_features = cv_dims[peep_layer],
                cls_kwargs = {
                    'covariance_regularization': 1e-4,
                    'convergence_tolerance': 1e-2
                    },
                reducer = svds[peep_layer],
                device = device,
                )
        
        # drillers[peep_layer]._classifier.trainer_params = {
        #     "accelerator": "cpu",   # or "gpu" if you want GPU
        #     "devices": 2
        #     }

    peepholes = Peepholes(
            path = phs_path,
            name = phs_name,
            device = device
            )

    # fitting classifiers
    with datasets as ds, corevecs as cv:
        ds.load_only(
                loaders = loaders,
                transforms = _transforms,
                inference_names = _inference_names,
                verbose = verbose
                )

        cv.load_only(
                loaders = list(ds._dss.keys()),
                verbose = verbose 
                ) 

        for drill_key, driller in drillers.items():
            if not driller.load():
                t0 = time()
                print(f'Fitting classifier for {drill_key}')
                driller.fit(
                        datasets = ds,
                        corevectors = cv,
                        loader = 'CIFAR100-train-convnext_small',
                        verbose=verbose
                        )
                print(f'Fitting time for {drill_key}  = ', time()-t0)

                # save classifiers
                print(f'Saving classifier for {drill_key}')
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
#----------Protoclass-----------
        scores, protoclasses = proto_score(
                        datasets = ds,
                        peepholes = ph,
                        proto_key = 'CIFAR100-train-convnext_small',
                        verbose = verbose
                        )
        print('Protoclass scores: ', scores)
        print('Protoclasses: ', protoclasses)
        print(protoclasses.shape)
        
#-----------------------------------
#        Conceptogram plot
#-----------------------------------
# -----------------------------------
# Plot Conceptograms
# -----------------------------------

        # Get CIFAR100 class names
        class_names = Cifar100.get_classes(
        meta_path=Path(cifar_path)/'cifar-100-python/meta'
        )

        # Convert list -> dictionary
        classes_dict = {i: name for i, name in enumerate(class_names)}

        # Path to save plots
        plot_path = Path.cwd() / "../data/conceptogram_plots"

        # Select sample indexes to visualize
        samples_to_plot = [0, 1, 2, 3, 4]

        # Call plotting function
        plot_conceptogram(
        path=plot_path,
        name='conceptogram',
        datasets=ds,
        peepholes=ph,
        loaders=['CIFAR100-test-convnext_small'],
        samples=samples_to_plot,
        target_modules=target_layers,
        protoclasses=protoclasses,
        scores=scores,
        classes=classes_dict,
        ticks=target_layers,
        krows=5,
        verbose=True
        )
        print("Conceptogram plots saved!")
#-----------------------------------
#-----------------------------------
#       Plot of one class heatmap
#-----------------------------------
        # import matplotlib.pyplot as plt
        # import numpy as np

        # class_id = 30

        # proto = protoclasses[class_id].cpu().numpy()

        # # Normalize for better visibility
        # proto = (proto - proto.min()) / (proto.max() - proto.min())

        # plt.figure(figsize=(14,8))

        # plt.imshow(proto, aspect='auto', cmap='viridis')

        # plt.colorbar(label='Activation Score')

        # plt.xlabel("Feature Dimension")
        # plt.ylabel("Prototype / Layer")

        # plt.title(f"Protoclass Heatmap - Class {class_id}")

        # plt.show(block=True)
        # plt.savefig("proto_heatmap.png")
        # print("saved")
        
# --------------------------------
# Conceptograms
# --------------------------------

# conceptogram_path = Path.cwd() / '../data/conceptograms'

# # CIFAR100 class names
# class_names = Cifar100.get_classes(
#     meta_path=Path(cifar_path) / 'cifar-100-python/meta'
# )

# # convert list -> dictionary
# classes_dict = {i: cls for i, cls in enumerate(class_names)}

# with datasets as ds, corevecs as cv, peepholes as ph:

#     # --------------------------------
#     # Load datasets
#     # --------------------------------
#     ds.load_only(
#             loaders = loaders,
#             transforms = _transforms,
#             inference_names = _inference_names,
#             verbose = verbose
#             )

#     # --------------------------------
#     # Load corevectors
#     # --------------------------------
#     cv.load_only(
#             loaders = list(ds._dss.keys()),
#             verbose = True
#             )

#     # --------------------------------
#     # Generate/load peepholes
#     # --------------------------------
#     ph.get_peepholes(
#             datasets = ds,
#             corevectors = cv,
#             target_modules = target_layers,
#             batch_size = bs,
#             drillers = drillers,
#             n_threads = n_threads,
#             verbose = True
#             )
    
#     # --------------------------------
#     # Print available dataset keys
#     # --------------------------------
#     print(ds._dss.keys())

#     # --------------------------------
#     # Example samples to visualize
#     # --------------------------------
#     samples_to_plot = [0, 1, 2, 3]

#     # --------------------------------
#     # Plot conceptograms
#     # --------------------------------
#     plot_conceptogram(
#             path = conceptogram_path,

#             name = 'convnext_conceptogram',

#             datasets = ds,

#             peepholes = ph,

#             loaders = ['CIFAR100-test-convnext_small'],

#             samples = samples_to_plot,

#             target_modules = target_layers,

#             classes = classes_dict,

#             ticks = target_layers,

#             krows = 5,

#             verbose = True
#     )

# print(type(ph._phs))
# print(len(ph._phs))
# print(ph._phs.keys())
# test_ph = ph._phs['CIFAR100-test-convnext_small']
# print(test_ph.keys())