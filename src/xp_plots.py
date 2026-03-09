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
from peepholelib.datasets.functional.transforms import TransformWrap 
from peepholelib.datasets.functional.transforms import vgg16_transform as ds_transform 

# peepholes
from peepholelib.peepholes.peepholes import Peepholes

# scoring
from peepholelib.scores.protoclass import conceptogram_protoclass_score as proto_score
from peepholelib.scores.model_confidence import  model_confidence_score as mconf_score 
from peepholelib.scores.dmd import DMD_score as dmd_score 

# plotting
from peepholelib.plots.confidence import plot_confidence
from peepholelib.plots.calibration import plot_calibration
from peepholelib.plots.ood import plot_ood
from peepholelib.plots.conceptograms import plot_conceptogram

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    cifar_path = '/srv/newpenny/dataset/CIFAR100'
    ds_path = Path.cwd()/'../data/datasets'

    phs_path = Path.cwd()/'../data/peepholes'
    phs_name = 'peepholes'
    avg_name = 'peepholes_avg'

    plots_path = Path.cwd()/'temp_plots/xp_plots/'
    verbose = True 
    
    # Peepholelib
    target_layers = [
            'features.26',
            'features.28',
            'classifier.0',
            ]
    
    n_conceptograms = 2 

    loaders = [
            'CIFAR100-train',
            'CIFAR100-val',
            'CIFAR100-test', 
            'CIFAR100-C-val-c0',
            'CIFAR100-C-test-c0', 
            ]

    _transforms = { 
            k: TransformWrap(transform=ds_transform, input_key='image') for k in loaders 
            }

    _inference_names = {
            k: ['vgg'] for k in loaders
            }

    #--------------------------------
    # Datasets 
    #--------------------------------
    
    # Assuming we have a parsed dataset in ds_path
    ds = ParsedDataset(
            path = ds_path,
            )

    #--------------------------------
    # Peepholes
    #--------------------------------

    ph = Peepholes(
            path = phs_path,
            name = phs_name,
            device = device
            )

    dmd_ph = Peepholes(
            path = phs_path,
            name = avg_name,
            device = device
            )

    with ds, ph, dmd_ph:
        ds.load_only(
                loaders = loaders,
                transforms = _transforms,
                inference_names = _inference_names,
                verbose = verbose
                )

        ph.load_only(
                loaders = list(ds._dss.keys()),
                verbose = verbose
                )

        dmd_ph.load_only(
                loaders = list(ds._dss.keys()),
                verbose = verbose
                )

        # get scores
        scores, protoclasses = proto_score(
                datasets = ds,
                peepholes = ph,
                proto_key = 'CIFAR100-train-vgg',
                verbose = verbose
                )
                                        
        scores = mconf_score(
                datasets = ds,
                append_scores = scores,
                verbose = verbose
                )

        scores = dmd_score(
                peepholes = dmd_ph,
                pos_loader_train = 'CIFAR100-val-vgg',
                pos_loader_test = 'CIFAR100-test-vgg',
                neg_loaders = {
                    'CIFAR100-C-test-c0-vgg': ['CIFAR100-C-val-c0-vgg'],
                    },
                append_scores = scores,
                )
        
        # make plots
        plot_confidence(
                datasets = ds,
                scores = scores,
                max_score = 1.,
                path = plots_path,
                verbose = verbose
                )

        plot_calibration(
                datasets = ds,
                scores = scores,
                calib_bin = 0.1,
                path = plots_path,
                verbose = verbose
                )

        plot_ood(
                scores = scores,
                id_loaders = {
                    'Proto-Class': 'CIFAR100-test-vgg',
                    'MSP': 'CIFAR100-test-vgg',
                    'DMD': 'CIFAR100-C-val-c0-vgg',
                    },
                ood_loaders = ['CIFAR100-C-test-c0-vgg'],
                path = plots_path,
                verbose = verbose
                )

        # plot conceptograms
        idx = [2, 5]
        plot_conceptogram(
                path = plots_path,
                name = 'conceptogram',
                datasets = ds,
                peepholes = ph,
                loaders = ['CIFAR100-test-vgg'],
                samples = idx,
                target_modules = target_layers,
                classes = Cifar100.get_classes(meta_path = Path(cifar_path)/'cifar-100-python/meta'),
                protoclasses = protoclasses,
                scores = scores,
                verbose = verbose,
                )
