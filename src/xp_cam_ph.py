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
from peepholelib.datasets.parsedDataset import ParsedDataset
from peepholelib.datasets.functional.transforms import TransformWrap
from peepholelib.datasets.functional.transforms import vgg16_transform as ds_transform

# corevecs
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds.conv2d_avg_kernel_svd import Conv2dAvgKernelSVD

# peepholes
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.peepholes.CoverageAnalysisMethods.MRC import MRC

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

    svds_path  = Path.cwd()/'../data/svds'
    cvs_path   = Path.cwd()/'../data/corevectors'
    cvs_name  = 'corevectors'

    drill_path = Path.cwd()/'../data/drillers'
    drill_name = 'MRC'

    phs_path   = Path.cwd()/'../data/peepholes'
    phs_name  = 'phs_mrc'

    # model / driller parameters
    n_classes = 100
    svd_rank = 300
    bs = 512
    n_threads = 1
    Q = 1   # number of MRC sub-ranges
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
    # Dataset
    #--------------------------------
    datasets = ParsedDataset(path=ds_path)

    #--------------------------------
    # SVDs  (Conv2dAvgKernelSVD for all conv layers)
    #--------------------------------

    svds = {l: Conv2dAvgKernelSVD(
        path = svds_path,
        layer = l,
        model = model,
        rank = svd_rank,
        cv_dim = cv_dims[l],
        ) for l in target_layers
        }

    #--------------------------------
    # CoreVectors
    #--------------------------------
    corevecs = CoreVectors(
            path = cvs_path,
            model = model,
            )

    #--------------------------------
    # MRC drillers
    #--------------------------------
    drillers = {}
    for layer in target_layers:
        drillers[layer] = MRC(
                path = drill_path,
                name = f'{drill_name}.{layer}.{n_classes}.{cv_dims[layer]}.Q{Q}',
                target_module = layer,
                nl_model = n_classes,
                n_features = cv_dims[layer],
                Q = Q,
                reducer = svds[layer],
                device = device
                )

    peepholes = Peepholes(
            path = phs_path,
            device = device
            )

    # fit drillers
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

        for layer, driller in drillers.items():
            if not driller.load():
                t0 = time()
                print(f'Fitting MRC for {layer}')
                driller.fit(
                        datasets = ds,
                        corevectors = cv,
                        loader = 'CIFAR100-train-vgg',
                        verbose = verbose
                        )
                print(f'  fit time: {time()-t0:.1f}s')
                driller.save()

    #--------------------------------
    # Peepholes
    #--------------------------------
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
                drillers = drillers,
                names = ph_names,
                batch_size = bs,
                n_threads = n_threads,
                verbose = verbose
                )

