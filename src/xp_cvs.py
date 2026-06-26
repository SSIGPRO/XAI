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
from peepholelib.coreVectors.dimReduction.svds.conv2d_avg_kernel_svd import Conv2dAvgKernelSVD
from peepholelib.coreVectors.dimReduction.svds.linear_svd import LinearSVD

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    cifar_path = '/srv/newpenny/dataset/CIFAR100'
    ds_path = Path.cwd()/'../data/datasets'

    bs = 512
    n_threads = 1

    model_path = '/srv/newpenny/XAI/models'
    model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'

    svds_path = Path.cwd()/'../data/svds'
    cvs_path = Path.cwd()/'../data/corevectors'
    cvs_name = 'corevectors'

    verbose = True

    # Peepholelib
    target_layers = [
            'features.26',
            'features.28',
            'classifier.0',
            ]

    cv_dims = {
            'features.26': 30,
            'features.28': 50,
            'classifier.0': 30,
            }

    cv_names = {
            'features.26': cvs_name,
            'features.28': cvs_name,
            'classifier.0': cvs_name,
            }

    svd_rank = 300

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

    n_classes = len(Cifar100.get_classes(meta_path = Path(cifar_path)/'cifar-100-python/meta'))

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
    datasets = ParsedDataset(path=ds_path)

    #--------------------------------
    # SVDs
    #--------------------------------
    svds = {
            'features.26': Conv2dAvgKernelSVD(
                path = svds_path,
                layer = 'features.26',
                model = model,
                rank = svd_rank,
                cv_dim = cv_dims['features.26'],
                ),
            'features.28': Conv2dAvgKernelSVD(
                path = svds_path,
                layer = 'features.28',
                model = model,
                rank = svd_rank,
                cv_dim = cv_dims['features.28'],
                ),
            'classifier.0': LinearSVD(
                path = svds_path,
                layer = 'classifier.0',
                model = model,
                rank = svd_rank,
                cv_dim = cv_dims['classifier.0'],
                verbose = verbose
                ),
            }

    #--------------------------------
    # CoreVectors
    #--------------------------------
    corevecs = CoreVectors(
            path = cvs_path,
            model = model,
            )

    with datasets as ds, corevecs as cv:
        ds.load_only(
                loaders = loaders,
                transforms = _transforms,
                inference_names = _inference_names,
                verbose = verbose
                )

        cv.get_coreVectors(
                datasets = ds,
                reducers = svds,
                save_input = True,
                save_output = False,
                names = cv_names,
                batch_size = bs,
                n_threads = n_threads,
                verbose = verbose
                )

        cv.normalize_corevectors(
                wrt = 'CIFAR100-train-vgg',
                batch_size = bs,
                n_threads = n_threads,
                verbose = verbose
                )
