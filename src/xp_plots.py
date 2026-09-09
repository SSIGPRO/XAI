import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

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

# peepholes
from peepholelib.peepholes.peepholes import Peepholes

# scoring
from peepholelib.scores.protoclass import ProtoClassScore
from peepholelib.scores.model_output import ModelOutputScore
from peepholelib.scores.dmd import DMDScore
from peepholelib.scores.cam import CAMLinScore, CAMExpScore
from peepholelib.scores.vim import VIMScore

# plotting, commented with the plots at the end of the file
'''
from peepholelib.datasets.cifar100 import Cifar100
from peepholelib.plots.confidence import plot_confidence
from peepholelib.plots.calibration import plot_calibration
from peepholelib.plots.ood import plot_ood
from peepholelib.plots.conceptograms import plot_conceptogram
'''

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    cifar_path = '/srv/newpenny/dataset/CIFAR100'
    ds_path = Path.cwd()/'../data/datasets'

    model_path = '/srv/newpenny/XAI/models'
    model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'

    phs_path = Path.cwd()/'../data/peepholes'
    macs_phs_name = 'phs_macs'
    dmd_phs_name = 'phs_dmd'
    cam_phs_name = 'phs_mrc'

    plots_path = Path.cwd()/'temp_plots/xp_plots/'
    scores_path = Path.cwd()/'temp_plots/scores/'

    # model / score parameters
    n_classes = 100
    bs = 256
    n_threads = 1
    verbose = True

    # Peepholelib
    target_layers = [f'features.{i}' for i in [7, 14, 21, 28]]
    output_layer = 'classifier.6'

    n_conceptograms = 2

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


    macs_names = {l: macs_phs_name for l in target_layers}
    dmd_names = {l: dmd_phs_name for l in target_layers}
    cam_names = {l: cam_phs_name for l in target_layers}

    #--------------------------------
    # Loaders used by the scores
    #--------------------------------
    pos_loader_train = 'CIFAR100-val-vgg'
    pos_loader_test = 'CIFAR100-test-vgg'
    fit_key = 'CIFAR100-train-vgg'

    # ordered as `ood_loaders`, so that `pair_id_loaders()` lines up with it
    neg_loaders = {
            'CIFAR100-C-test-c0-vgg': ['CIFAR100-C-val-c0-vgg'],
            'MNIST-test-vgg': ['MNIST-val-vgg'],
            'Textures-test-vgg': ['Textures-val-vgg'],
            'CIFAR100-test-BIM': ['CIFAR100-val-BIM'],
            'CIFAR100-test-PGD': ['CIFAR100-val-PGD'],
            }
    ood_loaders = list(neg_loaders.keys())

    #--------------------------------
    # Model
    #--------------------------------
    # needed by the scores taking the activations, e.g. ViM
    model = ModelWrap(
            model = vgg16(),
            target_modules = target_layers,
            device = device
            )

    model.update_output(
            output_layer = output_layer,
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
    ds = ParsedDataset(
            path = ds_path,
            )

    #--------------------------------
    # Peepholes
    #--------------------------------

    ph = Peepholes(
            path = phs_path,
            device = device
            )

    dmd_ph = Peepholes(
            path = phs_path,
            device = device
            )

    cam_ph = Peepholes(
            path = phs_path,
            device = device
            )

    with ds, ph, dmd_ph, cam_ph:
        ds.load_only(
                loaders = loaders,
                transforms = _transforms,
                inference_names = _inference_names,
                verbose = verbose
                )

        ph.load_only(
                loaders = list(ds._dss.keys()),
                names = macs_names,
                verbose = verbose
                )

        dmd_ph.load_only(
                loaders = list(ds._dss.keys()),
                names = dmd_names,
                verbose = verbose
                )

        cam_ph.load_only(
                loaders = list(ds._dss.keys()),
                names = cam_names,
                verbose = verbose
                )

        score_loaders = list(ds._dss.keys())

        #--------------------------------
        # Scores
        #--------------------------------
        macs = ProtoClassScore(path=scores_path, name='MACS')
        if not macs.load():
            macs.fit(
                    datasets = ds,
                    peepholes = ph,
                    fit_key = fit_key,
                    verbose = verbose
                    )
        macs(
                datasets = ds,
                peepholes = ph,
                loaders = score_loaders,
                verbose = verbose
                )

        msp = ModelOutputScore(path=scores_path, type='MSP')
        msp(
                datasets = ds,
                loaders = score_loaders,
                verbose = verbose
                )

        vim = VIMScore(path=scores_path, name='ViM')
        if not vim.load():
            vim.fit(
                    model = model,
                    datasets = ds,
                    output_layer = output_layer,
                    fit_key = fit_key,
                    batch_size = bs,
                    n_threads = n_threads,
                    verbose = verbose
                    )
        vim(
                model = model,
                datasets = ds,
                output_layer = output_layer,
                loaders = score_loaders,
                batch_size = bs,
                n_threads = n_threads,
                verbose = verbose
                )

        dmd = DMDScore(path=scores_path, name='DMD-a')
        if not dmd.load():
            dmd.fit(
                    peepholes = dmd_ph,
                    pos_loader_train = pos_loader_train,
                    neg_loaders = neg_loaders,
                    verbose = verbose
                    )
        dmd(
                peepholes = dmd_ph,
                pos_loader_test = pos_loader_test,
                verbose = verbose
                )

        cam_lin = CAMLinScore(path=scores_path, name='CAM-lin')
        cam_lin(
                datasets = ds,
                peepholes = cam_ph,
                loaders = score_loaders,
                verbose = verbose
                )

        cam_exp = CAMExpScore(path=scores_path, name='CAM-exp')
        if not cam_exp.load():
            cam_exp.fit(
                    datasets = ds,
                    peepholes = cam_ph,
                    pos_loader_train = pos_loader_train,
                    neg_loaders = neg_loaders,
                    verbose = verbose
                    )
        cam_exp(
                datasets = ds,
                peepholes = cam_ph,
                pos_loader_test = pos_loader_test,
                verbose = verbose
                )

        #--------------------------------
        # Plots
        #--------------------------------
        # commented until the plots take the scores DataFrames
        '''
        plot_confidence(
                datasets = ds,
                loaders = [pos_loader_test, 'CIFAR100-C-test-c0-vgg'],
                scores = scores,
                max_score = 1.,
                path = plots_path,
                verbose = verbose
                )

        plot_calibration(
                datasets = ds,
                loaders = [pos_loader_test, 'CIFAR100-C-test-c0-vgg'],
                scores = scores,
                calib_bin = 0.1,
                path = plots_path,
                verbose = verbose
                )

        plot_ood(
                scores = scores,
                id_loaders = {
                    'MACS': pos_loader_test,
                    'MSP': pos_loader_test,
                    'ViM': pos_loader_test,
                    'DMD-a': pos_loader_test,
                    'CAM-lin': pos_loader_test,
                    'CAM-exp': pos_loader_test,
                    },
                ood_loaders = ood_loaders,
                path = plots_path,
                verbose = verbose
                )

        # plot conceptograms
        idx = [2, 5, 15, 40, 86, 150]
        plot_conceptogram(
                path = plots_path,
                name = 'macs',
                datasets = ds,
                peepholes = ph,
                loaders = [pos_loader_test],
                samples = idx,
                target_modules = target_layers,
                classes = Cifar100.get_classes(meta_path = Path(cifar_path)/'cifar-100-python/meta'),
                protoclasses = macs.proto,
                scores = scores,
                verbose = verbose,
                )

        plot_conceptogram(
                path = plots_path,
                name = 'dmd',
                datasets = ds,
                peepholes = dmd_ph,
                loaders = [pos_loader_test],
                samples = idx,
                target_modules = target_layers,
                classes = Cifar100.get_classes(meta_path = Path(cifar_path)/'cifar-100-python/meta'),
                scores = scores,
                verbose = verbose,
                )

        plot_conceptogram(
                path = plots_path,
                name = 'cam',
                datasets = ds,
                peepholes = cam_ph,
                loaders = [pos_loader_test],
                samples = idx,
                target_modules = target_layers,
                classes = Cifar100.get_classes(meta_path = Path(cifar_path)/'cifar-100-python/meta'),
                scores = scores,
                verbose = verbose,
                )
        '''
