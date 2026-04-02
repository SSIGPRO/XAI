import sys
from pathlib import Path as Path

from treelite import Model
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from functools import partial

# torch stuff
import torch
from torchvision.models import vgg16 as Model
from cuda_selector import auto_cuda

###### Our stuff

# Model
from peepholelib.models.model_wrap import ModelWrap 

# datasets
from peepholelib.datasets.cifar100 import Cifar100
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.datasets.functional.inference_fns import img_classification_full as inference_fn 
from peepholelib.datasets.functional.transforms import means, stds, vgg16_transform as transform
from peepholelib.datasets.functional.samplers import random_subsampling 
from peepholelib.coreVectors.dimReduction.randomProjection import RandomProjection
from peepholelib.peepholes.ClassDependentClassifiers.tgmm import GMM as tGMM 

if __name__ == "__main__":
        use_cuda = torch.cuda.is_available()
        device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
        print(f"Using {device} device")

        #--------------------------------
        # Directories definitions
        #--------------------------------
        cifar_path = '/srv/newpenny/dataset/CIFAR100'
        main_path = Path.cwd()/'../data/RPR'
        ds_path = main_path/'datasets'

        cvs_path = main_path/'corevectors'
        cvs_name = 'randomvectors'

        drill_path = main_path/'drillers_parallel_0'
        drill_name = 'CDclassifier'

        # Hyperparameters
        seed = 29
        bs = 2**11
        n_threads = 1

        model_path = '/srv/newpenny/XAI/models'
        model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'

        random_path = main_path/'randomprojections'
        
        verbose = True 
        
        #--------------------------------
        # Model 
        #--------------------------------
        n_classes = 100
        output_layer = 'classifier.6'
        target_layers = [
                'model.features.26',
                'model.features.28',
                'model.classifier.0',
                ]


        model = ModelWrap(
                model = Model(),
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

        model.normalize_model(mean=means['CIFAR100'], std= stds['CIFAR100'])

        model.set_target_modules(
                target_modules = target_layers,
                verbose = verbose
                )
        print(model._target_modules)
                                                
        #--------------------------------
        # Datasets 
        #--------------------------------
        # original datasets
        _dss = {
                'CIFAR100': Cifar100(
                        path = cifar_path,
                        std_transform = transform,
                        seed = seed
                        ),
                }
        '''
        _dss_samplers = {
                k: partial(
                        random_subsampling, 
                        perc = 0.3
                        ) for k in _dss.keys()
                }
        '''
        #######################
        # parsing datasets
        #######################
        dataset = ParsedDataset.parse_ds(
                path = ds_path,
                dataset_wraps = _dss,
                # ds_samplers = _dss_samplers, 
                keys_to_copy = ['image', 'label'],
                inference_fn = partial(inference_fn, model=model), # comment for fine tuning the model
                batch_size = bs,
                n_threads = 1,
                verbose = verbose
                ) 
        
        #--------------------------------
        # Corevectors 
        #--------------------------------
        corevecs = CoreVectors(
                path = cvs_path,
                name = cvs_name,
                model = model,
                )
        
        loaders = [
                'CIFAR100-train',
                'CIFAR100-val',
                'CIFAR100-test',
                ]
        
        cv_dims = {_l: 100 for _l in target_layers} 

        cv_dims[target_layers[-1]] = 50

        n_classifiers = {_l: 3 for _l in target_layers}

        Reducer = RandomProjection
        reducers = {}
        drillers = {}
        
        with corevecs as cv, dataset as ds:
                
                ds.load_only(
                        loaders = loaders,
                        verbose = verbose
                        )
                for _layer in target_layers:

                        reducers[_layer] = Reducer(
                                model = model,
                                layer = _layer,
                                datasets = ds,
                                path = random_path,
                                cv_dim = cv_dims[_layer],
                                seed = seed
                                )
                        
                        drillers[_layer] = tGMM(
                                path = drill_path,
                                name = f'{drill_name}.GMM.{_layer}.{n_classes}.{cv_dims[_layer]}.{n_classifiers[_layer]}',
                                target_module = _layer,
                                nl_classifier = n_classifiers[_layer],
                                nl_model = n_classes,
                                n_features = cv_dims[_layer],
                                reducer = reducers[_layer],
                                device = device
                                )

                # computing the corevectors
                cv.get_coreVectors(
                        datasets = ds,
                        reducers = reducers,
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
                
        with corevecs as cv, dataset as ds:
                
                ds.load_only(
                        loaders = loaders,
                        verbose = verbose
                        )

                cv.load_only(
                        loaders = loaders,
                        verbose = verbose
                        )
                
                for _layer in target_layers:

                        print(f'Preparing driller for layer {_layer}')
                        if (drillers[_layer]._clas_path).exists():
                                drillers[_layer].load()
                        else:
                                drillers[_layer].fit(
                                        datasets = ds, 
                                        corevectors = cv, 
                                        loader = f'CIFAR100-train',
                                        verbose=verbose,
                                        max_workers = 10
                                )
                                drillers[_layer].save()