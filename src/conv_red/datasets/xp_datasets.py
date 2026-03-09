import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/conv_red').as_posix())

# python stuff
from functools import partial
from filelock import FileLock

# torch stuff
import torch
from cuda_selector import auto_cuda

# Peephoelib stuff
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.datasets.cifar100 import Cifar100
from peepholelib.datasets.cifarC import CifarC
from peepholelib.datasets.SVHN import SVHN 
from peepholelib.datasets.Places import Places 
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.datasets.functional.samplers import random_subsampling 
from peepholelib.datasets.functional.inference_fns import img_classification_full as img_cls_inf, img_classification_atks as img_cls_atk_inf 

# ATK dataset
from peepholelib.adv_atk.BIM import myBIM
from peepholelib.adv_atk.CW import myCW
from peepholelib.adv_atk.DeepFool import myDeepFool
from peepholelib.adv_atk.PGD import myPGD

from configs.common import *

lock_file = '../locks/datasets.cuda.lock'
if __name__ == "__main__":
    print(f'{args}') 
    lock = FileLock(lock_file)
    with lock.acquire(timeout=-1):
        use_cuda = torch.cuda.is_available()
        device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
        print(f"Using {device} device")

        #------------------
        # Model 
        #------------------
        model = ModelWrap(
                model = Model(),
                target_modules = target_layers,
                device = device
                )
    
    model.update_output(
            output_layer = output_layer, 
            to_n_classes = n_classes,
            overwrite = True 
            )
                                            
    model.load_checkpoint(
            path = model_path,
            name = model_name,
            verbose = True 
            )

    model.prepend_normalizer(
            mean = normalization_mean,
            std = normalization_std
            )

    #--------------------------------
    # Datasets 
    #--------------------------------
    # original datasets
    _dss = {
            'CIFAR100': Cifar100(
                path = cifar_path,
                seed = seed
                ),
            'CIFARC': CifarC(
                path = cifarc_path,
                seed = seed
                ),
            'SVHN': SVHN(
                path = svhn_path,
                seed = seed
                ),
            'Places': Places(
                path = places_path,
                seed = seed
                )
            }

    _dss_samplers = {
            k: partial(
                random_subsampling, 
                perc = 0.5
                ) for k in _dss.keys()
            }

    #######################
    # parsing datasets
    #######################
    
    # Instantiate DSs 
    dataset = ParsedDataset(
            path = ds_path,
            )

    # instantiate atks
    atks = {
            'BIM-'+args.model: myBIM(
                model = model,
                ),
            'CW-'+args.model: myCW(
                model = model,
                max_steps = 100,
                ),
            }

    # create inference functions for each atk
    atks_inf_fns = {
            atk_name: partial(
                img_cls_atk_inf,
                attack = atk,
                label_key = 'label'
                ) for atk_name, atk in atks.items()
            }

    with dataset as ds:
        ds.parse_dataset(
                dataset_wraps = _dss,
                ds_samplers = _dss_samplers, 
                keys_to_copy = ['image', 'label'],
                batch_size = bs_base,
                n_threads = n_threads,
                verbose = verbose
                )

        ds.parse_inference(
                inference_fns = {args.model: partial(img_cls_inf, model=model)},
                transforms = transforms,
                batch_size = bs_base,
                n_threads = n_threads,
                verbose = verbose
                )

        # Apply attacks
        ds.parse_inference(
                loaders = ['CIFAR100-val', 'CIFAR100-test'],
                inference_fns = atks_inf_fns, 
                transforms = transforms,
                batch_size = int(bs_base*bs_atk_scale),
                n_threads = n_threads,
                verbose = verbose 
                )
