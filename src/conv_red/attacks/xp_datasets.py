import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/conv_red').as_posix())

# python stuff
from functools import partial

# torch stuff
import torch
from cuda_selector import auto_cuda

# Our stuff
from peepholelib.models.model_wrap import ModelWrap 

from peepholelib.datasets.cifar100 import Cifar100
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.datasets.functional.parsers import from_dataset
from peepholelib.datasets.functional.samplers import random_subsampling 


# ATK dataset
from peepholelib.adv_atk.BIM import myBIM
from peepholelib.adv_atk.CW import myCW
from peepholelib.adv_atk.DeepFool import myDeepFool
from peepholelib.adv_atk.PGD import myPGD
from peepholelib.adv_atk.attacksDS import AttacksDS 

from configs.common import *

if __name__ == "__main__":
    print(f'{args}') 
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
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

    #--------------------------------
    # Datasets 
    #--------------------------------
    # original datasets
    _dss = {
            'CIFAR100': Cifar100(
                path = cifar_path,
                transform = transform,
                seed = seed
                )
            }

    _dss_parsers = {
            'CIFAR100': from_dataset
            }

    _dss_samplers = {
            k: partial(
                random_subsampling, 
                perc = 0.005
                ) for k in _dss.keys()
            }

    #######################
    # parsing datasets
    #######################
    
    # parse the original datasets into ds_path
    ds = ParsedDataset.parse_ds(
            save_path = ds_path,
            model = model,
            datasets = _dss,
            ds_parsers = _dss_parsers, 
            ds_samplers = _dss_samplers, 
            batch_size = bs_base,
            n_threads = n_threads,
            verbose = verbose
            )

    # close PTDs
    ds.__exit__(None, None, None)

    #######################
    # creating attk dataset 
    #######################
    # atks will be saved in ds_path
    atk_ds = AttacksDS(
            path = ds_path,
            )

    atks = {
            'CW': myCW(
                model = model,
                max_steps = 100,
                ),
            'BIM': myBIM(
                model = model,
                ),
            #'DF': myDeepFool(
            #    model = model,
            #    ),
            #'PGD': myPGD(
            #    model = model,
            #    ),
            }
    
    # Apply attks to ds
    with ds, atk_ds:
        ds.load_only(
                loaders = ['CIFAR100-val', 'CIFAR100-test'],
                verbose = verbose 
                )

        atk_ds.apply_attacks(
                dataset = ds,
                loaders = ['CIFAR100-val', 'CIFAR100-test'],
                attacks = atks,
                batch_size = int(bs_base*bs_atk_scale),
                verbose = verbose 
                )
