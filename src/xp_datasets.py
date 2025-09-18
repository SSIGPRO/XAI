import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial
from matplotlib import pyplot as plt

# torch stuff
import torch
from torchvision.models import vgg16
from cuda_selector import auto_cuda

###### Our stuff

# Model
from peepholelib.models.model_wrap import ModelWrap 

# datasets
from peepholelib.datasets.cifar100 import Cifar100
from peepholelib.datasets.cifarC import CifarC
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.datasets.functional.parsers import from_dataset
from peepholelib.datasets.functional.transforms import vgg16_cifar100 as ds_transform 
from peepholelib.datasets.functional.samplers import random_subsampling 

# ATK dataset
from peepholelib.adv_atk.BIM import myBIM
from peepholelib.adv_atk.CW import myCW
from peepholelib.adv_atk.DeepFool import myDeepFool
from peepholelib.adv_atk.PGD import myPGD
from peepholelib.adv_atk.attacksDS import AttacksDS 


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    cifar_path = '/srv/newpenny/dataset/CIFAR100'
    cifarc_path = '/srv/newpenny/dataset/CIFAR-100-C'
    ds_path = Path.cwd()/'../data/datasets'
    atk_path = '../data/attacks/'

    # model parameters
    seed = 29
    
    # bs = 512+256+128 # BIM
    bs = 256+128 # CW
    n_threads = 1

    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
     
    verbose = True 
    
    # Peepholelib
    target_layers = [
            'features.24',
            'features.26',
            'features.28',
            'classifier.0',
            'classifier.3',
            'classifier.6',
            ]
    
    #--------------------------------
    # Model 
    #--------------------------------
    
    nn = vgg16()
    n_classes = 100#len(ds.get_classes()) 
    model = ModelWrap(
            model = nn,
            device = device
            )
                                            
    model.update_output(
            output_layer = 'classifier.6', 
            to_n_classes = n_classes,
            overwrite = True 
            )
                                            
    model.load_checkpoint(
            name = model_name,
            path = model_dir,
            verbose = verbose
            )
                                            
    model.set_target_modules(
            target_modules = target_layers,
            verbose = verbose
            )

    #--------------------------------
    # Datasets 
    #--------------------------------
    # original datasets
    _dss = {
            'CIFAR100': Cifar100(
                path = cifar_path,
                transform = ds_transform,
                seed = seed
                ),
            'CIFARC': CifarC(
                path = cifarc_path,
                seed = seed
                )
            }

    _dss_parsers = {
            'CIFAR100': from_dataset,
            'CIFARC': from_dataset,
            }

    _dss_samplers = {
            k: partial(
                random_subsampling, 
                perc = 0.025
                ) for k in _dss.keys()
            }

    #######################
    # parsing datasets
    #######################
    
    # parse the original datasets into ds_path
    ParsedDataset.parse_ds(
            save_path = ds_path,
            model = model,
            datasets = _dss,
            ds_parsers = _dss_parsers, 
            ds_samplers = _dss_samplers, 
            batch_size = bs,
            n_threads = n_threads,
            verbose = verbose
            )
    
    #######################
    # creating attk dataset 
    #######################

    # create a DatasetBase object for the parsed dataset
    ds = ParsedDataset(
            path = ds_path,
            )

    # atks will be saved in atk_path
    atk_ds = AttacksDS(
            path = atk_path,
            )

    atks = {
            'CW': myCW(
                model = model,
                max_steps = 10,
                ),
            'BIM': myBIM(
                model = model,
                ),
            'DF': myDeepFool(
                model = model,
                ),
            'PGD': myPGD(
                model = model,
                ),
            }
    
    # Apply attks to ds
    with ds, atk_ds:
        ds.load_only(
                loaders = ['CIFAR100-test'],
                verbose = verbose 
                )

        atk_ds.apply_attacks(
                dataset = ds,
                loaders = ['CIFAR100-test'],
                attacks = atks,
                batch_size = bs,
                verbose = verbose 
                )

    #######################
    # lazy stacking 
    #######################
    with ds, atk_ds:
        ds.load_only(
                loaders = ['CIFAR100-test', 'CIFAR100-C-test-c0'],
                verbose = verbose 
                )

        atk_ds.load_only(
                loaders = ['BIM-CIFAR100-test', 'CW-CIFAR100-test', 'DF-CIFAR100-test', 'PGD-CIFAR100-test'],
                verbose = verbose 
                )

        ds.lazy_stack(others = [atk_ds])
        
        from matplotlib import pyplot as plt
        for k, v in ds._dss.items():
           plt.figure()
           plt.imshow(v['image'][4].squeeze(dim=0).permute(1,2,0))
           plt.title(k)
           plt.savefig(f'./temp_plots/{k}.png')
           plt.close()
