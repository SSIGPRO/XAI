import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# our stuff
from peepholelib.models.model_wrap import ModelWrap 

from peepholelib.datasets.cifar100 import Cifar100
from peepholelib.datasets.functional.parsers import from_dataset
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.datasets.functional.transforms import vgg16_cifar100 as ds_transform 
from peepholelib.datasets.cifarC import CifarC


# ATK dataset
from peepholelib.adv_atk.BIM import myBIM
from peepholelib.adv_atk.CW import myCW
from peepholelib.adv_atk.DeepFool import myDeepFool
from peepholelib.adv_atk.PGD import myPGD
from peepholelib.adv_atk.attacksDS import AttacksDS 


# other stuff
import torchvision
from torchvision.models import vgg16
from cuda_selector import auto_cuda
import torch

    
if __name__ == "__main__":
    cifar_path = '/srv/newpenny/dataset/CIFAR100'
    cifarc_path = '/srv/newpenny/dataset/CIFAR-100-C'
    ds_path = Path('/srv/newpenny/XAI/CN/vgg_data/cifar100')


    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    verbose = True 
    seed = 29
    bs = 256+128
    n_threads = 1

    #--------------------------------
    # Model 
    #--------------------------------
    
    nn = vgg16()
    n_classes = 100
    model = ModelWrap(
            model = nn,
            device = device
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


    #######################
    # parsing datasets
    #######################
    
    # parse the original datasets into ds_path
    ParsedDataset.parse_ds(
            save_path = ds_path,
            model = model,
            datasets = _dss,
            ds_parsers = _dss_parsers, 
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
            path = ds_path,
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
                loaders = ['CIFAR100-test', 'CIFAR100-val'],
                verbose = verbose 
                )

        atk_ds.apply_attacks(
                dataset = ds,
                loaders = ['CIFAR100-test', 'CIFAR100-val'],
                attacks = atks,
                batch_size = bs,
                verbose = verbose 
                )

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

