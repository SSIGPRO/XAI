import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# our stuff
from peepholelib.models.model_wrap import ModelWrap 

from peepholelib.datasets.imagenet import ImageNet
from peepholelib.datasets.functional.parsers import from_dataset
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.datasets.functional.transforms import mobilenet_imagenet as ds_transform 


# ATK dataset
from peepholelib.adv_atk.BIM import myBIM
from peepholelib.adv_atk.CW import myCW
from peepholelib.adv_atk.DeepFool import myDeepFool
from peepholelib.adv_atk.PGD import myPGD
from peepholelib.adv_atk.attacksDS import AttacksDS 


# other stuff
import torchvision
from cuda_selector import auto_cuda
import torch

    
if __name__ == "__main__":
    imagenet_path = '/srv/newpenny/dataset/ImageNet_torchvision'
    ds_path = Path('/srv/newpenny/XAI/CN/mobilenet_data/imagenet')

    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'CN_model=mobilenet_v2_dataset=CIFAR100_optim=Adam_scheduler=RoP_lr=0.001_factor=0.1_patience=5.pth'

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
    
    nn = torchvision.models.mobilenet_v2(pretrained=True)
    n_classes = 1000
    model = ModelWrap(
            model = nn,
            device = device
            )
                                            
                                
    #--------------------------------
    # Datasets 
    #--------------------------------
    # original datasets
    dss = {
            'ImageNet': ImageNet(
                path = imagenet_path,
                transform = ds_transform,
                seed = seed
                ),

            }

    dss_parsers = {
            'ImageNet': from_dataset,
            }

    #######################
    # parsing datasets
    #######################
    
    # parse the original datasets into ds_path
    ParsedDataset.parse_ds(
            save_path = ds_path,
            model = model,
            datasets = dss,
            ds_parsers = dss_parsers, 
            batch_size = bs,
            n_threads = n_threads,
            verbose = verbose,
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
                loaders = ['ImageNet-val'],
                verbose = verbose 
                )

        atk_ds.apply_attacks(
                dataset = ds,
                loaders = ['ImageNet-val'],
                attacks = atks,
                batch_size = bs,
                verbose = verbose 
                )

