import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff

# torch stuff
import torch
from cuda_selector import auto_cuda

###### Our stuff
# Model
from peepholelib.models.model_wrap import ModelWrap 

# datasets
from peepholelib.datasets.parsedDataset import ParsedDataset 

# ATK dataset
from peepholelib.adv_atk.BIM import myBIM
from peepholelib.adv_atk.CW import myCW
from peepholelib.adv_atk.DeepFool import myDeepFool
from peepholelib.adv_atk.PGD import myPGD
from peepholelib.adv_atk.attacksDS import AttacksDS 

if sys.argv[1] == 'vgg_cifar100':
    from config_cifar100_vgg16 import *
elif sys.argv[1] == 'vit_cifar100':
    from config_cifar100_ViT import *
else:
    raise RuntimeError('Select a configuration by runing \'python xp_get_corevectors.py <vgg|vit>\'')

if __name__ == "__main__":
    #--------------------------------
    # Directories definitions
    #--------------------------------
    # overwrite bs 
    bs = 512 

    #--------------------------------
    # Model 
    #--------------------------------
    model = ModelWrap(
            model = Model(),
            device = device
            )
                                            
    model.update_output(
            output_layer = output_layer, 
            to_n_classes = 100,
            overwrite = True 
            )
                                            
    model.load_checkpoint(
            name = model_name,
            path = model_dir,
            verbose = verbose
            )
                                            
    #--------------------------------
    # Creating attk dataset 
    #--------------------------------
    
    # create a DatasetBase object for the parsed dataset
    ds = ParsedDataset(
            path = ds_path,
            )

    # atks will be saved in ds_path
    atk_ds = AttacksDS(
            path = ds_path,
            )

    atks = {
            'CW': myCW(
                model = model,
                max_steps = 1000,
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
                loaders = ['CIFAR100-val', 'CIFAR100-test'],
                verbose = verbose 
                )

        atk_ds.apply_attacks(
                dataset = ds,
                loaders = ['CIFAR100-val', 'CIFAR100-test'],
                attacks = atks,
                batch_size = bs,
                verbose = verbose 
                )

