import sys
from pathlib import Path as Path # why path as path? shouldn't make a difference anyways
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix()) # check these folders
sys.path.insert(0, (Path.home()/'repos/XAI/src/conv_red').as_posix())

# python stuff
from functools import partial

# torch stuff
import torch # I got an error here
from cuda_selector import auto_cuda # select optimal cuda I assume

# Our stuff
from peepholelib.models.model_wrap import ModelWrap 
# this one is for normalization, storing activations  layer by layer = interpretability of AI?
from peepholelib.datasets.cifar100 import Cifar100 
# from my understanding includes superclasses but overall same
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.datasets.functional.samplers import random_subsampling # name explains


# ATK dataset
from peepholelib.adv_atk.BIM import myBIM
from peepholelib.adv_atk.CW import myCW
from peepholelib.adv_atk.DeepFool import myDeepFool
from peepholelib.adv_atk.PGD import myPGD
from peepholelib.adv_atk.APGD import myAPGD
from peepholelib.adv_atk.attacksDS import AttacksDS 

from configs.common import * # common configs like model we use, defualt Vgg for example, kernel avg pooling

if __name__ == "__main__":
    # print(f'{args}') 
    # use_cuda = torch.cuda.is_available()
    # device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    # print(f"Using {device} device")

    gpu_id = 3
    device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using {device} device")
    torch.cuda.set_device(device)

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

    _dss_samplers = {
            k: partial(
                random_subsampling, 
                perc = 1 # 0.005 for 50 samples
                ) for k in _dss.keys()
            }

    #######################
    # parsing datasets
    #######################
    
    # parse the original datasets into ds_path
    # Create dataset files
    ds = ParsedDataset.create_ds(
        path=ds_path,
        dataset_wraps=_dss,
        batch_size=bs_base,
        n_threads=n_threads,
        ds_samplers=_dss_samplers,
        verbose=verbose
    )

    # Parse
    with ds:
        ds.load_only(
            loaders=loaders,
            mode='r+',
            verbose=verbose
        )
        ds.parse_ds(
            model=model,
            loaders=loaders,
            batch_size=bs_base,
            verbose=verbose
        )

    #######################
    # creating attk dataset 
    #######################
    # atks will be saved in ds_path
    atk_ds = AttacksDS(
            path = ds_path,
            )

    atks = {
        #     'CW': myCW(
        #         model = model,
        #         max_steps = 500,
        #         mode = 'least-likely'
        #         ),
        #     'BIM': myBIM(
        #         model = model,
        #         mode = 'least-likely',
        #         steps = 500
        #         ),
        #     'DF': myDeepFool(
        #         model = model,
        #         ),
        #     'PGD': myPGD(
        #         model = model,
        #         mode = 'least-likely',
        #         steps = 500
        #         ),
            # 'APGDu': myAPGD(
            #     model = model,
            #     targeted = False
            #     ),            
            # 'APGDr': myAPGD(
            #     model = model,
            #     targeted = True,
            #     target_mode="random"
            #     ),
            # 'APGDl': myAPGD(
            #     model = model,
            #     targeted = True,
            #     target_mode="least_likely"
            #     ),
            # 'APGDf': myAPGD(
            #     model = model,
            #     targeted = True,
            #     target_mode="fixed",
            #     target_class = 5
            #     ),
            # 'APGD2u': myAPGD(
            #     model = model,
            #     norm = 'L2',
            #     eps = 2,
            #     targeted = False
            #     ),            
            # 'APGD2r': myAPGD(
            #     model = model,
            #     targeted = True,
            #     norm = 'L2',
            #     eps = 2,
            #     target_mode="random"
            #     ),
            # 'APGD2l': myAPGD(
            #     model = model,
            #     targeted = True,
            #     norm = 'L2',
            #     eps = 2,
            #     target_mode="least_likely"
            #     ),
            # 'APGD2f': myAPGD(
            #     model = model,
            #     targeted = True,
            #     target_mode="fixed",
            #     eps = 2,
            #     norm = "L2",
            #     target_class = 5
            #     ),
            # 'PGDl': myPGD(
            #     model = model,
            #     mode = 'least-likely'
            #     ),
            # 'PGDr': myPGD(
            #     model = model,
            #     mode = 'random'
            #     ),
            # 'PGDf': myPGD(
            #     model = model,
            #     mode = 'fixed',
            #     target_class = 5
            #     ),
            # 'BIMl': myBIM(
            #     model = model,
            #     mode = 'least-likely'
            #     ),
            # 'BIMr': myBIM(
            #     model = model,
            #     mode = 'random'
            #     ),
            # 'BIMf': myBIM(
            #     model = model,
            #     mode = 'fixed'
            #     ),
            # 'DF': myDeepFool(
            #     model = model,
            #     ),
            'APGDf': myAPGD(
                model = model,
                targeted = True,
                target_mode="fixed",
                target_class = 5
                ),
            'PGDf': myPGD(
                model = model,
                mode = 'fixed',
                target_class = 5
                ),
            'BIMf': myBIM(
                model = model,
                mode = 'fixed',
                target_class = 5
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
                batch_size = int(bs_base*bs_atk_scale),
                verbose = verbose 
                )
