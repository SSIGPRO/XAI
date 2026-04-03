import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
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
from peepholelib.datasets.functional.inference_fns import img_classification_full as inference_fn 
from peepholelib.datasets.functional.samplers import random_subsampling # name explains
from peepholelib.datasets.functional.transforms import means, stds, wrn_cifar100_transform as transform, wrn_cifar100_augmentations as augmentation, TransformWrap

# ATK dataset
# from peepholelib.adv_atk.BIM import myBIM
# from peepholelib.adv_atk.CW import myCW
# from peepholelib.adv_atk.DeepFool import myDeepFool
# from peepholelib.adv_atk.PGD import myPGD
# from peepholelib.adv_atk.APGD import myAPGD
from peepholelib.adv_atk.AutoAttack import myAutoAttack
# from peepholelib.adv_atk.attacksDS import AttacksDS 


from configs.common import * # common configs like model we use, defualt Vgg for example, kernel avg pooling

if __name__ == "__main__":
    # print(f'{args}') 
    # use_cuda = torch.cuda.is_available()
    # device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    # print(f"Using {device} device")

    gpu_id = 5
    device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using {device} device")
    torch.cuda.set_device(device)
    name_dataset = 'CIFAR100'
    #------------------
    # Model 
    #------------------
    model = ModelWrap(
            model = Model, # change to Model(num_classes=100) for vgg16
            target_modules = target_layers,
            device = device
            )

#     model.normalize_model(mean=means[name_dataset], std= stds[name_dataset]) # move to after load checkpoint for vgg, or comment it
    
#     model_path = Path('/srv/newpenny/XAI/conceptograms/LC/WRNRobust28_10/cifar100/Linf') #WRNRobust70_16
#     model_name = 'Wang2023Better_WRN-28-10.pt' #Wang2023Better_WRN-70-16.pt

#     model.load_checkpoint(
#             path = model_path,
#             name = model_name,
#             verbose = True 
#             )
    
            
    #--------------------------------
    # Datasets 
    #--------------------------------
    # original datasets
    _dss = {
            'CIFAR100': Cifar100(
                path = cifar_path,
                # std_transform = transform,
                seed = seed
                )
            }

    _dss_samplers = {
            k: partial(
                random_subsampling, 
                perc = 0.005 # 0.005 for 50 samples
                ) for k in _dss.keys()
            }

    #######################
    # parsing datasets
    #######################
    
    # parse the original datasets into ds_path
    # Create dataset files
    atks = {
        #     'APGDf': myAPGD(
        #         model = model,
        #         targeted = True,
        #         mode = 'random'
        #         # target_class = 71
        #         ),
        #     'PGDf': myPGD(
        #         model = model,
        #         mode = 'fixed',
        #         target_class = 5
        #         ),
        #     'BIMf': myBIM(
        #         model = model,
        #         mode = 'fixed',
        #         target_class = 5
        #         ),
            'auto': myAutoAttack(
                model = model
                ),
            }    
    
    with ParsedDataset(path=ds_path) as dataset:

        dataset.parse_dataset(
                dataset_wraps = _dss,
                ds_samplers = _dss_samplers,
                keys_to_copy = ['image', 'label'],
                batch_size = bs_base,
                n_threads = 1,
                verbose = verbose
                )

        dataset.parse_inference(
                loaders = ['CIFAR100-test'],
                inference_fns = {'clean': partial(inference_fn, model=model)},
                transforms = None,
                batch_size = bs_base,
                n_threads = 1,
                verbose = verbose
                )

        dataset.parse_inference(
                loaders = ['CIFAR100-test'],
                inference_fns = atks,
                transforms = None,
                batch_size = int(bs_base * bs_atk_scale),
                n_threads = 1,
                verbose = verbose
                )

    #######################
    # creating attk dataset 
    #######################
    # atks will be saved in ds_path
#     atk_ds = AttacksDS(
#             path = ds_path,
#             )

#     atks = {
#         #     'APGDf': myAPGD(
#         #         model = model,
#         #         targeted = True,
#         #         mode = 'random'
#         #         # target_class = 71
#         #         ),
#         #     'PGDf': myPGD(
#         #         model = model,
#         #         mode = 'fixed',
#         #         target_class = 5
#         #         ),
#         #     'BIMf': myBIM(
#         #         model = model,
#         #         mode = 'fixed',
#         #         target_class = 5
#         #         ),
#             'auto': myAutoAttack(
#                 model = model
#                 ),
#             }
    
#     # Apply attks to ds
#     with dataset as ds:
#         ds.load_only(
#                 loaders = ['CIFAR100-test'],
#                 verbose = verbose
#                 )

#         # Apply attks to ds
#         with atk_ds:
#             atk_ds.apply_attacks(
#                     dataset = ds,
#                     loaders = ['CIFAR100-test'],
#                     attacks = atks,
#                     batch_size = int(bs_base*bs_atk_scale),
#                     verbose = verbose 
#                     )