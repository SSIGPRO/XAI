import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
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
from peepholelib.datasets.MNIST import MNIST 
from peepholelib.datasets.textures import Textures 
from peepholelib.datasets.parsedDataset import ParsedDataset 

from peepholelib.datasets.functional.inference_fns import img_classification_full as img_cls_inf, img_classification_atks as img_cls_atk_inf 
from peepholelib.datasets.functional.transforms import TransformWrap 
from peepholelib.datasets.functional.transforms import vgg16_transform as ds_transform 
from peepholelib.datasets.functional.samplers import random_subsampling 

# ATK dataset
from peepholelib.adv_atk.BIM import myBIM
from peepholelib.adv_atk.CW import myCW
from peepholelib.adv_atk.DeepFool import myDeepFool
from peepholelib.adv_atk.PGD import myPGD

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    cifar_path = '/srv/newpenny/dataset/CIFAR100'
    cifarc_path = '/srv/newpenny/dataset/CIFAR-100-C'
    mnist_path = '/srv/newpenny/dataset/MNIST'
    textures_path = '/srv/newpenny/dataset/DTD'
    ds_path = Path.cwd()/'../data/datasets'
    atk_path = '../data/attacks/'

    # model parameters
    seed = 29
    
    bs = 2**8
    n_threads = 1

    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
     
    verbose = True 
    
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
            #'MNIST': MNIST(
            #    path = mnist_path,
            #    seed = seed
            #    ),
            'Textures': Textures(
                path = textures_path,
                seed = seed
                )

            }

    _dss_samplers = {
            k: partial(
                random_subsampling, 
                perc = 0.1
                ) for k in _dss.keys()
            }

    loaders = [
            'CIFAR100-train',
            'CIFAR100-test',
            'CIFAR100-val',
            'CIFAR100-C-train-c0',
            'CIFAR100-C-val-c0',
            'CIFAR100-C-test-c0',
            'CIFAR100-C-train-c1',
            'CIFAR100-C-val-c1',
            'CIFAR100-C-test-c1',
            'CIFAR100-C-train-c2',
            'CIFAR100-C-val-c2',
            'CIFAR100-C-test-c2',
            'CIFAR100-C-train-c3',
            'CIFAR100-C-val-c3',
            'CIFAR100-C-test-c3',
            'CIFAR100-C-train-c4',
            'CIFAR100-C-val-c4',
            'CIFAR100-C-test-c4',
            ]

    _transforms = {
            k: TransformWrap(transform=ds_transform, input_key='image') for k in loaders 
            }

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

    # create inference functions for each atk
    atks_inf_fns = {
            atk_name: partial(
                img_cls_atk_inf,
                attack = atk,
                label_key = 'label'
                ) for atk_name, atk in atks.items()
            }

    #######################
    # parsing datasets
    #######################
    dataset = ParsedDataset(
            path = ds_path
            )

    with dataset as ds:
        ds.parse_dataset(
                dataset_wraps = _dss,
                ds_samplers = _dss_samplers, 
                keys_to_copy = ['image', 'label'],
                batch_size = bs,
                n_threads = 1,
                verbose = verbose
                ) 

        ds.parse_inference(
                inference_fns = {'vgg': partial(img_cls_inf, model=model)},
                transforms = _transforms,
                batch_size = bs,
                n_threads = 1,
                verbose = verbose
                )

        ds.parse_inference(
                loaders = ['CIFAR100-test-vgg'],
                inference_fns = atks_inf_fns, 
                transforms = _transforms,
                batch_size = bs,
                verbose = verbose 
                )

