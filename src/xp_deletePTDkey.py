import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial
from matplotlib import pyplot as plt

# Our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.models.viz import viz_singular_values

from peepholelib.coreVectors.coreVectors import CoreVectors 
from peepholelib.peepholes.peepholes import Peepholes

# torch stuff
import torch
from torchvision.models import vgg16, VGG16_Weights, vit_b_16
from cuda_selector import auto_cuda

if __name__ == "__main__":
    # use_cuda = torch.cuda.is_available()
    # device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    # print(f"Using {device} device")
    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 1
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    # model parameters
    name_model = 'vgg16' # 'ViT' 
    dataset = 'ImageNet' 
    
    cvs_path = Path.cwd()/f'../data/{dataset}/corevectors'
    cvs_name = 'corevectors'
     
    #--------------------------------
    # Delete corevectors
    #--------------------------------

    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )    
    

    target_layers = [
           'features.28'
            ]

    with corevecs as cv:
        cv.load_only(
                loaders = [
                        'train', 
                        #'test', 
                        'val'
                        ],
                verbose = True,
                mode = 'r+'
                )
        # ph.load_only(
        #         loaders = ['train', 'test', 'val'],
        #         verbose = True,
        #         mode = 'r+'
        #         )
        print('------------------------')
        print(cv._corevds['train'],cv._corevds['val'])
        cv._corevds['train'].del_('features.28')
        print(cv._corevds['train'],cv._corevds['val'])
        quit()
        
        for layer in target_layers:
            print(f'DELETE {layer}')
            print('BEFORE DELETE')
            print('------------------------')
            print(cv._corevds['val'])
            cv._corevds['val'].del_(layer)
            print('AFTER DELETE')
            print('------------------------')
            print(cv._corevds['val'])
        # print(ph._phs['train'].del_('output'))
        # print(ph._phs['train'])
        print('------------------------')
       