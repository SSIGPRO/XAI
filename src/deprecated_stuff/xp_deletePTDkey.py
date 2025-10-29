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
    
    cvs_path = Path('/srv/newpenny/XAI/generated_data/corevectors/CIFAR100_ViT')
    cvs_name = 'coreavg'

    phs_path = Path.cwd()/f'../data/{name_model}/peepholes' #Path('/srv/newpenny/XAI/Peephole-Analysis/channel_wise/peepholes_on_class') #
    phs_name = 'peepholes_avg' #'peepholes_avg'
     
    #--------------------------------
    # Delete corevectors
    #--------------------------------

    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )    

    peepholes = Peepholes(
            path = phs_path,
            name = phs_name,
            device = device
            )

    with corevecs as cv, peepholes as ph:
        cv.load_only(
                loaders = [
                        'train', 'test', 'val', 
                        'test-ood-c0', 'test-ood-c1', 'test-ood-c2', 'test-ood-c3', 'test-ood-c4', 
                        'val-ood-c0', 'val-ood-c1', 'val-ood-c2', 'val-ood-c3', 'val-ood-c4',
                        'test-ood-SVHN', 'test-ood-Places365', 'test-ood-DTD',
                        'val-ood-SVHN', 'val-ood-Places365', 'val-ood-DTD'
                        ],
                verbose = True,
                mode = 'r+'
                )
        
        keys_n = []
        for ds_key, c  in cv._corevds.items():
            if 'encoder.ln' in c:
                print(c['encoder.ln'].shape)
                print(f'{ds_key} DELETE encoder.ln')
                print('BEFORE DELETE')
                print('------------------------')
                print(c)
                c.del_('encoder.ln')
                print('AFTER DELETE')
                print('------------------------')
                print(c)
            else:
                print(f'{ds_key} does not contain encoder.ln')
       