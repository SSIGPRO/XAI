import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from functools import partial

# Our stuff
from peepholelib.models.svd_fns import linear_svd, conv2d_toeplitz_svd
from peepholelib.coreVectors.dimReduction.svds import linear_svd_projection, conv2d_toeplitz_svd_projection

# torch stuff
import torch
from torchvision.models import vgg16 as Model
from torchvision.models import VGG16_Weights 
from cuda_selector import auto_cuda

use_cuda = torch.cuda.is_available()
device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
print(f"Using {device} device")

#--------------------------------
# Directories definitions
#--------------------------------
#ds_path = '/srv/newpenny/XAI/generated_data/parsed_datasets/VGG16'
ds_path = Path.cwd()/'../../data/CLIP/parsed_datasets/ImageNet_VGG16'

# model parameters
bs = 2**10
n_threads = 1
n_classes = 1000

#cvs_path = Path('/srv/newpenny/XAI/generated_data/corevectors/CIFAR100_VGG16')
emb_path = Path.cwd()/'../../data/CLIP/CLIP_embeds/ImageNet_VGG16'
emb_name = 'CLIP_embeds'

verbose = True 

save_input = True
save_output = False

loaders = [
        'ImageNet-train',
        'ImageNet-val',
        # 'CIFAR100-test',
        # 'CIFAR100-C-val-c0',
        # 'CIFAR100-C-test-c0',
        # 'CIFAR100-C-val-c1',
        # 'CIFAR100-C-test-c1',
        # 'CIFAR100-C-val-c2',
        # 'CIFAR100-C-test-c2',
        # 'CIFAR100-C-val-c3',
        # 'CIFAR100-C-test-c3',
        # 'CIFAR100-C-val-c4',
        # 'CIFAR100-C-test-c4',
        # 'SVHN-val',
        # 'SVHN-test',
        # 'Places365-val',
        # 'Places365-test',
        # 'CW-CIFAR100-val',
        # 'CW-CIFAR100-test',
        # 'BIM-CIFAR100-val',
        # 'BIM-CIFAR100-test',
        # 'DF-CIFAR100-val',
        # 'DF-CIFAR100-test',
        # 'PGD-CIFAR100-val',
        # 'PGD-CIFAR100-test',
        ]

dataset_name = 'imagenet'
weights = VGG16_Weights.DEFAULT
verbose = True 

output_layer = 'classifier.6'

model = Model(weights=weights)
    
