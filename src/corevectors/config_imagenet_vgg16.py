import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from functools import partial

# Our stuff
from peepholelib.models.svd_fns import linear_svd, conv2d_toeplitz_svd
from peepholelib.coreVectors.dimReduction.svds import linear_svd_projection
from peepholelib.coreVectors.dimReduction.avgPooling import ChannelWiseMean_conv

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

# svds_path = Path('/srv/newpenny/XAI/generated_data/svds/ImageNet_vgg16')
svds_path = Path.cwd()/'../../data/CLIP/svds/ImageNet_VGG16'
svds_name = 'svds' 

#cvs_path = Path('/srv/newpenny/XAI/generated_data/corevectors/CIFAR100_VGG16')
cvs_path = Path.cwd()/'../../data/CLIP/corevectors/ImageNet_VGG16'
cvs_name = 'corevectors'

verbose = True 

save_input = True
save_output = False

# Peepholelib
target_layers_svd = [
        'model.classifier.0',
        'model.classifier.3',
        'model.classifier.6',
        ]

target_layers = [
        # 'model.features.7',
        # 'model.features.10',
        # 'model.features.12',
        # 'model.features.14',
        # 'model.features.17',
        'model.features.19',
        'model.features.21',
        'model.features.24',
        'model.features.26',
        'model.features.28',
        'model.classifier.0',
        'model.classifier.3',
        'model.classifier.6',
        ]

svd_rank = 300 
output_layer = 'classifier.6'

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

#--------------------------------
# SVDs 
#--------------------------------
svd_fns = {}
for _layer in target_layers_svd:
    if 'features' in _layer:
        svd_fns[_layer] = partial(
                conv2d_toeplitz_svd, 
                rank = svd_rank, 
                channel_wise = False,
                device = device,
                )
    elif 'classifier' in _layer:
        svd_fns[_layer] = partial(
                linear_svd,
                rank = svd_rank,
                device = device,
                )
        
# define a dimensionality reduction function for each layer
reduction_fns = {}
for _layer in target_layers:
    if 'features' in _layer:
        reduction_fns[_layer] = ChannelWiseMean_conv
        # reduction_fns[_layer] = partial(
        #         conv2d_toeplitz_svd_projection, 
        #         use_s = True,
        #         device=device
        #         )
    elif 'classifier' in _layer:
        reduction_fns[_layer] = partial(
                linear_svd_projection,
                use_s = True,
                device=device
                )    

dataset_name = 'imagenet'
weights = VGG16_Weights.DEFAULT
verbose = True 

output_layer = 'classifier.6'

model = Model(weights=weights)
    
