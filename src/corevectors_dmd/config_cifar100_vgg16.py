import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# Our stuff
from peepholelib.coreVectors.dimReduction.avgPooling import ChannelWiseMean_conv as parser_act
from peepholelib.coreVectors.dimReduction.utils import activation_base

# torch stuff
import torch
from torchvision.models import vgg16 as Model
from cuda_selector import auto_cuda

use_cuda = torch.cuda.is_available()
device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
print(f"Using {device} device")

#--------------------------------
# Directories definitions
#--------------------------------
ds_path = '/srv/newpenny/XAI/generated_data/TPAMI/parsed_datasets/CIFAR100_VGG16'
#ds_path = Path.cwd()/'../../data/parsed_datasets/CIFAR100_VGG16'

# model parameters
bs = 2**9
n_threads = 1
n_classes = 100

model_dir = '/srv/newpenny/XAI/models'
model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'

cvs_path = Path('/srv/newpenny/XAI/generated_data/TPAMI/corevectors/CIFAR100_VGG16')
#cvs_path = Path.cwd()/'../../data/corevectors/CIFAR100_VGG16'
cvs_name = 'coreavg'

verbose = True 

save_input = False
save_output = True

# Peepholelib
target_layers = [
           'features.2',
           'features.7',
           'features.14',
           'features.21',
           'features.28'
            ]
 
output_layer = 'classifier.6'

loaders = [
        'CIFAR100-train',
        'CIFAR100-val',
        'CIFAR100-test',
        'CIFAR100-C-val-c0',
        'CIFAR100-C-test-c0',
        'CIFAR100-C-val-c1',
        'CIFAR100-C-test-c1',
        'CIFAR100-C-val-c2',
        'CIFAR100-C-test-c2',
        'CIFAR100-C-val-c3',
        'CIFAR100-C-test-c3',
        'CIFAR100-C-val-c4',
        'CIFAR100-C-test-c4',
        'SVHN-val',
        'SVHN-test',
        'Places365-val',
        'Places365-test',
        'CW-CIFAR100-val',
        'CW-CIFAR100-test',
        'BIM-CIFAR100-val',
        'BIM-CIFAR100-test',
        'DF-CIFAR100-val',
        'DF-CIFAR100-test',
        'PGD-CIFAR100-val',
        'PGD-CIFAR100-test',
        ]

reduction_fns = {layer: parser_act for layer in target_layers}
