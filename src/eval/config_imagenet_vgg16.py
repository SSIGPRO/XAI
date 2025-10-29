# python stuff
from pathlib import Path as Path

# torch stuff
import torch
from torchvision.models import vgg16 as Model
from cuda_selector import auto_cuda

#--------------------------------
# Directories definitions
#--------------------------------

use_cuda = torch.cuda.is_available()
device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
print(f"Using {device} device")

ds_path = Path.cwd()/'../../data/parsed_datasets/ImageNet_VGG16'

cvs_path = Path.cwd()/'../../data/corevectors/ImageNet_VGG16'
cvs_name = 'corevectors'

#phs_path = Path('/srv/newpenny/XAI/generated_data/peepholes_post_tune/CIFAR100_vgg16')
phs_path = Path.cwd()/'../../data/peepholes/ImageNet_VGG16'
phs_name = 'peepholes'
dmd_phs_name = 'peepavg'

verbose = True

#plots_path = Path.cwd()/'../../Papers/TPAMI-2025/figures/CIFAR100_vgg16/' 
plots_path = Path.cwd()/'../temp_plots/ImageNet_VGG16/'

# for xp_conceptrograms
#from peepholelib.datasets.transforms import vgg16_cifar100 as ds_transform 

#ds_path = '/srv/newpenny/dataset/CIFAR100'
ds_path = Path.cwd()/'../../data/parsed_datasets/ImageNet_VGG16'

bs = 2**14 
n_threads = 1
output_layer = 'classifier.6'

loaders = ['ImageNet-val'] #'ImageNet-train',

target_layers = [
        # 'features.7',
        # 'features.10',
        # 'features.12',
        # 'features.14',
        # 'features.17',
        # 'features.19',
        # 'features.21',
        'model.features.24',
        'model.features.26',
        'model.features.28',
        'model.classifier.0',
        'model.classifier.3',
        'model.classifier.6',
        ]

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
