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

model_dir = '/srv/newpenny/XAI/models'
model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'

phs_path = Path('/srv/newpenny/XAI/generated_data/TPAMI/peepholes_post_tune/CIFAR100_VGG16')
# phs_path = Path.cwd()/'../../data/peepholes_post_tune/CIFAR100_vgg16'
phs_name = 'peepholes'
dmd_phs_name = 'peepavg'

verbose = True

#plots_path = Path.cwd()/'../../Papers/TPAMI-2025/figures/CIFAR100_vgg16/' 
plots_path = Path.cwd()/'../temp_plots/cifar100_vgg16/'

#ds_path = '/srv/newpenny/dataset/CIFAR100'
ds_path = '/srv/newpenny/XAI/generated_data/TPAMI/parsed_datasets/VGG16'
# ds_path = Path.cwd()/'../../data/parsed_datasets/CIFAR100_VGG16'

bs = 2**14 
n_threads = 1
output_layer = 'classifier.6'

target_layers = [
        'features.7',
        'features.10',
        'features.12',
        'features.14',
        'features.17',
        'features.19',
        'features.21',
        'features.24',
        'features.26',
        'features.28',
        'classifier.0',
        'classifier.3',
        'classifier.6',
        ]

cp_ticks = ['f.7', 'f.10', 'f.12', 'f.14', 'f.17', 'f.19', 'f.21', 'f.24', 'f.26', 'f.28', 'c01', 'c.3', 'c.6']

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
