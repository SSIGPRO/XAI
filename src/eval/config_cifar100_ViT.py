# python stuff
from pathlib import Path as Path

# torch stuff
import torch
from torchvision.models import vit_b_16 as Model
from cuda_selector import auto_cuda

#--------------------------------
# Directories definitions
#--------------------------------

use_cuda = torch.cuda.is_available()
device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
print(f"Using {device} device")

model_dir = '/srv/newpenny/XAI/models'
model_name = 'SV_model=vit_b_16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau_withInfo.pth'

phs_path = Path('/srv/newpenny/XAI/generated_data/TPAMI/peepholes_post_tune/CIFAR100_ViT')
# phs_path = Path.cwd()/'../../data/peepholes_post_tune/CIFAR100_ViT'
phs_name = 'peepholes'
dmd_phs_name = 'peepavg'

verbose = True

#plots_path = Path.cwd()/'../../Papers/TPAMI-2025/figures/CIFAR100_ViT/' 
plots_path = Path.cwd()/'../temp_plots/cifar100_ViT/'

# for xp_conceptrograms
#from peepholelib.datasets.transforms import vgg16_cifar100 as ds_transform 
# ds_path = '/srv/newpenny/dataset/CIFAR100'
# ds_path = Path.cwd()/'../../data/parsed_datasets/CIFAR100_ViT'
ds_path = '/srv/newpenny/XAI/generated_data/TPAMI/parsed_datasets/CIFAR100_ViT'

bs = 2**14 
n_threads = 1
output_layer = 'heads.head'

target_layers = [
        f'encoder.layers.encoder_layer_{i}.mlp.{j}' for i in range(12) for j in [0,3]
        ]
target_layers.append('heads.head')

cp_ticks = [f'mlp.{i}.{j}' for i in range(12) for j in [0,3]] + ['heads.head']

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
