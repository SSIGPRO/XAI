import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from functools import partial

# Our stuff
from peepholelib.models.svd_fns import linear_svd
from peepholelib.coreVectors.dimReduction.svds import linear_svd_projection_ViT, linear_svd_projection

# torch stuff
import torch
from torchvision.models import vit_b_16 as Model
from cuda_selector import auto_cuda

use_cuda = torch.cuda.is_available()
device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
print(f"Using {device} device")

#--------------------------------
# Directories definitions
#--------------------------------
ds_path = '/srv/newpenny/XAI/generated_data/TPAMI/parsed_datasets/CIFAR100_ViT'
#ds_path = Path.cwd()/'../../data/parsed_datasets/CIFAR100_ViT'
ds_kwargs = {}

# model parameters
dataset = 'CIFAR100' 
bs = 2**10
n_threads = 1 
n_classes = 100

model_dir = '/srv/newpenny/XAI/models'
model_name = 'SV_model=vit_b_16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau_withInfo.pth'

# Running on Agath local. transfer data later
# svds_path = Path('/srv/newpenny/XAI/generated_data/TPAMI/svds/CIFAR100_ViT')
svds_path = Path.cwd()/'../../data/svds/CIFAR100_ViT'
svds_name = f'svds' 

cvs_path = Path('/srv/newpenny/XAI/generated_data/TPAMI/corevectors/CIFAR100_ViT')
#cvs_path = Path.cwd()/'../../data/corevectors/CIFAR100_ViT'
cvs_name = 'corevectors'

verbose = True 

save_input = True
save_output = False

loaders = [
        'CIFAR100-train',
        'CIFAR100-val',
        'CIFAR100-test',
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
target_layers = [
        f'encoder.layers.encoder_layer_{i}.mlp.{j}' for i in range(11,12) for j in [0] #[0,3]
        ]
#target_layers.append('heads.head')

svd_rank = 500 
output_layer = 'heads.head'

svd_fns = {}
for _layer in target_layers:
    svd_fns[_layer] = partial(
            linear_svd,
            rank = svd_rank,
            device = device,
            )

# define a dimensionality reduction function for each layer
reduction_fns = {}
for _layer in target_layers:
    if 'encoder.layers.encoder_layer_' in _layer:
        reduction_fns[_layer] = partial(
                linear_svd_projection_ViT,
                use_s = True,
                device=device
                )
    elif 'heads.head' in _layer:
        reduction_fns[_layer] = partial(
                linear_svd_projection,
                use_s = True,
                device=device
                )
