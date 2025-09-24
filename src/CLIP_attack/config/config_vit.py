import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

from peepholelib.datasets.imagenet import ImageNet
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.datasets.transforms import vit_b_16_imagenet as ds_transform
from peepholelib.models.svd_fns import linear_svd

from peepholelib.coreVectors.dimReduction.svds import linear_svd_projection_ViT, linear_svd_projection

from peepholelib.peepholes.parsers import trim_corevectors

# torch stuff
import torch
from torchvision.models import vit_b_16
from cuda_selector import auto_cuda

from functools import partial

# robustbench
from robustbench.model_zoo.architectures.utils_architectures import normalize_model
from autoattack import AutoAttack
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from robustbench.utils import clean_accuracy, load_model, parse_args, update_json
from robustbench.data import get_preprocessing, load_clean_dataset

use_cuda = torch.cuda.is_available()
device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
print(f"Using {device} device")

# use_cuda = torch.cuda.is_available()
# cuda_index = torch.cuda.device_count() - 4
# device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
# print(f"Using {device} device")

model_name = 'vit'
dataset = 'imagenet'
dataset_ = 'ImageNet'
seed = 29

ds_path = '/srv/newpenny/dataset/ImageNet_torchvision'

svds_path = Path.cwd()/f'../../data/{model_name}'
svds_name = 'svds' 

svd_rank = 300
verbose = True

#--------------------------------
# Dataset 
#--------------------------------

ds = ImageNet(
        data_path = ds_path,
        dataset = dataset_
        )

ds.load_data(
        transform = ds_transform,
        seed = seed,
        )

n_classes = len(ds.get_classes())

nn = vit_b_16(weights='IMAGENET1K_V1')
model = ModelWrap(
        model = nn,
        device = device
        )

# Peepholelib
target_layers = [
        f'encoder.layers.encoder_layer_{i}.mlp.{j}' for i in range(11,12) for j in [0,3]
        ]
target_layers.append('heads.head')

#--------------------------------
# SVDs 
#--------------------------------
svd_fns = {}
for _layer in target_layers:
    svd_fns[_layer] = partial(
            linear_svd,
            rank = svd_rank,
            device = device,
            )
    
#--------------------------------
# Model 
#--------------------------------    

model.set_target_modules(
        target_modules = target_layers,
        verbose = verbose
        )

model.get_svds(
        path = svds_path,
        name = svds_name,
        target_modules = target_layers,
        sample_in = ds._dss['train'][0][0],
        svd_fns = svd_fns,
        verbose = verbose
        )

# define a dimensionality reduction function for each layer
reduction_fns = {}
for _layer in target_layers:
    if 'encoder.layers.encoder_layer_' in _layer:
        reduction_fns[_layer] = partial(
                linear_svd_projection_ViT,
                svd = model._svds[_layer],
                use_s = True,
                device=device
                )
    elif 'heads.head' in _layer:
        reduction_fns[_layer] = partial(
                linear_svd_projection,
                svd = model._svds[_layer],
                use_s = True,
                device=device
                )
#--------------------------------
# Reduction functions 
#--------------------------------

feature_sizes = {}
cv_parsers = {}

cv_dim = 100

for _layer in target_layers:
    cv_parsers[_layer] = partial(
            trim_corevectors,
            module = _layer,
            cv_dim = cv_dim,
            )
    feature_sizes[_layer] = cv_dim