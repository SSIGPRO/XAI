# Python stuff
from pathlib import Path as Path

# Torch stuff
import torch
from cuda_selector import auto_cuda

# Robustbench stuff
from robustbench.model_zoo.architectures.wide_resnet import WideResNet
from robustbench.utils import load_model

# Our stuff
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.datasets.functional.transforms import means, stds
from peepholelib.datasets.functional.transforms import TransformWrap 

from configs.datasets.cifar import *

#------------------
# Device
#------------------

# use_cuda = torch.cuda.is_available()
# device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
# print(f"Using {device} device")

gpu_id = 5  # physical GPU index
use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > gpu_id
device = torch.device(f"cuda:{gpu_id}" if use_cuda else "cpu")
print(f"Using {device} device")

cfg = {}
n_classes = 100
transform = lambda x:x
model_path = '/srv/newpenny/XAI/models'

### Standard Model

model_name = 'wrn70_16_cifar100.pt'

Model = WideResNet(depth=70, widen_factor=16, num_classes=n_classes, dropRate=0.0,)

model = ModelWrap(
            model = Model(),
            device = device
            )

model.prepend_normalizer(mean=means['CIFAR100'], std= stds['CIFAR100'])

model.load_checkpoint(
        path = model_path,
        name = model_name,
        verbose = True 
        )

target_layers = ['conv1', 'block1', 'block2', 'block3']

cfg['standard'] ={
    'model': model, 
    'target_layers': target_layers
    }

### Robust Model

model_name = 'Wang2023Better_WRN-70-16.pt' 

Model = load_model(model_name = model_name, threat_model='Linf', dataset='cifar100', model_dir = model_path)

model = ModelWrap(
            model = Model(),
            device = device
            )

target_layers = ['init_conv', 'layer.0', 'layer.1', 'layer.2']

cfg['robust'] ={
    'model': model, 
    'target_layers': target_layers
    }