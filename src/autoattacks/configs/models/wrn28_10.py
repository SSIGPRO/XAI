# Python stuff
from pathlib import Path as Path

# Torch stuff
import torch
from cuda_selector import auto_cuda
from collections import defaultdict

# Robustbench stuff
from robustbench.model_zoo.architectures.wide_resnet import WideResNet
from robustbench.utils import load_model

# Our stuff
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.datasets.functional.transforms import means, stds
from peepholelib.datasets.functional.transforms import TransformWrap 

from configs.datasets.cifar import *

def _normalize_layers(m):
    return {
        'linear':    m.get('Conv2d', []) + m.get('Linear', []),
        'nonlinear': m.get('ReLU', []),
        'batchnorm': m.get('BatchNorm2d', []),
    }

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
dataset_name = 'CIFAR100'
transform = lambda x:x
model_path = '/srv/newpenny/XAI/models'

### Standard Model

model_name = 'wrn28_10_cifar100.pt'

Model = WideResNet(depth=28, widen_factor=10, num_classes=n_classes, dropRate=0.0,)

model = ModelWrap(
            model = Model,
            device = device
            )

model.prepend_normalizer(mean=means['CIFAR100'], std= stds['CIFAR100'])

model.load_checkpoint(
        path = model_path,
        name = model_name,
        verbose = True 
        )

modules_by_type = defaultdict(list)

for name, module in model._model.named_modules():
    if name == "":
        continue
    modules_by_type[type(module).__name__].append(name)

modules_by_type = dict(modules_by_type)

modules_by_type.pop('Sequential'); modules_by_type.pop('NetworkBlock'); modules_by_type.pop('BasicBlock')
modules_by_type = _normalize_layers(modules_by_type)

cfg['standard'] ={
    'model': model, 
    'layers': modules_by_type
    }

### Robust Model
model_name = 'Wang2023Better_WRN-28-10'

Model = load_model(model_name = model_name, threat_model='Linf', dataset='cifar100', model_dir = model_path)

model = ModelWrap(
            model = Model,
            device = device
            )

modules_by_type = defaultdict(list)

for name, module in model._model.named_modules():
    if name == "":
        continue
    modules_by_type[type(module).__name__].append(name)

modules_by_type = dict(modules_by_type)

modules_by_type.pop('Sequential'); modules_by_type.pop('_BlockGroup'); modules_by_type.pop('_Block')
modules_by_type = _normalize_layers(modules_by_type)

cfg['robust'] ={
    'model': model, 
    'layers': modules_by_type
    }