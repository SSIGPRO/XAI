# Python stuff
from pathlib import Path as Path

# Torch stuff
import torch
from cuda_selector import auto_cuda

# Robustbench stuff — importing xcit registers the debenedetti2022 models with timm
import robustbench.model_zoo.architectures.xcit
import timm
from robustbench.utils import load_model

# Our stuff
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.datasets.functional.transforms import TransformWrap

from configs.datasets.cifar import *
from configs.models.model_utils import get_linear_conv2d_layers

#------------------
# Device
#------------------

use_cuda = torch.cuda.is_available()
device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
print(f"Using {device} device")

cfg = {}
n_classes = 100
dataset_name = 'CIFAR100'
transform = lambda x: x
model_path = '/srv/newpenny/XAI/models'

model_name = 'xcit_l12_cifar100_standard.pt'

_std_arch = timm.create_model(
    'debenedetti2022light_xcit_l12_cifar100_linf',
    pretrained=False,
    num_classes=n_classes,
)

model = ModelWrap(model=_std_arch, device=device)
target_layers = get_linear_conv2d_layers(model._model)

model.load_checkpoint(
    path=model_path,
    name=model_name,
    verbose=True,
)

cfg['standard'] = {
    'model': model,
    'layers': target_layers,
}

### Robust Model

model_name = 'Debenedetti2022Light_XCiT-L12'

_rob_arch = load_model(
    model_name=model_name,
    threat_model='Linf',
    dataset='cifar100',
    model_dir=model_path,
)

model = ModelWrap(model=_rob_arch, device=device)
target_layers = get_linear_conv2d_layers(model._model)

cfg['robust'] = {
    'model': model,
    'layers': target_layers,
}
