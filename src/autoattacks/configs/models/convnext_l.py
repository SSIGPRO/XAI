from pathlib import Path
import torch
from cuda_selector import auto_cuda
from torchvision.models import convnext_large, ConvNeXt_Large_Weights
from robustbench.utils import load_model

from peepholelib.models.model_wrap import ModelWrap
from peepholelib.datasets.functional.transforms import TransformWrap
from configs.datasets.imagenet import *  
from configs.models.model_utils import get_linear_conv2d_layers

device = torch.device(auto_cuda('memory')) if torch.cuda.is_available() else torch.device("cpu")

gpu_id = 5  # physical GPU index
use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > gpu_id
device = torch.device(f"cuda:{gpu_id}" if use_cuda else "cpu")
print(f"Using {device} device")

cfg = {}
n_classes   = 1000
dataset_name = 'ImageNet'
transform    = lambda x: x
model_path   = '/srv/newpenny/XAI/models'

def get_linear_conv2d_layers(net):
    return [
        name
        for name, module in net.named_modules()
        if isinstance(module, torch.nn.Linear)
        or isinstance(module, torch.nn.Conv2d)
    ]

# Standard model
_std = convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)
target_layers = get_linear_conv2d_layers(_std)
model = ModelWrap(model=_std, device=device)

cfg['standard'] = {'model': model, 'layers': target_layers}

# Robust model
_rob = load_model('Liu2023Comprehensive_ConvNeXt-L', threat_model='Linf', dataset='imagenet', model_dir=model_path)
target_layers = get_linear_conv2d_layers(_rob)
model = ModelWrap(model=_rob, device=device)

cfg['robust'] = {'model': model, 'layers': target_layers}
