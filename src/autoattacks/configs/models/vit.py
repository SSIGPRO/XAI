# Python stuff
from pathlib import Path as Path

# Torch stuff
import torch
from cuda_selector import auto_cuda
from collections import defaultdict

# Robustbench stuff
from torchvision.models import vit_b_16 as Model
from robustbench.utils import load_model

# Our stuff
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.datasets.functional.transforms import means, stds
from peepholelib.datasets.functional.transforms import TransformWrap 

from configs.datasets.cifar import *

#------------------
# Device
#------------------

use_cuda = torch.cuda.is_available()
device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
print(f"Using {device} device")

# gpu_id = 5  # physical GPU index
# use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > gpu_id
# device = torch.device(f"cuda:{gpu_id}" if use_cuda else "cpu")
# print(f"Using {device} device")

cfg = {}
n_classes = 100
dataset_name = 'CIFAR100'
transform = lambda x:x
model_path = '/srv/newpenny/XAI/models'

### Standard Model

model_name = 'SV_model=vit_b_16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau_withInfo.pth'

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

cfg['standard'] ={
    'model': model, 
    'layers': [
            f'encoder.layers.encoder_layer_{i}.mlp.{j}' for i in range(12) for j in [0,3]
                ] + ['heads.head'],
    }