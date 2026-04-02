# Python stuff
from pathlib import Path as Path

# Torch stuff
import torch
from torchvision.models import resnet50 as Model
from torchvision.models import ResNet50_Weights
from cuda_selector import auto_cuda

# Peepholelib stuff
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.datasets.functional.transforms import TransformWrap 
from peepholelib.datasets.functional.transforms import means, stds, resnet50_transform as transform

cfg = {}

verbose = True
output_layer = 'fc'

#------------------
# Device
#------------------

# device = torch.device("cpu")

# use_cuda = torch.cuda.is_available()
# device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
# print(f"Using {device} device")

gpu_id = 2  # physical GPU index
use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > gpu_id
device = torch.device(f"cuda:{gpu_id}" if use_cuda else "cpu")
print(f"Using {device} device")

#------------------
# CIFAR 100
#------------------
model_path = '/srv/newpenny/XAI/models'
n_classes = 100

model = ModelWrap(
        model = Model(),
        device = device
        )

model.update_output(
        output_layer = output_layer, 
        to_n_classes = n_classes,
        overwrite = True 
        )

model.prepend_normalizer(mean=means['CIFAR100'], std= stds['CIFAR100'])

model.load_checkpoint(
        name = f'resnet_cifar100.pt',
        path = model_path,
        verbose = verbose
        )

cfg['CIFAR100'] = {
                'model': model,
                'n_classes': n_classes
                }

#------------------
# Imagenet
#------------------
n_classes = 1000

weights = ResNet50_Weights.DEFAULT

model = ModelWrap(
                model = Model(weights=weights),
                device = device
                )

model.prepend_normalizer(mean=means['ImageNet'], std= stds['ImageNet'])

cfg['ImageNet'] = {
                'model': model,
                'n_classes': n_classes
                }