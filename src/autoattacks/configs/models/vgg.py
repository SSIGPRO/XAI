# Python stuff
from pathlib import Path as Path

# Torch stuff
import torch
from torchvision.models import vgg16 as Model
from torchvision.models import VGG16_Weights 
from cuda_selector import auto_cuda

# Our stuff
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.datasets.functional.transforms import TransformWrap 
from peepholelib.datasets.functional.transforms import means, stds, vgg16_transform as transform

cfg = {}

verbose = True
output_layer = 'classifier.6'

#------------------
# Device
#------------------

device = torch.device("cpu")

# use_cuda = torch.cuda.is_available()
# device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
# print(f"Using {device} device")

gpu_id = 5  # physical GPU index
use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > gpu_id
device = torch.device(f"cuda:{gpu_id}" if use_cuda else "cpu")
print(f"Using {device} device")
#------------------
# CIFAR 100
#------------------
name_dataset = 'CIFAR100'
model_path = '/srv/newpenny/XAI/models'
model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
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

model.load_checkpoint(
        name = model_name,
        path = model_path,
        verbose = verbose
        )

model.prepend_normalizer(mean=means['CIFAR100'], std= stds['CIFAR100'])

cfg[name_dataset] = {
                'model': model,
                'n_classes': n_classes
                }

#------------------
# Imagenet
#------------------
name_dataset = 'ImageNet'
n_classes = 1000

weights = VGG16_Weights.DEFAULT

model = ModelWrap(
                model = Model(weights=weights),
                device = device
                )

model.prepend_normalizer(mean=means['ImageNet'], std= stds['ImageNet'])

cfg[name_dataset] = {
                'model': model,
                'n_classes': n_classes
                }
