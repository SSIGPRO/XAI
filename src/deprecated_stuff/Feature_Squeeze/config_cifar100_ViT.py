import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from functools import partial

# Our stuff
from peepholelib.datasets.cifar import Cifar as Dataset
from peepholelib.datasets.transforms import vit_b_16_cifar100 as ds_transform
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.featureSqueezing.FeatureSqueezingDetector import FeatureSqueezingDetector
from peepholelib.featureSqueezing.preprocessing import NLM_filtering_torch, NLM_filtering_cv, bit_depth_torch, MedianPool2d

# torch stuff
import torch
from torchvision.models import vit_b_16 as Model
from cuda_selector import auto_cuda

# use_cuda = torch.cuda.is_available()
# device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
# print(f"Using {device} device")

use_cuda = torch.cuda.is_available()
cuda_index = torch.cuda.device_count() - 4
device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
print(f"Using {device} device")

#--------------------------------
# Directories definitions
#--------------------------------
ds_path = '/srv/newpenny/dataset/CIFAR100'

# model parameters
dataset = 'CIFAR100' 
seed = 29
bs = 2**6 
n_threads = 1 

model_dir = '/srv/newpenny/XAI/models'
model_name = 'SV_model=vit_b_16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau_withInfo.pth'

name_model = 'ViT'
verbose = True 

output_layer = 'heads.head'