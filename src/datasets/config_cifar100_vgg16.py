import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# torch stuff
import torch
from torchvision.models import vgg16 as Model
from cuda_selector import auto_cuda

# our stuff
from peepholelib.datasets.functional.transforms import vgg16_cifar100 as transform

use_cuda = torch.cuda.is_available()
device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
print(f"Using {device} device")

#--------------------------------
# Directories definitions
#--------------------------------

# model parameters
seed = 29
bs = 1024 
n_threads = 1

model_dir = '/srv/newpenny/XAI/models'
model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
ds_path = '/srv/newpenny/XAI/generated_data/TPAMI/old/parsed_datasets/CIFAR100_VGG16'
ds_path = Path.cwd()/'../../data/parsed_datasets/CIFAR100_VGG16'

name_model = 'vgg16'
dataset = 'cifar100'
verbose = True 

output_layer = 'classifier.6'

model = Model()
