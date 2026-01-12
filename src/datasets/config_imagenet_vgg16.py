import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# torch stuff
import torch
from torchvision.models import vgg16 as Model
from torchvision.models import VGG16_Weights 
from cuda_selector import auto_cuda

# our stuff
from peepholelib.datasets.functional.transforms import vgg16_imagenet as transform

use_cuda = torch.cuda.is_available()
device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
print(f"Using {device} device")

#--------------------------------
# Directories definitions
#--------------------------------

# model parameters
seed = 29
bs = 2**10
n_threads = 1

ds_path = '/srv/newpenny/dataset/ImageNet_torchvision'

ds_path = Path.cwd()/'../../data/CLIP/parsed_datasets/ImageNet_VGG16'

name_model = 'vgg16'
dataset = 'imagenet'
weights = VGG16_Weights.DEFAULT
verbose = True 

output_layer = 'classifier.6'

model = Model(weights=weights)
