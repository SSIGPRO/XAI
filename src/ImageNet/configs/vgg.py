# Python stuff
from pathlib import Path as Path

# Torch stuff
import torch
from torchvision.models import vgg16 as Model
from torchvision.models import VGG16_Weights 
from cuda_selector import auto_cuda

# Our stuff
from peepholelib.models.model_wrap import ModelWrap

verbose = True
output_layer = 'classifier.6'

#------------------
# Imagenet
#------------------
name_dataset = 'ImageNet'
n_classes = 1000

weights = VGG16_Weights.DEFAULT
