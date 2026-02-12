from pathlib import Path

# Our stuff
from torchvision.models import wide_resnet50_2 as Model
from peepholelib.datasets.functional.transforms import wide_resnet50_2_transform as transform

#------------------
# Paths
#------------------
model_path = Path(__file__).resolve().parent
model_name = 'checkpoints.29.pt'

#------------------
# Defs
#------------------
output_layer = 'fc'
bs_model_scale = 1
target_layers = ['layer3', 'layer4']