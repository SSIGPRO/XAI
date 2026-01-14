# Python stuff
from pathlib import Path as Path

# Our stuff
from torchvision.models import mobilenet_v2 as Model
from peepholelib.datasets.functional.transforms import mobilenet_v2 as transform

#------------------
# Paths 
#------------------
model_path = '/srv/newpenny/XAI/models'
model_name = 'CN_model=mobilenet_v2_dataset=CIFAR100_optim=Adam_scheduler=RoP_lr=0.001_factor=0.1_patience=5.pth'

#------------------
# Defs 
#------------------

output_layer = 'classifier.1'
bs_model_scale = 1 

target_layers = [f'features.{i}.conv.2' for i in [14, 17]]# range(2,18,3)]
