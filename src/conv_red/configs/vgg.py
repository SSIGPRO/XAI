# Torch stuff
from torchvision.models import vgg16 as Model

# Peepholelib stuff
from peepholelib.datasets.functional.transforms import vgg16_transform as transform

#------------------
# Paths 
#------------------
model_path = '/srv/newpenny/XAI/models'
model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'

#------------------
# Defs 
#------------------

output_layer = 'classifier.6'
bs_model_scale = 2**-2 

target_layers = [f'features.{i}' for i in [7, 14, 21, 28]]
