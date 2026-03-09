# Torch stuff
from torchvision.models import resnet50 as Model

# Peepholelib stuff
from peepholelib.datasets.functional.transforms import resnet50_transform as transform

#------------------
# Paths 
#------------------
model_path = '/srv/newpenny/XAI/models'
model_name = 'resnet_cifar100_clean_sd.pt'

#------------------
# Defs 
#------------------

output_layer = 'fc'
bs_model_scale = 2**-2 

target_layers = [f'layer{i}.{j}.conv3' for (i, j) in zip([1, 2, 3, 4], [2, 3, 5, 2])]

