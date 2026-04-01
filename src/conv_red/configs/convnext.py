# Torch stuff
from torchvision.models import convnext_small as Model

# Peepholelib stuff
from peepholelib.datasets.functional.transforms import convnext_small_transform as transform

#------------------
# Paths 
#------------------
model_path = '/srv/newpenny/XAI/models'
model_name = 'convnext_cifar100_clean_sd.pt'

#------------------
# Defs 
#------------------

output_layer = 'classifier.2'
bs_model_scale = 2**-2 

target_layers = [f'features.{i}.{j}' for (i, j) in zip([2, 4, 6], [1, 1, 1])]

