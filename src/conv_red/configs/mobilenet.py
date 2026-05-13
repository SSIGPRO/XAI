# Torch stuff
from torchvision.models import mobilenet_v2 as Model
from torchvision.models import MobileNet_V2_Weights as pre_train_weights

# Peepholelib stuff
from peepholelib.datasets.functional.transforms import mobilenet_v2_transform as transform
from peepholelib.datasets.functional.transforms import mobilenet_v2_cifar10_augmentations as augmentation 

#------------------
# Paths 
#------------------
model_name = 'mobilenet_cifar100.pt'

#------------------
# Defs 
#------------------

output_layer = 'classifier.1'
bs_model_scale = 2**0

target_layers = [f'features.{i}.conv.2' for i in [8, 11, 14, 17]]
