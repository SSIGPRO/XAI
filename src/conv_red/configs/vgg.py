# Torch stuff
from torchvision.models import vgg16 as Model
from torchvision.models import VGG16_Weights as pre_train_weights

# Peepholelib stuff
from peepholelib.datasets.functional.transforms import vgg16_transform as transform
from peepholelib.datasets.functional.transforms import vgg16_cifar_augmentations as augmentation 

#------------------
# Paths 
#------------------
model_name = 'vgg16_cifar100.pt'

#------------------
# Defs 
#------------------

output_layer = 'classifier.6'
bs_model_scale = 2**-2 

target_layers = [f'features.{i}' for i in [7, 14, 21, 28]]
