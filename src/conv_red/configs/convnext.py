# Torch stuff
from torchvision.models import convnext_small as Model
from torchvision.models import ConvNeXt_Small_Weights as pre_train_weights

# Peepholelib stuff
from peepholelib.datasets.functional.transforms import convnext_small_transform as transform
from peepholelib.datasets.functional.transforms import convnext_small_augmentations as augmentation 

#------------------
# Paths 
#------------------
model_name = 'convnext_cifar100.pt'

#------------------
# Defs 
#------------------

output_layer = 'classifier.2'
bs_model_scale = 2**-2 

target_layers = [f'features.{i}.{j}' for (i, j) in zip([2, 4, 6], [1, 1, 1])]
