# Torch stuff
from torchvision.models import resnet50 as Model
from torchvision.models import ResNet50_Weights as pre_train_weights

# Peepholelib stuff
from peepholelib.datasets.functional.transforms import resnet50_transform as transform
from peepholelib.datasets.functional.transforms import resnet50_cifar100_augmentations as augmentation 

#------------------
# Paths 
#------------------
model_name = 'resnet_cifar100.pt'

#------------------
# Defs 
#------------------

output_layer = 'fc'
bs_model_scale = 2**-2 

target_layers = [f'layer{i}.{j}.conv3' for (i, j) in zip([1, 2, 3, 4], [2, 3, 5, 2])]
