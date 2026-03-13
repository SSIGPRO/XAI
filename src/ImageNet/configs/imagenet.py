from peepholelib.datasets.imagenet import ImageNet
from peepholelib.datasets.functional.transforms import vgg16_transform as transform

#--------------------------------
# Paths and Definitions 
#--------------------------------
imagenet_path = '/srv/newpenny/dataset/ImageNet_torchvision'

#--------------------------------
# Datasets 
#--------------------------------
# original datasets
dss = {
        'ImageNet': ImageNet(
            path = imagenet_path,
            std_transform = transform
            )
        }

loaders = [
        'ImageNet-train',
        ]
