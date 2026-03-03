from peepholelib.datasets.imagenet import ImageNet
from peepholelib.datasets.functional.transforms import vgg16_imagenet as transform

seed = 29

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
            seed = seed
            )
        }

loaders = [
        'ImageNet-train',
        ]
