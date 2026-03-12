from peepholelib.datasets.imagenet import ImageNet
from peepholelib.datasets.functional.transforms import means, stds, vgg16_transform as transform

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
            std_transform = transform,
            seed = seed
            )
        }

loaders = [
        'ImageNet-train',
        ]
