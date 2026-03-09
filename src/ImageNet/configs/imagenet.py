from peepholelib.datasets.imagenet import ImageNet
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
            )
        }

loaders = [
        'ImageNet-train',
        ]
