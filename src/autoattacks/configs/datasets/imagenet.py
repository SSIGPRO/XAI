from peepholelib.datasets.imagenet import ImageNet
from peepholelib.datasets.functional.samplers import random_subsampling 
from functools import partial

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
            ),
        }

loaders = [
        'ImageNet-train',
        'ImageNet-val',
        'ImageNet-test',
        ]

dss_samplers = {
    'ImageNet': partial(
        random_subsampling, 
        perc = [1, 0.04, 0.2] # train, val, test
        ),
    }