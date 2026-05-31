from peepholelib.datasets.imagenet import ImageNet
from peepholelib.datasets.functional.samplers import random_subsampling 

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

def _imagenet_sampler(ds):
    ds.__dataset__.pop('ImageNet-train', None)
    random_subsampling(ds, perc={'ImageNet-val': 0.04, 'ImageNet-test': 0.2})

dss_samplers = {
    'ImageNet': _imagenet_sampler,
    }