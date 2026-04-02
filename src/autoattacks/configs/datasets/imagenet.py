from peepholelib.datasets.imagenet import ImageNet
from peepholelib.datasets.imagenetC import ImageNetC
from peepholelib.datasets.cifar100 import Cifar100
from peepholelib.datasets.SVHN import SVHN 
from peepholelib.datasets.Places import Places
from peepholelib.datasets.functional.samplers import random_subsampling 
from functools import partial

seed = 29

#--------------------------------
# Paths and Definitions 
#--------------------------------
imagenet_path = '/srv/newpenny/dataset/ImageNet_torchvision'
imagenetc_path = '/srv/newpenny/dataset/Imagenet-C'
svhn_path = '/srv/newpenny/dataset/SVHN' 
places_path = '/srv/newpenny/dataset/Places365'


#--------------------------------
# Datasets 
#--------------------------------
# original datasets
dss = {
        'ImageNet': ImageNet(
            path = imagenet_path,
            seed = seed
            ),
        'ImageNetC': ImageNetC(
            path = imagenetc_path,
            seed = seed
            ),
        'SVHN': SVHN(
            path = svhn_path,
            seed = seed
            ),
        'Places': Places(
            path = places_path,
            seed = seed
            ),
        }

loaders = [
        'ImageNet-train',
        'ImageNet-val',
        'ImageNet-test',
        'ImageNet-C-val-c0',
        'ImageNet-C-test-c0',
        'ImageNet-C-val-c1',
        'ImageNet-C-test-c1',
        'ImageNet-C-val-c2',
        'ImageNet-C-test-c2',
        'ImageNet-C-val-c3',
        'ImageNet-C-test-c3',
        'ImageNet-C-val-c4',
        'ImageNet-C-test-c4',
        'SVHN-val',
        'SVHN-test',
        'Places365-val',
        'Places365-test',
        ]

sampler_1 = partial(random_subsampling, perc=1)
sampler_004 = partial(random_subsampling, perc=0.04)

dss_samplers = {
    'ImageNet': partial(
        random_subsampling, 
        perc = [1, 0.04, 0.2] # train, val, test
        ),
    'ImageNetC': partial(
        random_subsampling, 
        perc = 0.2
        ), 
    **{
        f'{dataset}': partial(
                        random_subsampling, 
                        perc = 0.2
                        )
        for dataset in ('CIFAR100', 'SVHN', 'Places365')
        },
    }