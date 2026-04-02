from peepholelib.datasets.cifar100 import Cifar100

seed = 29

#--------------------------------
# Paths and Definitions 
#--------------------------------
cifar_path = '/srv/newpenny/dataset/CIFAR100'

#--------------------------------
# Datasets 
#--------------------------------
# original datasets
dss = {
        'CIFAR100': Cifar100(
            path = cifar_path,
            seed = seed
            ),
        }

loaders = [
        'CIFAR100-train',
        'CIFAR100-val',
        'CIFAR100-test',
        ]