import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# Our stuff
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.coreVectors.coreVectors import CoreVectors

import matplotlib.pyplot as plt

# Load one configuration file here
if sys.argv[1] == 'vgg_cifar100':
    from config_cifar100_vgg16 import *
elif sys.argv[1] == 'vit_cifar100':
    from config_cifar100_ViT import *
elif sys.argv[1] == 'vgg_imagenet':
    from config_imagenet_vgg16 import *
elif sys.argv[1] == 'vit_imagenet':
    from config_imagenet_ViT import *
else:
    raise RuntimeError('Select a configuration by runing \'python xp_get_corevectors.py <vgg|vit>\'')
    
if __name__ == "__main__":
    #--------------------------------
    # Dataset 
    #--------------------------------

    dataset = ParsedDataset(
            path = ds_path,
            )
    
    loaders = [
                f'{name_dataset}-train', 
                f'{name_dataset}-test', 
                f'{name_dataset}-val', 
                f'{name_dataset}-C-test-c0', 
                f'{name_dataset}-C-val-c0', 
                f'{name_dataset}-C-test-c1', 
                f'{name_dataset}-C-val-c1', 
                f'{name_dataset}-C-test-c2', 
                f'{name_dataset}-C-val-c2', 
                f'{name_dataset}-C-test-c3', 
                f'{name_dataset}-C-val-c3', 
                f'{name_dataset}-C-test-c4', 
                f'{name_dataset}-C-val-c4', 
                f'BIM-{name_dataset}-test', 
                f'CW-{name_dataset}-test', 
                f'DF-{name_dataset}-test',
                f'PGD-{name_dataset}-test'
                ]

    with dataset as ds: 
        ds.load_only(
                loaders = loaders,
                verbose = verbose
                )
        
        for loader in loaders:

            plt.imshow(ds._dss[loader]['image'][0].permute(1,2,0))
            plt.title(f'pred: {ds._dss[loader]["pred"][0]}')
            plt.savefig(f'{loader}_example.png')