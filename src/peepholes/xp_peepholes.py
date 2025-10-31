import sys
from pathlib import Path as Path
from time import time
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# Our stuff
import peepholelib
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.coreVectors.coreVectors import CoreVectors 
from peepholelib.peepholes.peepholes import Peepholes

# Load one configuration file here
if sys.argv[1] == 'vgg_cifar100':
    from config_cifar100_vgg16 import *
elif sys.argv[1] == 'vit_cifar100':
    from config_cifar100_ViT import *
elif sys.argv[1] == 'vgg_imagenet':
    from config_imagenet_vgg16 import *
else:
    raise RuntimeError('Select a configuration by runing \'python xp_get_corevectors.py <vgg|vit>\'')

if __name__ == "__main__":

    dataset = ParsedDataset(
            path = ds_path,
            )

    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )

    with dataset as ds, corevecs as cv: 
        ds.load_only(
                loaders = loaders,
                verbose = verbose
                )

        cv.load_only(
                loaders = loaders,
                verbose = verbose 
                ) 

        # peepholes = Peepholes(
        #         path = phs_path,
        #         name = phs_name,
        #         device = device
        #         )
        for drill_key, driller in drillers.items():
            if (driller._empp_file).exists():
                print(f'Loading Classifier for {drill_key}') 
                driller.load()
            else:
                t0 = time()
                print(f'Fitting classifier for {drill_key}')
                driller.fit(
                        corevectors = cv,
                        loader = 'ImageNet-train',
                        verbose=verbose
                        )
                print(f'Fitting time for {drill_key}  = ', time()-t0)

                driller.compute_empirical_posteriors(
                        datasets = ds,
                        corevectors = cv,
                        loader = 'ImageNet-train',
                        batch_size = bs,
                        verbose=verbose
                        )
        
                # save classifiers
                print(f'Saving classifier for {drill_key}')
                driller.save()

        # with peepholes as ph:
        #     ph.get_peepholes(
        #         datasets = ds,
        #         corevectors = cv,
        #         target_modules = target_layers,
        #         batch_size = bs,
        #         drillers = drillers,
        #         n_threads = 1,
        #         verbose = verbose 
        #         )
            
