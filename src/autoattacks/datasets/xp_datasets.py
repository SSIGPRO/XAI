import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/Peepholes-Analysis/src').as_posix())

# python stuff
from functools import partial

###### Our stuff
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.datasets.functional.samplers import random_subsampling 
from peepholelib.datasets.functional.inference_fns import img_classification_full as img_cls_inf

from configs.common import *

if __name__ == "__main__":
    print(args)
    
    '''
    just for testing
    
    dss_samplers = {
            k: partial(
                random_subsampling, 
                perc = 0.003
                ) for k in dss.keys()
            }
    '''   
    #######################
    # parsing datasets
    #######################

    dataset = ParsedDataset(
            path = ds_path
            )

    with dataset as ds:

        if args.dataset == 'CIFAR100':
            
                ds.parse_dataset(
                        dataset_wraps = dss,
                        #ds_samplers = dss_samplers, 
                        keys_to_copy = ['image', 'label'],
                        batch_size = bs_base,
                        n_threads = n_threads,
                        verbose = verbose
                        )
                
        elif args.dataset == 'ImageNet':     

                ds.parse_dataset(
                        dataset_wraps = dss,
                        ds_samplers = dss_samplers, 
                        keys_to_copy = ['image', 'label'],
                        batch_size = bs_base,
                        n_threads = n_threads,
                        verbose = verbose
                        )   

        ds.parse_inference(
                inference_fns = {args.model: partial(img_cls_inf, model=model)},
                transforms = transforms,
                batch_size = bs_base,
                n_threads = n_threads,
                verbose = verbose
                )