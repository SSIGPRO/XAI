import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/ImageNet').as_posix())

# python stuff
from functools import partial
from time import time

# torch stuff
import torch
from cuda_selector import auto_cuda

###### Our stuff
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.datasets.functional.samplers import random_subsampling 
from peepholelib.datasets.functional.transforms import vgg16_transform as transform
from peepholelib.datasets.functional.transforms import TransformWrap 
from peepholelib.datasets.functional.inference_fns import img_classification_full as inference_fn 

from configs.common import *

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    '''
    just for testing
    '''
    dss_samplers = {
            k: partial(
                random_subsampling, 
                perc = 0.01
                ) for k in dss.keys()
            }

    transforms = {
            'ImageNet-train': TransformWrap(transform=transform, input_key='image'),
            'ImageNet-val': TransformWrap(transform=transform, input_key='image'),
            'ImageNet-test': TransformWrap(transform=transform, input_key='image'),
            }

    model = ModelWrap(
            model = Model(weights=weights),
            device = device
            )

    dataset = ParsedDataset(
        path = ds_path
        )

    t0 = time() 
    with dataset as ds:
        ds.parse_dataset(
                dataset_wraps = dss,
                ds_samplers = dss_samplers, 
                keys_to_copy = ['image', 'label'],
                batch_size = bs_base,
                n_threads = 1,
                verbose = verbose,
                )

        ds.parse_inference(
                name = 'ImgNet',
                inference_fn = partial(inference_fn, model=model),
                transforms = transforms,
                batch_size = bs_base,
                n_threads = 1,
                verbose = verbose
                )
    print('time: ', time()-t0)
