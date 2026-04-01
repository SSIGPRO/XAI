# python stuff
import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/conv_red').as_posix())

# Our stuff
from peepholelib.datasets.parsedDataset import ParsedDataset 

from configs.common import *

if __name__ == "__main__":

    datasets = ParsedDataset(
            path = ds_path,
            )

    with datasets as ds:
        ds.load_only(
                loaders = ['CIFAR100-test'],
                transforms = transforms,
                inference_names = inference_names,
                verbose = verbose
                )

        for ds_key, ds in ds._dss.items():
            r = ds['result']
            acc = r.sum()/len(r)
            print(f'{ds_key}: {acc}')
