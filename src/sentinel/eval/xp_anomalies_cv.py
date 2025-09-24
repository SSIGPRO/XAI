from functools import partial
import sys
from time import time
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import DataLoader


import torch
from cuda_selector import auto_cuda
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/sentinel').as_posix())

# Our stuff
from peepholelib.datasets.sentinel import Sentinel
from peepholelib.models.sentinel.model_cv_sentinel import AE2D_instance
from peepholelib.datasets.sentinel import SentinelWrap
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.datasets.cv_dataset import ConvDataset



from configs.common import *
from configs.peepholes import *
from configs.anomalies import *


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
    #device = torch.device("cpu")
    print(f"Using {device} device")


    #--------------------------------
    # Model
    #--------------------------------
    
    cv_model = AE2D_instance(
        num_sensors = num_sensors,
        seq_len = seq_len,
        kernel_size = kernel,
        embedding_size = emb_size,
        lay3 = lay3
    )
    #print(cv_model)
    #quit()
    
    model = ModelWrap(
            model = cv_model,
            device = device
            )

    
    sentinel_ds = Sentinel(
        path = parsed_path
    )

    corevectors = CoreVectors(
        path = cvs_path,
        name = cvs_name,
        model = model
    )

    
    with sentinel_ds as s, corevectors as cv:
        s.load_only(
            loaders = loaders,
            verbose = verbose
        )

        cv.load_only(
            loaders = loaders,
            verbose = verbose
        )
        #-------------------------Create DS with CVs--------------------------#
        dataset_train = torch.cat([v for v in cv._corevds['train'].values()], dim=1)
        dataset_test = torch.cat([v for v in cv._corevds['test'].values()], dim=1)  
        
        #print(f'dataset_train_.shape{dataset_train.shape}')

        train_ds = ConvDataset(dataset_train)
        test_ds  = ConvDataset(dataset_test)

        train_iter = DataLoader(train_ds, batch_size=bs, shuffle=True)
        test_iter  = DataLoader(test_ds, batch_size=bs, shuffle=False)
        #-----------------------------      -------------------------------------#

        cv.get_corruptions_all(loaders = ['val'],
                          model = model,
                          corruptions = corruptions['high'],
                          n_samples = 1000,
                          thr = 0.002,
                          bs = 2**11)

        