import sys
sys.path.insert(0, '/home/saravorabbi/repos/peepholelib')

# torch
import torch
import torchvision
from tensordict import TensorDict, PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT

# python
import h5py
from pathlib import Path as Path
import numpy as np
from matplotlib import pyplot as plt
from contextlib import ExitStack
from itertools import islice
import pickle

# our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.coreVectors.coreVectors import CoreVectors 
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.utils.testing import trim_dataloaders


# check what's inside Livias peephole

if __name__ == "__main__":

    # gpu selection
    use_cuda = torch.cuda.is_available()
    cuda_index = 5  # torch.cuda.device_count() -1
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(device)

    verbose = True


    ph_path = '/srv/newpenny/XAI/generated_data/peepholes/CIFAR100/vgg16/tKMeans'
    ph_name = 'peepholes'#.128.100.test'


    #file_path = '/srv/newpenny/XAI/generated_data/peepholes/CIFAR100/vgg16/tKMeans/peepholes.128.100.test'
    file_path = '/home/saravorabbi/Desktop/new_ph/peepholes.200.50.test'

    # ph = PersistentTensorDict.from_h5(file_path, mode='r')

    # print(ph)


    mapping_dir = '/srv/newpenny/XAI/models/superclass_mapping_CIFAR100.pkl'

    
    with open(Path(mapping_dir), 'rb') as f:
        mapping_dict = pickle.load(f)

    print(mapping_dict.keys())
    print(mapping_dict[0])
    