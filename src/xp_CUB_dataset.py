import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from functools import partial
from matplotlib import pyplot as plt
import matplotlib.patches as patches

# torch stuff
import torch
from torchvision.models import vgg16 as Model
from cuda_selector import auto_cuda

# Model
from peepholelib.models.model_wrap import ModelWrap 
from torchvision.models import VGG16_Weights 
from peepholelib.datasets.functional.transforms import means, stds, vgg16_imagenet as transform
from peepholelib.datasets.CUB import CUB, CUBWrap as wrapper

# from peepholelib.datasets.functional.parsers import from_dataset

use_cuda = torch.cuda.is_available()
device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
print(f"Using {device} device")

if __name__ == "__main__":
        #--------------------------------
        # Dataset 
        #--------------------------------
        parts = [
                'back',
                'beak',
                'belly',
                'breast',
                'crown',
                'forehead',
                'left eye',
                'left leg',
                'left wing',
                'nape',
                'right eye',
                'right leg',
                'right wing',
                'tail',
                'throat'
                ]

        name_model = 'VGG16'
        name_dataset = 'CUB'
        normalizing_dataset = 'ImageNet'

        ds = wrapper(
                path='/srv/newpenny/dataset/CUB_200_2011',
                transform=transform
                )

        cub = CUB(path=Path.cwd()/f'../../data/parsed_datasets/{name_dataset}_{name_model}')

        weights = VGG16_Weights.DEFAULT

        model = ModelWrap(
                model = Model(weights=weights),
                device = device
                )

        model.normalize_model(mean=means[normalizing_dataset], std= stds[normalizing_dataset])

        portion = 'train'
        idx = 1000

        with cub as c:
                c.create_ds(path=Path.cwd()/f'../../data/parsed_datasets/cub_vgg', cub_wrap = ds, verbose=True)
                c.parse_ds(model=model, verobse=True)
                

