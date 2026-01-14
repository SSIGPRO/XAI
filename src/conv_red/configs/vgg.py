# python stuff
from pathlib import Path as Path

# torch stuff
from torchvision.models import vgg16 as Model

# Our stuff
from peepholelib.datasets.functional.transforms import vgg16_cifar100 as transform

model_dir = '/srv/newpenny/XAI/models'
model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'

ds_path = Path.home()/'repos/XAI/data/vgg/datasets'

output_layer = 'classifier.6'
bs = 2**9 # CW

target_layers = [
        'features.7',
        'features.10',
        'features.12',
        'features.14',
        'features.17',
        'features.19',
        'features.21',
        'features.24',
        'features.26',
        'features.28',
        ]
