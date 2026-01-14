# python stuff
from pathlib import Path as Path

# torch stuff
from torchvision.models import vit_b_16 as Model

# Our stuff
from peepholelib.datasets.functional.transforms import vit_b_16_cifar100 as transform

model_dir = '/srv/newpenny/XAI/models'
model_name = 'SV_model=vit_b_16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau_withInfo.pth'

ds_path = Path.home()/'repos/XAI/data/vit/datasets'

output_layer = 'heads.head'
bs = 2**7 # CW


target_layers = [
        f'encoder.layers.encoder_layer_{i}.mlp.{j}' for i in range(12) for j in [0,3]
        ]
