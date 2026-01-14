# python stuff
from pathlib import Path as Path

# torch stuff
from cuda_selector import auto_cuda
from torchvision.models import mobilenet_v2 as Model

# Our stuff
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.datasets.functional.transforms import mobilenet_v2 as transform

#------------------
# Paths 
#------------------
model_dir = '/srv/newpenny/XAI/models'
model_name = 'SV_model=vit_b_16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau_withInfo.pth'

ds_path = Path.home()/'repos/XAI/data/mobilenet/datasets'

svds_path = Path.home()/'repos/XAI/data/mobilenet/svds'

cvs_path = Path.home()/'repos/XAI/data/mobilenet/corevectors'

#------------------
# Defs 
#------------------
import torch
use_cuda = torch.cuda.is_available()
device = torch.device('cpu')#auto_cuda('utilization')) if use_cuda else torch.device("cpu")

output_layer = 'heads.head'
n_classes = 100
bs = 2**8 # CW

target_layers = [f'features.{i}.conv.2' for i in range(2,18,3)]

#------------------
# instances
#------------------
# Model 
model = ModelWrap(
        model = Model(pretrained=True),
        target_modules = target_layers,
        device = device
        )
