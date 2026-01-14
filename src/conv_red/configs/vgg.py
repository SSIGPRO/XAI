# python stuff
from pathlib import Path as Path

# torch stuff
import torch
from cuda_selector import auto_cuda
from torchvision.models import vgg16 as Model

# Our stuff
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.datasets.functional.transforms import vgg16_cifar100 as transform


#------------------
# Paths 
#------------------
model_dir = '/srv/newpenny/XAI/models'
model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'

ds_path = Path.home()/'repos/XAI/data/vgg/datasets'

svds_path = Path.home()/'repos/XAI/data/vgg/svds'

cvs_path = Path.home()/'repos/XAI/data/vgg/corevectors'

#------------------
# Defs 
#------------------
use_cuda = torch.cuda.is_available()
device = torch.device('cpu')#auto_cuda('utilization')) if use_cuda else torch.device("cpu")

output_layer = 'classifier.6'
n_classes = 100
bs = 2**9 # CW

target_layers = [f'features.{i}' for i in [2, 7, 14, 21, 28]]

#------------------
# instances
#------------------

# Model 
model = ModelWrap(
        model = Model(),
        target_modules = target_layers,
        device = device
        )
                                        
model.update_output(
        output_layer = output_layer, 
        to_n_classes = n_classes,
        overwrite = True 
        )
                                        
model.load_checkpoint(
        name = model_name,
        path = model_dir,
        verbose = True 
        )
