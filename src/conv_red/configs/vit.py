from torchvision.models import vit_b_16 as Model
from peepholelib.datasets.functional.transforms import vit_b_16_cifar100 as transform

output_layer = 'heads.head'
model_dir = '/srv/newpenny/XAI/models'
model_name = 'SV_model=vit_b_16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau_withInfo.pth'
