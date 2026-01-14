from torchvision.models import vgg16 as Model
from peepholelib.datasets.functional.transforms import vgg16_cifar100 as transform

output_layer = 'classifier.6'
model_dir = '/srv/newpenny/XAI/models'
model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'


