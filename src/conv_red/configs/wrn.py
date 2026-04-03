# Python stuff
from pathlib import Path as Path

# Torch stuff
from robustbench.model_zoo.architectures.wide_resnet import WideResNet
from peepholelib.datasets.functional.transforms import wrn_cifar100_transform as transform

#------------------
# Paths 
#------------------

# model_path = Path('/srv/newpenny/XAI/conceptograms/LC/WRN28_10_CIFAR100/checkpoints/20260319_134646/best_model') # this is 28-10
model_path = Path('/srv/newpenny/XAI/conceptograms/LC/WRN70_16_CIFAR100/checkpoints/20260328_215734/best_model') #this is wrn 70-16

model_name = 'best_model_config.pt'

n_classes = 100
Model = WideResNet(depth=70, widen_factor=16, num_classes=n_classes, dropRate=0.0,)
print(Model)
# ------------------
# Defs
# ------------------
output_layer = 'fc'
bs_model_scale = 1

# valid module names in RobustBench WideResNet
target_layers = ['conv1', 'block1', 'block2', 'block3']