# Python stuff
from pathlib import Path as Path

# RobustBench stuff
from robustbench.utils import load_model
from peepholelib.datasets.functional.transforms import wrn_cifar100_transform as transform

#------------------
# Paths 
#------------------
model_path = Path('/srv/newpenny/XAI/conceptograms/LC/WRNRobust28_10') #WRNRobust70_16
model_name = 'Wang2023Better_WRN-28-10' #Wang2023Better_WRN-70-16.pt

n_classes = 100

# Load robsut model
Model = load_model(model_name = model_name, threat_model='Linf', dataset='cifar100', model_dir = model_path)
print(Model)
# ------------------
# Defs
# ------------------
output_layer = 'logits'
bs_model_scale = 1

target_layers = ['init_conv', 'layer.0', 'layer.1', 'layer.2']