from robustbench.model_zoo import model_dicts
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from robustbench.utils import load_model
import torch

print(model_dicts[BenchmarkDataset.cifar_10].keys(), model_dicts[BenchmarkDataset.cifar_10][ThreatModel.Linf]['Sehwag2020Hydra'])  
quit()
# Example: load a ResNet-50 model robust on CIFAR-10 against l∞ attacks (ε = 8/255)
model = load_model(model_name='Andriushchenko2020Understanding', dataset='cifar10', threat_model='Linf')
print(model)