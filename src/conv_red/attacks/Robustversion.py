from robustbench.data import load_cifar10
x_test, y_test = load_cifar10(n_examples=50)

from robustbench.utils import load_model
model = load_model(model_name='Wang2023Better_WRN-70-16', threat_model='Linf', dataset='cifar10') # Wang2023Better_WRN-28-10