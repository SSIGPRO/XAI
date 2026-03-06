from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.optim import SGD
import torch
import inspect

schedulers = [CosineAnnealingLR, LinearLR, ReduceLROnPlateau]

model_param = [torch.nn.Parameter(torch.randn(1))]
optimizer = SGD(model_param, lr=0.1)
scheduler = ReduceLROnPlateau(optimizer) 

for scheduler in schedulers:
    print('-------------')
    if issubclass(scheduler, CosineAnnealingLR):
        s = scheduler(optimizer, T_max = 1)  
    else:
        s = scheduler(optimizer)
    print(s.__dict__)         # shows num_bad_epochs, patience, etc.
    # print(scheduler.num_bad_epochs)   # direct access
    # print(scheduler.patience)