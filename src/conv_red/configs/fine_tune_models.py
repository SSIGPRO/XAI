# torch stuff
from torch.utils.data.dataloader import default_collate
from torch.optim.lr_scheduler import ReduceLROnPlateau as Scheduler 
from torch.optim import AdamW as Optim

#--------------------------------
# Model Parameters
#--------------------------------
seed = 29
n_threads = 1
dataset_name = 'CIFAR100'                                          

#--------------------------------
# Training defs
#--------------------------------
model_fine_tune_bs = 2**8
iterations = 3 
save_every = 2000
max_epochs = 40000 
                                                  
dl_kwargs = dict(
        collate_fn = default_collate,
        num_workers = n_threads,
        persistent_workers=n_threads > 0,
)
                                                  
opt_kwargs = {
        'lr': 1e-3,
        'weight_decay': 0.05,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        }
                                                  
scheduler_kwargs = {
        'patience': 50,
        'min_lr': 1e-3,
        'cooldown': 10,
        }

