import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# Our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.datasets.transforms import vit_b_16_cifar100_augumentations as ds_transform 
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.utils.fine_tune import fine_tune 
from peepholelib.utils.samplers import random_subsampling 

# torch stuff
import torch
from torchvision.models import vit_b_16 
from torchvision.models import ViT_B_16_Weights as pre_train_weights
from cuda_selector import auto_cuda

def ds_parser(batch):
    images, labels = zip(*batch)
    images, labels = list(images), list(labels)
    ims = torch.stack(images)
    lbs = torch.tensor(labels)
    return {'image': ims, 'label': lbs}

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    ds_path = '/srv/newpenny/dataset/CIFAR100'

    # model parameters
    dataset = 'CIFAR100' 
    seed = 29
    bs = 2**9 
    n_threads = 32
    
    tune_dir = Path.cwd()/'../../data/vit_cifar100_AdamW'
    tune_name = 'checkpoints'

    verbose = True 
    
    #model_dir = '/srv/newpenny/XAI/models/'
    #model_name = 'SV_model=vit_b_16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau_withInfo.pth'
    # 'epoch': 7, 'initial_lr': 0.001, 'final_lr': 0.001, 'train_accuracy': 88.13499999999999, 'val_accuracy': 86.24000000000001
    
    #--------------------------------
    # Dataset 
    #--------------------------------
    
    ds = Cifar(
            data_path = ds_path,
            dataset = dataset
            )

    ds.load_data(
            transform = ds_transform,
            seed = seed,
            )

    #random_subsampling(ds, 0.025)
    
    #--------------------------------
    # Model 
    #--------------------------------
    
    nn = vit_b_16(weights=pre_train_weights.DEFAULT)
    n_classes = len(ds.get_classes()) 
    model = ModelWrap(
            model = nn,
            device = device
            )

    model.update_output(
            output_layer = 'heads.head', 
            to_n_classes = n_classes,
            overwrite = True 
            )
    
    # use AdamW for training. See https://arxiv.org/pdf/2211.09359
    # slow and not very good
    fine_tune(
            path = tune_dir,
            name = tune_name,
            model = model,
            dataset = ds,
            ds_parser = ds_parser, 
            loss_fn = torch.nn.CrossEntropyLoss,
            loss_kwargs = {'reduction': 'mean'},
            optimizer = torch.optim.AdamW,#SGD,
            optim_kwargs = {},#{'momentum': 0.9},
            #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau,
            #sched_kwargs = {'mode': 'min', 'factor': 0.1, 'patience': 5},
            lr = 5e-5,
            iterations = 'full',
            batch_size = bs,
            max_epochs = 400,
            save_every = 40,
            n_threads = 1,
            devices = [1, 4, 5], 
            verbose = verbose
            )
