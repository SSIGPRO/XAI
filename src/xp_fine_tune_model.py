import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# Our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.datasets.transforms import vgg16_cifar10_augumentations as ds_transform 
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.utils.fine_tune import fine_tune 
from peepholelib.utils.samplers import random_subsampling 

# torch stuff
import torch
from torchvision.models import vgg16
from torchvision.models import VGG16_Weights as pre_train_weights
from cuda_selector import auto_cuda

def ds_parser(batch):
    images, labels = zip(*batch)
    images, labels = list(images), list(labels)
    ims = torch.stack(images)
    lbs = torch.tensor(labels)
    return {'image': ims, 'label': lbs}

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:1')#auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    ds_path = '/srv/newpenny/dataset/CIFAR100'

    # model parameters
    dataset = 'CIFAR10' 
    seed = 29
    bs = 512 
    n_threads = 32
    
    tune_dir = Path.cwd()/'../data/vgg16_cifar10'
    tune_name = 'checkpoints'

    verbose = True 
    
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
    
    nn = vgg16(weights=pre_train_weights.DEFAULT)
    n_classes = len(ds.get_classes()) 
    model = ModelWrap(
            model = nn,
            device = device
            )

    model.update_output(
            output_layer = 'classifier.6', 
            to_n_classes = n_classes,
            overwrite = True 
            )
    
    fine_tune(
            path = tune_dir,
            name = tune_name,
            model = model,
            dataset = ds,
            ds_parser = ds_parser, 
            loss_fn = torch.nn.CrossEntropyLoss,
            loss_kwargs = {'reduction': 'mean'},
            optimizer = torch.optim.SGD,
            optim_kwargs = {'momentum': 0.9},
            #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau,
            #sched_kwargs = {'mode': 'min', 'factor': 0.1, 'patience': 5},
            lr = 1e-3,
            iterations = 'full',
            batch_size = 512*3,
            max_epochs = 30000,
            save_every = 500,
            n_threads = 1,
            devices = [i for i in range(1, 4)], 
            verbose = verbose
            )
