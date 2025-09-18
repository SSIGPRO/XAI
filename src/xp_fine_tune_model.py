import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# Our stuff
from peepholelib.datasets.parsedDataset import ParsedDataset
from peepholelib.datasets.cifar100 import Cifar100
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.utils.fine_tune import fine_tune 

# torch stuff
import torch
from torchvision.models import vgg16
from torchvision.models import VGG16_Weights as pre_train_weights
from cuda_selector import auto_cuda

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    cifar_path = '/srv/newpenny/dataset/CIFAR100'
    ds_path = Path.cwd()/'../data/datasets'

    # model parameters
    bs = 512 
    n_threads =1 
    
    tune_dir = Path.cwd()/'../data/vgg16_cifar100'
    tune_name = 'checkpoints'

    verbose = True 
    
    #--------------------------------
    # Model 
    #--------------------------------

    nn = vgg16(weights=pre_train_weights.DEFAULT)
    n_classes = len(Cifar100.get_classes(meta_path = Path(cifar_path)/'cifar-100-python/meta')) 

    model = ModelWrap(
            model = nn,
            device = device
            )

    model.update_output(
            output_layer = 'classifier.6', 
            to_n_classes = n_classes,
            overwrite = True 
            )

    #--------------------------------
    # Dataset 
    #--------------------------------
    
    # Assuming we have a parsed dataset in ds_path
    datasets = ParsedDataset(
            path = ds_path,
            )

    #--------------------------------
    # Tuning 
    #--------------------------------
    with datasets as ds:    
        ds.load_only(
                loaders = ['CIFAR100-train', 'CIFAR100-val'],
                verbose = verbose
                )

        fine_tune(
                path = tune_dir,
                name = tune_name,
                model = model,
                dataset = ds,
                train_key = 'CIFAR100-train',
                val_key = 'CIFAR100-val',
                loss_fn = torch.nn.CrossEntropyLoss,
                loss_kwargs = {'reduction': 'mean'},
                optimizer = torch.optim.SGD,
                optim_kwargs = {'momentum': 0.9},
                #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau,
                #sched_kwargs = {'mode': 'min', 'factor': 0.1, 'patience': 5},
                lr = 1e-3,
                iterations = 'full',
                batch_size = 256,
                max_epochs = 30,
                save_every = 5,
                n_threads = 1,
                verbose = verbose
                )
