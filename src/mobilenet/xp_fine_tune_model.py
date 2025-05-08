import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# Our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.datasets.transforms import mobile_netv2_augumentations as ds_transform 
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.utils.fine_tune import fine_tune 
from peepholelib.utils.samplers import random_subsampling 

# torch stuff
import torch
from torchvision.models import mobilenet_v2 
from torchvision.models import MobileNet_V2_Weights as pre_train_weights
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
    dataset = 'CIFAR10' 
    seed = 29
    bs = 512 
    n_threads = 32
    
    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'CN_model=mobilenet_v2_dataset=CIFAR100_optim=Adam_scheduler=RoP_lr=0.001_factor=0.1_patience=5.pth'
    
    tune_dir = Path.cwd()/'../data/mobile_net_fine_tune_model'
    tune_name = 'vgg16_cifar100'

    verbose = True 
    
    #--------------------------------
    # Dataset 
    #--------------------------------

    ds = Cifar(
            data_path = ds_path,
            dataset=dataset
            )

    ds.load_data(
            transform = ds_transform,
            seed = seed,
            )

    random_subsampling(ds, 0.025)
    
    #--------------------------------
    # Model 
    #--------------------------------
    
    # pretrained weights
    nn = mobilenet_v2(weights=pre_train_weights)
    n_classes = len(ds.get_classes()) 
    model = ModelWrap(
            model = nn,
            device = device
            )

    model.update_output(
            output_layer = 'classifier.1', 
            to_n_classes = n_classes,
            overwrite = True 
            )

    # commenting load_checkpoint() means training model from scratch
    '''
    model.load_checkpoint(
            name = model_name,
            path = model_dir,
            verbose = verbose
            )
    '''

    fine_tune(
            path = tune_dir,
            name = tune_name,
            model = model,
            dataset = ds,
            ds_parser = ds_parser, 
            lr = 1e-3,
            iterations = 16,
            batch_size = 256,
            max_epochs = 10000,
            save_every = 50,
            n_threads = 1,
            verbose = verbose
            )
