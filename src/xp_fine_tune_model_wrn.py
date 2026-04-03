import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
from datetime import datetime

# torch stuff
import torch
from torch.utils.data.dataloader import default_collate
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.optim import AdamW
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from robustbench.model_zoo.architectures.wide_resnet import WideResNet
from cuda_selector import auto_cuda

# Our stuff
from peepholelib.datasets.cifar100 import Cifar100
from peepholelib.datasets.functional.transforms import means, stds, wrn_cifar100_transform as transform, wrn_cifar100_augmentations as augmentation
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.training.trainingBase import Trainer

if __name__ == "__main__":

    # use_cuda = torch.cuda.is_available()
    # device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    # print(f"Using {device} device")
    gpu_id = 5
    device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using {device} device")
    torch.cuda.set_device(device)

    #--------------------------------
    # Model Parameters
    #--------------------------------
    name_dataset = 'CIFAR100' 
    name_model = 'WRN70_16'
    seed = 29
    n_threads = 1
    output_layer = 'fc'

    #--------------------------------
    # Directories definitions
    #--------------------------------
    ds_path = '/srv/newpenny/dataset/CIFAR100'

    basic_dir = Path(f'/srv/newpenny/XAI/conceptograms/LC/{name_model}_{name_dataset}/')

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tune_dir = basic_dir / "checkpoints" / run_id
    tune_name = 'model'

    verbose = True 

    #--------------------------------
    # Dataset 
    #--------------------------------

    dataset = Cifar100(
                    path = ds_path,
                    std_transform = transform,
                    aug_transform = augmentation
                    )

    dataset.__load_data__()

    #--------------------------------
    # Model 
    #--------------------------------
    n_classes = len(Cifar100.get_classes(meta_path = Path(ds_path)/'cifar-100-python/meta'))
    
    nn = WideResNet(depth=70, widen_factor=16, num_classes=n_classes, dropRate=0.0,) #28 - 10
    print(nn)

    model = ModelWrap(
            model = nn,
            device = device
            )

    # model.update_output(output_layer=output_layer, to_n_classes=n_classes)

    model.normalize_model(mean=means[name_dataset], std= stds[name_dataset])

    #-----------------------------------------------
    # Phase 3: Finetune the whole model
    #-----------------------------------------------
    dl_kwargs = dict(
        collate_fn=default_collate,
        num_workers=n_threads,
        pin_memory=device.type == "cuda",
        persistent_workers=n_threads > 0,
    )
    if n_threads > 0:
        dl_kwargs["prefetch_factor"] = 2

    bs = 2**7
    num_epochs = 200
    # after model is created / normalized
    
    trainable_params = model.get_trainable_parameters(
                                                    layers_to_train=None,
                                                    verbose=verbose
                                            )

    
    optimizer = SGD(
                    trainable_params,
                    lr=0.1,              
                    momentum=0.9,
                    weight_decay=5e-4, #for 28-10 was 2e-4
                    nesterov=True,
                )

    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2,) # for 28-10 used 60, 120, 160 and gamma 0.2
    ## Trainer

    finetuner = Trainer(
                    model = model,
                    path = tune_dir,
                    name = tune_name,
                    dataset = dataset,
                    train_key = f'{name_dataset}-train',
                    val_key = f'{name_dataset}-val',
                    test_key = f'{name_dataset}-test',
                    batch_size = bs,
                    dataloader_kwargs = dl_kwargs,
                    max_epochs = num_epochs,
                    iterations = 'full',
                    optimizer = optimizer,
                    scheduler = scheduler,
                    early_stopping_patience = 200,
                    save_every = 1
                )
    
    finetuner.fit()
    finetuner.test()