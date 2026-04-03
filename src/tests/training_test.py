import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
from datetime import datetime

# torch stuff
import torch
from cuda_selector import auto_cuda
from torchvision.models import swin_v2_b
from torchvision.models import Swin_V2_B_Weights as pre_train_weights
from torch.utils.data.dataloader import default_collate
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.optim import AdamW

# Our stuff
from peepholelib.datasets.AwA import AwA
from peepholelib.datasets.functional.transforms import means, stds, swin_b as transform, swin_b_cifar100_augmentations as augmentation 
from peepholelib.models.model_wrap import ModelWrap  
from peepholelib.training.trainingBase import Trainer

if __name__ == "__main__":

    #--------------------------------
    # SETTING DEFINITION
    #--------------------------------

    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Model Parameters
    #--------------------------------
    name_dataset = 'AwA' 
    normalizing_dataset = 'ImageNet'
    name_model = 'Swin'
    seed = 29
    n_threads = 1
    output_layer = 'head'

    #--------------------------------
    # Directories definitions
    #--------------------------------
    ds_path = '/srv/newpenny/dataset/Animals_with_Attributes2'

    basic_dir = Path.cwd()

    parsed_ds_path = basic_dir/'dataset/'

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tune_dir = basic_dir / "checkpoints" / run_id
    tune_name = 'model'

    verbose = True 

    #--------------------------------
    # Dataset 
    #--------------------------------

    dataset = AwA(
            path = ds_path,
            transform = transform,
            augmentation = augmentation
            )

    dataset.__load_data__()

    #--------------------------------
    # Model 
    #--------------------------------

    nn = swin_v2_b(weights=pre_train_weights.IMAGENET1K_V1)

    n_classes = 50

    model = ModelWrap(
            model = nn,
            device = device
            )

    model.update_output(output_layer=output_layer, to_n_classes=n_classes)

    model.normalize_model(mean=means[normalizing_dataset], std= stds[normalizing_dataset])

    #----------------------------
    # TRAINING
    #----------------------------

    #----------------------------
    # Phase 1: Head-only Warm-up
    #----------------------------

    bs = 2**2
    num_epochs = 20

    ## DataLoader

    dl_kwargs = dict(
        collate_fn=default_collate,
        num_workers=n_threads,
        pin_memory=device.type == "cuda",
        persistent_workers=n_threads > 0,
    )
    if n_threads > 0:
        dl_kwargs["prefetch_factor"] = 2

    ## Optimizer & Scheduler
    trainable_params = model.get_trainable_parameters(
                                                    layers_to_train=None,
                                                    verbose=verbose
                                                )

    optimizer = AdamW(
                    trainable_params,
                    lr=1e-3,          
                )

    # scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs-5)
    
    # ## Trainer

    # finetuner = Trainer(
    #                 model = model,
    #                 path = tune_dir,
    #                 name = tune_name,
    #                 dataset = dataset,
    #                 train_key = f'{name_dataset}-train',
    #                 val_key = f'{name_dataset}-val',
    #                 test_key = f'{name_dataset}-test',
    #                 batch_size = bs,
    #                 dataloader_kwargs = dl_kwargs,
    #                 max_epochs = num_epochs,
    #                 iterations = 5,
    #                 optimizer = optimizer,
    #                 scheduler = scheduler,
    #                 save_every = 1,
    #                 early_stopping_patience = 3,
    #             )
    
    # finetuner.fit()
    # finetuner.test()

    scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    
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
                    max_epochs = num_epochs+5,
                    iterations = 5,
                    optimizer = optimizer,
                    scheduler = scheduler,
                    early_stopping_patience = 3,
                    save_every = 1,
                )
    
    finetuner.fit()
    finetuner.test()