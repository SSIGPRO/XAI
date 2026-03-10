import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
from datetime import datetime

# torch stuff
import torch
from torch.utils.data.dataloader import default_collate
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.optim import AdamW
from torchvision.models import convnext_base
from torchvision.models import convnext_base_Weights as pre_train_weights
from cuda_selector import auto_cuda

# Our stuff
from peepholelib.datasets.cifar100 import Cifar100
from peepholelib.datasets.functional.transforms import means, stds, convnext_base_cifar100 as transform, convnext_base_cifar100_augmentations as augmentation
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.training.trainingBase import Trainer

if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Model Parameters
    #--------------------------------
    name_dataset = 'CIFAR100' 
    name_model = 'convnext_base'
    seed = 29
    n_threads = 1
    output_layer = 'fc'

    #--------------------------------
    # Directories definitions
    #--------------------------------
    ds_path = '/srv/newpenny/dataset/CIFAR100'

    basic_dir = Path("/home/arshakumari/repos/XAI/src/Conv_NeXt")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tune_dir = basic_dir / "checkpoints" / run_id
    tune_name = 'model'

    verbose = True 

    #--------------------------------
    # Dataset 
    #--------------------------------

    dataset = Cifar100(
                    path = ds_path,
                    transform = transform,
                    augmentation = augmentation
                    )

    dataset.__load_data__()

    #--------------------------------
    # Model 
    #--------------------------------

    nn = convnext_base(weights=pre_train_weights.DEFAULT)
    for name, module in nn.named_modules():
        print(f"{name}: {type(module).__name__}")
    for k in nn.state_dict().keys():
        print(k)
    quit()
 

    n_classes = len(Cifar100.get_classes(meta_path = Path(ds_path)/'cifar-100-python/meta'))

    model = ModelWrap(
            model = nn,
            device = device
            )

    model.update_output(output_layer=output_layer, to_n_classes=n_classes)

    model.normalize_model(mean=means[name_dataset], std= stds[name_dataset])

    layers_to_train = [

            'model.layer3.0.conv1',
            'model.layer3.0.bn1',
            'model.layer3.0.conv2',
            'model.layer3.0.bn2',
            'model.layer3.0.conv3',
            'model.layer3.0.bn3',
            'model.layer3.0.downsample.0',
            'model.layer3.0.downsample.1',
            'model.layer3.0.downsample.1',
            
            'model.layer3.1.conv1',
            'model.layer3.1.bn1',
            'model.layer3.1.conv2',
            'model.layer3.1.bn2',
            'model.layer3.1.conv3',
            'model.layer3.1.bn3',

            'model.layer3.2.conv1',
            'model.layer3.2.bn1',
            'model.layer3.2.conv2',
            'model.layer3.2.bn2',
            'model.layer3.2.conv3',
            'model.layer3.2.bn3',

            'model.layer3.3.conv1',
            'model.layer3.3.bn1',
            'model.layer3.3.conv2',
            'model.layer3.3.bn2',
            'model.layer3.3.conv3',
            'model.layer3.3.bn3',

            'model.layer3.4.conv1',
            'model.layer3.4.bn1',
            'model.layer3.4.conv2',
            'model.layer3.4.bn2',
            'model.layer3.4.conv3',
            'model.layer3.4.bn3',

            'model.layer3.5.conv1',
            'model.layer3.5.bn1',
            'model.layer3.5.conv2',
            'model.layer3.5.bn2',
            'model.layer3.5.conv3',
            'model.layer3.5.bn3',

            'model.layer4.0.conv1.weight',
            'model.layer4.0.bn1',
            'model.layer4.0.conv2.weight',
            'model.layer4.0.bn2',
            'model.layer4.0.conv3.weight',
            'model.layer4.0.bn3',
            'model.layer4.0.downsample.0.weight',
            'model.layer4.0.downsample.1.weight',
            'model.layer4.0.downsample.1.bias',

            'model.layer4.1.conv1.weight',
            'model.layer4.1.bn1',
            'model.layer4.1.conv2.weight',
            'model.layer4.1.bn2',
            'model.layer4.1.conv3.weight',
            'model.layer4.1.bn3',

            'model.layer4.2.conv1.weight',
            'model.layer4.2.bn1',
            'model.layer4.2.conv2.weight',
            'model.layer4.2.bn2',
            'model.layer4.2.conv3.weight',
            'model.layer4.2.bn3',
    ]


    #----------------------------
    # Phase 1: Head-only Warm-up
    #----------------------------

    bs = 2**10
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
                                                    layers_to_train=[output_layer],
                                                    verbose=verbose
                                                )

    optimizer = AdamW(
                    trainable_params,
                    lr=1e-3,          
                )

    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine = CosineAnnealingLR(optimizer, T_max=num_epochs-5)

    scheduler = SequentialLR(
                        optimizer,
                        schedulers=[warmup, cosine],
                        milestones=[5]
                    )
    
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
                    save_every = 1,
                    early_stopping_patience = 10,
                )
    
    finetuner.fit()
    finetuner.test()

    #-----------------------------------------------
    # Phase 2: Feature-extractor and Head finetuning
    #-----------------------------------------------

    bs = 2**7
    backbone_lr = 5e-5    
    head_lr = 1e-3             
    weight_decay = 0.05

    head_params = model.get_trainable_parameters(
                                            layers_to_train=[f'model.{output_layer}'],
                                            verbose=verbose
                                        )
    backbone_params = model.get_trainable_parameters(
                                            layers_to_train=layers_to_train,
                                            verbose=verbose
                                        )

    optimizer = AdamW(
                    [
                        {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': weight_decay},
                        {'params': head_params, 'lr': head_lr, 'weight_decay': weight_decay},
                    ],
                    betas=(0.9, 0.999),
                    eps=1e-8,
                )

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs-5)
    
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
                    max_epochs = 40,
                    iterations = 'full',
                    optimizer = optimizer,
                    scheduler = scheduler,
                    save_every = 1,
                    early_stopping_patience = 10,
                )
    
    finetuner.fit()
    finetuner.test()

    #-----------------------------------------------
    # Phase 3: Finetune the whole model
    #-----------------------------------------------

    bs = 2**7

    # after model is created / normalized
    
    trainable_params = model.get_trainable_parameters(
                                                    layers_to_train=None,
                                                    verbose=verbose
                                            )

    
    optimizer = AdamW(
                    trainable_params,
                    lr=1e-4,              
                    weight_decay=0.05,    
                    betas=(0.9, 0.999),
                    eps=1e-8,
                )

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs-5)
    
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
                    max_epochs = 70,
                    iterations = 'full',
                    optimizer = optimizer,
                    scheduler = scheduler,
                    early_stopping_patience = 10,
                    save_every = 1
                )
    
    finetuner.fit()
    finetuner.test()