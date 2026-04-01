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
from torchvision.models import ConvNeXt_Base_Weights as pre_train_weights
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
    output_layer = 'classifier.2'

    #--------------------------------
    # Directories definitions
    #--------------------------------
    ds_path = '/srv/newpenny/dataset/CIFAR100'
    basic_dir = Path.cwd()/(f'../../data')# basic_dir = Path("/home/arshakumari/repos/XAI/src/Conv_NeXt")

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

    nn = convnext_base(weights=pre_train_weights.DEFAULT)
    # for name, module in nn.named_modules():
    #     print(f"{name}: {type(module).__name__}")
    # for k in nn.state_dict().keys():
    #     print(k)
    # quit()

    n_classes = len(Cifar100.get_classes(meta_path = Path(ds_path)/'cifar-100-python/meta'))

    model = ModelWrap(
            model = nn,
            device = device
            )
    for name, _ in model._model.named_parameters():
        print(name)
    
    model.update_output(output_layer=output_layer, to_n_classes=n_classes)

    model.normalize_model(mean=means[name_dataset], std= stds[name_dataset])

    layers_to_train = [
        name for name, _ in model._model.named_parameters()
    # Stage 3 (deepest stage) blocks
    # 'features.3.0.block.0.weight',
    # 'features.3.0.block.0.bias',
    # 'features.3.0.block.2.weight',
    # 'features.3.0.block.2.bias',
    # 'features.3.0.block.3.weight',
    # 'features.3.0.block.3.bias',
    # 'features.3.0.block.5.weight',
    # 'features.3.0.block.5.bias',

    # 'features.3.1.block.0.weight',
    # 'features.3.1.block.0.bias',
    # 'features.3.1.block.2.weight',
    # 'features.3.1.block.2.bias',
    # 'features.3.1.block.3.weight',
    # 'features.3.1.block.3.bias',
    # 'features.3.1.block.5.weight',
    # 'features.3.1.block.5.bias',

    # 'features.3.2.block.0.weight',
    # 'features.3.2.block.0.bias',
    # 'features.3.2.block.2.weight',
    # 'features.3.2.block.2.bias',
    # 'features.3.2.block.3.weight',
    # 'features.3.2.block.3.bias',
    # 'features.3.2.block.5.weight',
    # 'features.3.2.block.5.bias',

    # # Classifier (always train)
    # 'classifier.2.weight',
    # 'classifier.2.bias'
]

    #----------------------------
    # Phase 1: Head-only Warm-up
    #----------------------------

    ## DataLoader

    dl_kwargs = dict(
        collate_fn=default_collate,
        num_workers=n_threads,
        pin_memory=device.type == "cuda",
        persistent_workers=n_threads > 0,
    )
    if n_threads > 0:
        dl_kwargs["prefetch_factor"] = 2

    bs = 2**2#10
    num_epochs = 1#20

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
                    iterations = 1,#'full',
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

    # bs = 2**7
    # backbone_lr = 5e-5    
    # head_lr = 1e-3             
    # weight_decay = 0.05

    # head_params = model.get_trainable_parameters(
    #                                         layers_to_train=[f'model.{output_layer}'],
    #                                         verbose=verbose
    #                                     )
    # backbone_params = model.get_trainable_parameters(
    #                                         layers_to_train=layers_to_train,
    #                                         verbose=verbose
    #                                     )

    # optimizer = AdamW(
    #                 [
    #                     {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': weight_decay},
    #                     {'params': head_params, 'lr': head_lr, 'weight_decay': weight_decay},
    #                 ],
    #                 betas=(0.9, 0.999),
    #                 eps=1e-8,
    #             )

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
    #                 max_epochs = 40,
    #                 iterations = 'full',
    #                 optimizer = optimizer,
    #                 scheduler = scheduler,
    #                 save_every = 1,
    #                 early_stopping_patience = 10,
    #             )
    
    # finetuner.fit()
    # finetuner.test()

    #-----------------------------------------------
    # Phase 3: Finetune the whole model
    #-----------------------------------------------

    bs = 128 
    # after model is created / normalized
    
    trainable_params = model.get_trainable_parameters(
                                                    layers_to_train=None,
                                                    verbose=verbose
                                            )

    
    optimizer = AdamW(
                    trainable_params,
                    lr=5e-5,              
                    weight_decay=1e-8,    
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
                    max_epochs = 30,
                    iterations = 1,#'full',
                    optimizer = optimizer,
                    scheduler = scheduler,
                    early_stopping_patience = 10,
                    save_every = 1
                )
    
    finetuner.fit()
    finetuner.test()
