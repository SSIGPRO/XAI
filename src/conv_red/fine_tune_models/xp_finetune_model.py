import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/conv_red').as_posix())
from datetime import datetime

# torch stuff
import torch
from cuda_selector import auto_cuda

# Peepholelibe stuff
from peepholelib.datasets.cifar100 import Cifar100
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.training.trainingBase import Trainer

# Our stuff
from configs.common import *
from configs.fine_tune_models import *

if __name__ == "__main__":
    print(f'{args}') 
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tune_dir = args.data_path/"checkpoints"/args.model 
    tune_name = 'model'

    #--------------------------------
    # Dataset 
    #--------------------------------

    dataset = Cifar100(
            path = cifar_path,
            std_transform = transform,
            aug_transform = augmentation
            )

    dataset.__load_data__()

    #--------------------------------
    # Model 
    #--------------------------------

    nn = Model(weights=pre_train_weights.DEFAULT)

    model = ModelWrap(
            model = nn,
            device = device
            )

    model.update_output(
            output_layer = output_layer,
            to_n_classes = n_classes
            )

    model.prepend_normalizer(
            mean = normalization_mean,
            std = normalization_std
            )

    #-----------------------------------------------
    # Finetune the model
    #-----------------------------------------------
    
    # after model is created / normalized
    trainable_params = model.get_trainable_parameters(
            layers_to_train = None,
            verbose = verbose
            )

    optimizer = Optim(
            trainable_params,
            **opt_kwargs,
            )
    
    scheduler = Scheduler(
            optimizer,
            **scheduler_kwargs
            )

    finetuner = Trainer(
            model = model,
            path = tune_dir,
            name = tune_name,
            dataset = dataset,
            train_key = f'{dataset_name}-train',
            val_key = f'{dataset_name}-val',
            test_key = f'{dataset_name}-test',
            batch_size = model_fine_tune_bs,
            dataloader_kwargs = dl_kwargs,
            max_epochs = max_epochs,
            iterations = iterations,
            optimizer = optimizer,
            scheduler = scheduler,
            early_stopping_patience = max_epochs,
            save_every = save_every 
            )
    
    finetuner.fit()
    finetuner.test()
