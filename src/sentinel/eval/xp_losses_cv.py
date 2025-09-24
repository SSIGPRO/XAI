import random
import torch
import torch.nn.functional as F
import sys
import matplotlib.pyplot as plt
import math 
#import pandas as pd
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/sentinel').as_posix())

from cuda_selector import auto_cuda

#peepholelib stuff
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.datasets.sentinel import Sentinel
from peepholelib.datasets.sentinel import SentinelWrap
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.models.sentinel.model_cv_sentinel import AE2D_instance, train_AE2D_instance

from model.xp_train_AE import test_iter

from configs.common import *
from configs.peepholes import *


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
    print(f'Using {device} device')
    
    
    model_path = Path.home()/'repos/XAI/data/cv_model'
    model_name = 'checkpoints.499.pt.pth'

    cv_model = AE2D_instance(
        num_sensors = num_sensors,
        seq_len = seq_len,
        kernel_size = kernel,
        embedding_size = emb_size,
        lay3 = lay3
    )
    
    model = ModelWrap(
            model = cv_model,
            device = device
        )

    model.load_checkpoint(
        path = model_path,
        name = model_name,
        sd_key = model_sd_key
    )

    model.set_target_modules(
        target_modules = target_layers
    )

    sentinel_ds = Sentinel(
        path = parsed_path
    )

    corevectors = CoreVectors(
        path = cvs_path,
        name = cvs_name,
        model = model
    )

    def evaluate_losses(predicted_dict, real_dict):
        losses = {}

        for feature in predicted_dict.keys():
            pred = torch.tensor(predicted_dict[feature])
            real = torch.tensor(real_dict[feature])
            
            loss = F.mse_loss(pred, real)
            losses[feature] = loss.item()
        return losses

    #-------------------------Reconstruct the Loss---------------------------#
    loss_fn = torch.nn.MSELoss(reduction='mean')
    pred = [] 
    data = [] 
    loss_clean = []

    window_size = 1000
    max_index = len(test_iter) 
 
    test_list = list(test_iter)
    start_idx = random.randint(0, max_index - window_size)
    end_idx = start_idx + window_size

    for i in range(start_idx, end_idx):
        input =test_list[i]['data'].to(device)
        target = test_list[i]['label'].to(device)

        input = input.view(input.size(0), 1, 1, -1)
        target = target.view(input.size(0), 1, 1, -1)

        output,_ = model(input)
        pred.extend([out.cpu().detach() for out in output])
        data.extend([out.cpu().detach() for out in input])
        
        with torch.no_grad():
            loss = loss_fn(output, target)  
            loss_clean.append(loss.item())  

    pred_ = torch.cat(pred, dim=1)
    data_ = torch.cat(data, dim=1)

    print(f'pred_.shape{pred_.shape}')
    print(f'data_.shape{data_.shape}')

    pred_dict = {
        target_layers[i]: pred_[:, i].flatten().tolist() for i in range(len(target_layers))  
    }
    data_dict = {
        target_layers[i]: data_[:, i].flatten().tolist() for i in range(len(target_layers))  
    }

    #------------------------------Visualize the Losses---------------------------#

    n_features = len(target_layers)  
    n_cols = 4
    n_rows = math.ceil(n_features / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3*n_rows))

    axes = axes.flatten()  
    for i, feature_name in enumerate(target_layers):
        ax = axes[i]
        ax.plot(data_dict[feature_name], label="Original", color="blue", alpha=0.7)
        ax.plot(pred_dict[feature_name], label="Predicted", color="red", linestyle="dashed", alpha=0.7)
        ax.set_title(f"Layer: {feature_name}")
        ax.legend()

    for j in range(len(target_layers), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Original vs Predicted data", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  
    plt.savefig('predvsreal.png')
    plt.show()
    plt.close()
    losses_clean = evaluate_losses(pred_dict, data_dict)
    #print(loss_clean)
