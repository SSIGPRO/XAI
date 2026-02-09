import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# torch stuff
import torch
from torchvision.models import vgg16
from cuda_selector import auto_cuda

###### Our stuff

# Model
from peepholelib.models.model_wrap import ModelWrap 

# datasets
from peepholelib.datasets.cifar100 import Cifar100
from peepholelib.datasets.parsedDataset import ParsedDataset 

# corevecs
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.clip_embedding import get_clip_embeddings
from peepholelib.utils.viz_corevecs import *


import matplotlib.pyplot as plt

import cupy as cp

import torch
import matplotlib.pyplot as plt
from pathlib import Path
from CLIPwindow import *

def sliding_window_extremes_gallery(nu_layer,dss,ds_key,window_size,device,out_dir,out_name,step=None,max_windows=None, title=None):
    """
    For a layer corevector matrix nu_layer of shape (N, K), this function:
      - Builds sliding windows over the feature dimension (K) with size = window_size and step = step.
      - For each window, scores each sample by the mean of the window coordinates.
      - Selects the sample that maximizes the score and the sample that minimizes the score.
      - Skips adding a sample if it was already selected in either the max or min list.
      - Saves a single figure: max samples on the top row, min samples on the bottom row.
    """

    if step is None:
        step = window_size

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Move corevectors to GPU
    nu = nu_layer
    if not isinstance(nu, torch.Tensor):
        nu = torch.as_tensor(nu)
    nu = nu.to(device)

    if nu.ndim != 2:
        raise ValueError(f"nu_layer must be 2D (N,K). Got shape={tuple(nu.shape)}")

    n_samples, k = nu.shape
    if window_size <= 0 or window_size > k:
        raise ValueError(f"window_size must be in [1, K]. Got window_size={window_size}, K={k}")
    if step <= 0:
        raise ValueError("step must be > 0")

    windows = nu.unfold(dimension=1, size=window_size, step=step)
    scores = windows.mean(dim=2)

    n_windows = scores.shape[1]
    if max_windows is not None:
        n_windows = min(n_windows, int(max_windows))

    max_samples = []
    min_samples = []
    selected = set()

    # For each window, pick argmax/argmin sample index.
    for w in range(n_windows):
        col = scores[:, w]

        # Max candidate
        max_idx = int(torch.argmax(col).item())
       # if max_idx not in selected:
        max_samples.append(max_idx)
        selected.add(max_idx)

        # Min candidate
        min_idx = int(torch.argmin(col).item())
        #if min_idx not in selected:
        min_samples.append(min_idx)
        selected.add(min_idx)

    # Fetch images from your dataset container (CPU-side)
    _dss_max = dss._dss[ds_key][max_samples] if len(max_samples) > 0 else []
    _dss_min = dss._dss[ds_key][min_samples] if len(min_samples) > 0 else []

    # Build a compact gallery: 2 rows, C columns where C=max(#max, #min)
    cols = max(len(max_samples), len(min_samples), 1)
    fig, axes = plt.subplots(
        2, cols,
        figsize=(cols * 1.1, 2.6),
        constrained_layout=True
    )

    # If cols == 1, axes is (2,) not (2,1)
    if cols == 1:
        axes = axes.reshape(2, 1)

    if title is not None:
        fig.suptitle(title)

    # Helper to place images
    def _imshow(ax, d):
        img = d["image"].squeeze(dim=0).permute(1, 2, 0)
        ax.imshow(img.detach().cpu().numpy())
        ax.axis("off")

    # Top row: max
    for j in range(cols):
        ax = axes[0, j]
        if j < len(_dss_max):
            _imshow(ax, _dss_max[j])
            ax.set_title(f"max:{max_samples[j]}", fontsize=7)
        else:
            ax.axis("off")

    # Bottom row: min
    for j in range(cols):
        ax = axes[1, j]
        if j < len(_dss_min):
            _imshow(ax, _dss_min[j])
            ax.set_title(f"min:{min_samples[j]}", fontsize=7)
        else:
            ax.axis("off")

    save_path = out_dir / out_name
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

    return max_samples, min_samples




if __name__ == "__main__":
        use_cuda = torch.cuda.is_available()
        device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
        # torch.cuda.empty_cache()
        #device  = torch.device('cuda:2')
        #device = torch.device("cpu")
        print(f"Using {device} device")

        #--------------------------------
        # Directories definitions
        #--------------------------------
        cifar_path = '/srv/newpenny/dataset/CIFAR100'
        ds_path = Path('/srv/newpenny/XAI/CN/vgg_data/cifar100')

        model_dir = '/srv/newpenny/XAI/models'
        model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
        
        cvs_path = Path('/srv/newpenny/XAI/CN/vgg_data/corevectors')
        cvs_name = 'corevectors'

        emb_path = Path('/srv/newpenny/XAI/CN/embeddings')
        emb_name = 'embeddings'

        drill_path = Path('/srv/newpenny/XAI/CN/vgg_data/drillers_all/drillers_100')
        drill_name = 'classifier'


        plots_path = Path.cwd()/'temp_plots/min_max_samples'
        
        verbose = True 
        
        # Peepholelib
        target_layers = [ 'features.0', 'features.2', 'features.5','features.7', 'features.10', 'features.12', 'features.14', 'features.17', 'features.19', 'features.21',
                                 'features.24','features.26','features.28','classifier.0','classifier.3', 
                                 'classifier.6',
                        ]
        
        loaders = [
        'CIFAR100-train',
        'CIFAR100-val',
        'CIFAR100-test',
        ]

    #--------------------------------
    # Model 
    #--------------------------------
    
        nn = vgg16()
        print(target_layers)
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
                                                
        model.load_checkpoint(
                name = model_name,
                path = model_dir,
                verbose = verbose
                )
                                                
        model.set_target_modules(
                target_modules = target_layers,
                verbose = verbose
                )

        datasets = ParsedDataset(
                path = ds_path,
                )
    #--------------------------------
    # CoreVectors 
    #--------------------------------
        corevecs = CoreVectors(
                path = cvs_path,
                name = cvs_name,
                model = model,
                )

        embeds = CoreVectors(
                path = emb_path,
                name = emb_name,
                model = model,
                )

        with datasets as ds, corevecs as cv, embeds as embeds:
                ds.load_only(

                        loaders = loaders,
                        verbose = verbose
                        )
                #embeds.get_clip_embeddings(datasets=ds, device=device)
                embeds.load_only(
                        loaders = loaders,
                        verbose = verbose
                        )
                
                cv.load_only(
                        loaders = loaders,
                        verbose = verbose 
                        ) 
                
                layer = 'classifier.0'
                X = cv._corevds['CIFAR100-train'][layer]   # shape: [N, D]
                D = X.shape[1]

                window_size = 3
                save_base = Path('/home/claranunesbarrancos/repos/XAI/src/temp_plots/corevectors')

                for i, start in enumerate(range(0, D, window_size)):
                    end = min(start + window_size, D)

                    if end - start < window_size:
                        break

                    X_window = X[:, start:end]
                    X_np = X_window.cpu().numpy()

                    file_name = f"vgg_class0_superclass_tsne_dims_{start}_{end-1}"

                    plot_corevec3D(
                        corevector=cv,
                        ds=ds,
                        loader ='CIFAR100-test',
                        start_dim = start,
                        end_dim = 20,
                        save_path=save_base,
                        layer=layer,
                        file_name=file_name,
                        n_classes=10
                    )
                quit()
                label_and_plot_windows_for_layer(
                    emb = embeds,
                    layer=cv._corevds['CIFAR100-test']['features.0'],
                    window_size=10,
                    step = 10,
                    dss=ds,
                    ds_key='CIFAR100-test',
                    device=device,
                    out_dir=plots_path,
                    out_name='features_0_windows_10.png',
                    concept_data_dir= '/home/claranunesbarrancos/repos/XAI/data/concepts',
                    concept_filenames=['3k.txt', '10k.txt', '20k.txt', 'broden_labels_clean.txt', 'categories_places365.txt',
                    'imagenet_labels.txt'],
                    title ='Features.0 Layer CLIP Labels'
                    )


                # sliding_window_extremes_gallery(
                #     nu_layer=cv._corevds['CIFAR100-test']['features.0'],
                #     dss=ds,
                #     ds_key='CIFAR100-test',
                #     window_size=5,
                #     device=device,
                #     out_dir=plots_path,
                #     out_name='features_0_windows_5.png',
                # )
