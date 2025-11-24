import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

import cuml
cuml.accel.install()
import numpy as np


# torch stuff
import torch
from torchvision.models import vgg16
from cuda_selector import auto_cuda

###### Our stuff
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.datasets.cifar100 import Cifar100

from peepholelib.datasets.parsedDataset import ParsedDataset 


# corevecs
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import linear_svd_projection, conv2d_toeplitz_svd_projection

from cuml.cluster import AgglomerativeClustering
from cuml.decomposition import PCA


def compute_empp_aggl(hard_labels, y, n_classes):
    """
    hard_labels: tensor[N]  (after noise reassignment)
    soft_membership: tensor[N, C]
    y: tensor[N] class labels
    """

    hard_labels = torch.tensor(hard_labels)
    y = torch.tensor(y)

    n_clusters = int(hard_labels.max().item() + 1)
    empp = torch.zeros((n_clusters, n_classes))

    # compute P(class = s | c)
    for c in range(n_clusters):
        in_cluster = (hard_labels == c) 

        if in_cluster.sum() == 0:
            continue
 
        labels = y[in_cluster]                   

        # accumulate weighted counts
        for s in range(n_classes):
            empp[c, s] = (labels == s).sum().item()  

        # normalize 
        total = empp[c].sum()
        if total > 0:
            empp[c] /= total

    return empp

if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    torch.cuda.empty_cache()

    ds_path = Path.cwd()/'../data/datasets'

    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'

    cifar_path = '/srv/newpenny/dataset/CIFAR100'

    cvs_path = Path.cwd()/'../data/corevectors'
    cvs_name = 'corevectors'

    plots_path = Path.cwd()/'temp_plots/coverage/'

    target_layers = [
        'features.7', 'features.10', 'features.12', 'features.14', 'features.17',
        'features.19', 'features.21', 'features.24', 'features.26', 'features.28',
        'classifier.0', 'classifier.3',
        'classifier.6',
    ]

    loaders = ['CIFAR100-train', 'CIFAR100-val', 'CIFAR100-test']
    n_classes = len(Cifar100.get_classes(meta_path = Path(cifar_path)/'cifar-100-python/meta')) 


    verbose = True

    #--------------------------------
    # Dataset and CoreVectors 
    #--------------------------------
    
    dataset = ParsedDataset(
            path = ds_path,
            )

    corevecs = CoreVectors(
        path=cvs_path,
        name=cvs_name,
    )

    drillers = {}

    with corevecs as cv, dataset as ds:
        ds.load_only(
            loaders = loaders,
            verbose = verbose
            )

        cv.load_only(
            loaders=loaders,
            verbose=verbose
        )

        for layer in target_layers:

            X = cv._corevds['CIFAR100-train'][layer]
            X_reduced = X[:, :5]
            
            X_np = X_reduced.cpu().numpy()

            y = ds._dss["CIFAR100-train"][:]["label"]  
            y_np = y.cpu().numpy()


            aggl = AgglomerativeClustering(
                n_clusters=25,
                metric='euclidean',        
                linkage='single',            
                handle=None,                
                verbose=False,
                connectivity='knn',        
            )
            with cuml.accel.profile():                

                pred_labels = aggl.fit_predict(X_np)

                if hasattr(pred_labels, "to_array"):  
                    pred_labels = labels.to_array()
                pred_labels = torch.tensor(np.array(pred_labels, copy=False), dtype=torch.long)

            empp = compute_empp_aggl(
                hard_labels=pred_labels,
                y=y_np,
                n_classes=n_classes
            )

            drillers[layer] = {
                "_empp": empp
            }

            unique, counts = torch.unique(pred_labels, return_counts=True)
            for u, c in zip(unique.tolist(), counts.tolist()):
                print(f"Cluster {u}: {c} points")
    quit()
    coverage = empp_coverage_scores(
        drillers=drillers,
        threshold=0.8,
        plot=True,
        save_path=plots_path,
        file_name='coverage_aggl.png'
    )

