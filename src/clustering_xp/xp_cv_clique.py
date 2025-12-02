import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

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

import numpy as np

from pyclustering.cluster.clique import clique


def compute_empp_clique(hard_labels, y, n_classes):
    """
    Computes EM–PP for CLIQUE clustering.
    
    hard_labels: numpy array or tensor of shape [N], with noise = -1
    y:           true labels, shape [N]
    n_classes:   number of true classes
    """
    hard_labels = torch.tensor(hard_labels, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)

    # ignore noise (-1)
    valid = hard_labels >= 0
    valid_labels = hard_labels[valid]
    valid_y = y[valid]

    if len(valid_labels) == 0:
        raise ValueError("CLIQUE produced no valid clusters (all points marked noise).")

    # number of clusters = highest label + 1
    n_clusters = int(valid_labels.max().item() + 1)

    empp = torch.zeros((n_clusters, n_classes), dtype=torch.float32)

    for c in range(n_clusters):
        in_cluster = (valid_labels == c)

        if in_cluster.sum() == 0:
            continue

        labels = valid_y[in_cluster]

        # count class frequencies
        for s in range(n_classes):
            empp[c, s] = (labels == s).sum().item()

        # normalize to form probability distribution
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
        # 'features.7', 'features.10', 'features.12', 'features.14', 'features.17',
        # 'features.19', 'features.21', 'features.24', 'features.26', 'features.28',
        # 'classifier.0', 'classifier.3',
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
            X_reduced = X[:, :10]
                    
            X_np = X_reduced.cpu().numpy()

            y = ds._dss["CIFAR100-train"][:]["label"]  
            y_np = y.cpu().numpy()

            # -----------------------------
            # CLIQUE 
            # -----------------------------

            clique = clique(
                    data = X_np,
                    amount_intervals = 10, # grid bins per dimension.
                    density_threshold = 10 # min points required in a grid cell for it to be considered dense
            )

            clique.process()
            clusters = clique.get_clusters() 
            noise = clique.get_noise()

            pred = np.full(len(X_np), -1, dtype=int)
            for cid, pts in enumerate(clusters):
                pred[np.array(pts, dtype=int)] = cid

            pred_labels = torch.tensor(pred, dtype=torch.long)

            empp = compute_empp_clique(
                hard_labels=pred_labels,
                y=y_np,
                n_classes=n_classes
            )

            drillers[layer] = {
                "_empp": empp
            }

            # just to see whatsup
            unique, counts = torch.unique(pred_labels, return_counts=True)
            for u, c in zip(unique.tolist(), counts.tolist()):
                print(f"Cluster {u}: {c} points")
