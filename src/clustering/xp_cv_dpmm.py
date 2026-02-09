
import os
os.environ["OPENBLAS_NUM_THREADS"] = "64"
os.environ["OMP_NUM_THREADS"] = "64"
os.environ["MKL_NUM_THREADS"] = "64"
os.environ["NUMEXPR_NUM_THREADS"] = "64"

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
from peepholelib.utils.viz_empp import *


from sklearn.mixture import BayesianGaussianMixture


def compute_empp_dpmm(hard_labels, y, n_classes):
    """
    hard_labels: tensor[N]  (after noise reassignment)
    y: tensor[N] class labels
    """

    hard_labels = torch.tensor(hard_labels, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)

    # Map cluster labels to 0..n_clusters-1
    unique_clusters = hard_labels.unique()
    mapping = {old.item(): new for new, old in enumerate(unique_clusters)}
    mapped_labels = hard_labels.clone()
    for old, new in mapping.items():
        mapped_labels[hard_labels == old] = new

    n_clusters = len(unique_clusters)
    empp = torch.zeros((n_clusters, n_classes))

    # Count class frequencies per cluster
    for c in range(n_clusters):
        mask = mapped_labels == c
        if mask.sum() == 0:
            continue
        cluster_classes = y[mask]
        counts = torch.bincount(cluster_classes, minlength=n_classes)
        empp[c] = counts.float() / counts.sum()  # normalize to sum=1

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

            X = cv._corevds['CIFAR100-train'][layer][:]
            X_np = X.cpu().numpy()
            X_reduced = X[:, :51]

            y = ds._dss["CIFAR100-train"][:]["label"]  
            y_np = y.cpu().numpy()

            dpmm = BayesianGaussianMixture(
                n_components=100,
                weight_concentration_prior_type='dirichlet_process',
                weight_concentration_prior=1e-2,
                max_iter=500,
                init_params='kmeans',
                random_state=0,
            )

            # Fit model on corevectors
            dpmm.fit(X_reduced)


            pred_labels = torch.tensor(dpmm.predict(X_reduced), dtype=torch.int64)


            empp = compute_empp_dpmm(
                hard_labels=pred_labels,
                y=y_np,
                n_classes=n_classes
            )

            drillers[layer] = {
                "_empp": empp
            }

            # just to see whats up
            unique_clusters = pred_labels.unique()
            print(f"\n=== Layer: {layer} ===")
            print(f"Number of clusters: {len(unique_clusters)}")
            
            # Cluster sizes
            for c in unique_clusters:
                mask = pred_labels == c
                top_class = torch.mode(torch.tensor(y_np)[mask])[0].item()
                print(f"Cluster {c.item()}: size={mask.sum().item()}, top class={top_class}")


    coverage = empp_coverage_scores(
        drillers=drillers,
        threshold=0.8,

    )

