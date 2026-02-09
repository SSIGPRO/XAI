import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

import cuml
cuml.accel.install()

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

from sklearn.metrics import pairwise_distances
from cuml.cluster import DBSCAN 
import cupy as cp
import numpy as np
from scipy.spatial import distance




def compute_empp_dbscan(hard_labels, y, n_classes):
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


    plots_path = Path.cwd()/'temp_plots/coverage/'


    loaders = ['CIFAR100-train', 'CIFAR100-val', 'CIFAR100-test']
    
    cvs_path = Path('/srv/newpenny/XAI/CN/vgg_data/corevectors')
    cvs_name = 'corevectors'

    verbose = True

    target_layers = ['features.0', 'features.7','features.26','features.28','classifier.0','classifier.3', 'classifier.6']

    #--------------------------------
    # Dataset and CoreVectors 
    #--------------------------------
    
    dataset = ParsedDataset(
            path = ds_path,
            )
    print(dataset._dss['CIFAR100-test'].keys())
    quit()

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
            print(X.shape)
            method = 'cosine'
            if layer in {'features.0', 'features.7'}:
                X = X[:, :5]
                method = 'euclidean'
            
            y = ds._dss["CIFAR100-train"][:]["label"]  

            db = DBSCAN(
                eps=0.2,
                min_samples=50,
                metric = method,

            )
            # using a custom distance metrix  (manhattan)
            # X_distance = cp.array(X_reduced.detach().cpu().numpy())
            # dist_matrix = cp.array(pairwise_distances(X_distance.get(), metric='manhattan'))

            X = X.detach().cpu().numpy()  
            # VI = np.linalg.inv(np.cov(X, rowvar=False))
            # dist_matrix = distance.cdist(X, X, metric='mahalanobis', VI=VI)

            with cuml.accel.profile():
                # Fit model on corevectors
                pred_labels = db.fit_predict(X)

                if hasattr(pred_labels, "to_array"):  
                    pred_labels = pred_labels.to_array()
                pred_labels = torch.tensor(np.array(pred_labels, copy=False), dtype=torch.long)


            # # map labels to do the empirical posterior
            # unique_labels = pred_labels.unique()
            # mapping = {old.item(): new for new, old in enumerate(unique_labels)}
            # pred_labels_remapped = pred_labels.clone()
            # for old, new in mapping.items():
            #     pred_labels_remapped[pred_labels == old] = new

            # pred_labels = pred_labels_remapped

            # empp = compute_empp_dbscan(
            #     hard_labels=pred_labels,
            #     y=y.detach().int().cpu().numpy(),
            #     n_classes=n_classes
            # )

            # drillers[layer] = {
            #     "_empp": empp
            # }
            # print("emp: ", empp)

            # -----------------------------------------------------
            # some cluster diagnostics
            # -----------------------------------------------------
            unique_labels = pred_labels.unique()
            n_clusters = (unique_labels != -1).sum().item()
            n_noise = (pred_labels == -1).sum().item()

            print(f"\n=== Layer: {layer} ===")
            print(f"Clusters found: {n_clusters}")
            print(f"Noise points: {n_noise} / {len(pred_labels)}")

            # Cluster size distribution
            sizes = [(lbl.item(), (pred_labels == lbl).sum().item())
                    for lbl in unique_labels if lbl.item() != -1]
            sizes_sorted = sorted(sizes, key=lambda x: x[1], reverse=True)

            print("Top 5 cluster sizes:")
            for lbl, sz in sizes_sorted[:5]:
                print(f"   Cluster {lbl}: {sz} points")

    # coverage = empp_coverage_scores(
    #     drillers=drillers,
    #     threshold=0.8,
    # )

