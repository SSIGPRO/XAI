import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial
from matplotlib import pyplot as plt
import numpy as np

from skimage.filters import threshold_otsu
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

from scipy.signal import find_peaks

# Our stuff 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 

# torch stuff
import torch

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=["vgg", "vit"], help="Model type to use")
parser.add_argument("--layer", required=True, help="Layer name to use")
parser.add_argument("--n_cluster", type=int, required=True, help="Clusters number")
parser.add_argument("--concept", required=True, help="concept to find")
args = parser.parse_args()

model_type = args.model
_layer = args.layer
n_cluster = args.n_cluster
_concept = args.concept

# Load config depending on model_type
if model_type == "vgg":
    from config.config_vgg import *
elif model_type == "vit":
    from config.config_vit import *
else:
    raise RuntimeError(
        "Select a configuration: vgg or vit"
    )

def find_thr_gmm(x):
    X = np.asarray(x).reshape(-1,1)
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0).fit(X)
    grid = np.linspace(X.min(), X.max(), 4096).reshape(-1,1)
    resp = gmm.predict_proba(grid)   # posteriori per comp.
    i = np.argmin(np.abs(resp[:,0] - resp[:,1]))
    return grid[i,0]

def find_thr_kde_valley(x):
    x = np.asarray(x).reshape(-1,1)
    n = len(x); std = x.std(ddof=1)
    bw = 1.06*std*n**(-1/5)  # Silverman
    kde = KernelDensity(bandwidth=bw).fit(x)
    grid = np.linspace(x.min(), x.max(), 4096).reshape(-1,1)
    dens = np.exp(kde.score_samples(grid))
    peaks, _ = find_peaks(dens) 
    if len(peaks) >= 2:
        top2 = np.argsort(dens[peaks])[-2:]
        a, b = np.sort(peaks[top2])
        valley = np.argmin(dens[a:b+1]) + a
        return grid[valley,0]
    return grid[np.argmin(dens),0]

if __name__ == "__main__":

    #--------------------------------
    # Directories definitions
    #--------------------------------

    drill_path = Path.cwd()/f'../../data/{model_name}/drillers'
    drill_name = 'classifier' 
        
    #--------------------------------
    # CoreVectors 
    #--------------------------------
    
    drillers = {_layer: tGMM(
                            path = drill_path,
                            name = drill_name+'.'+_layer,
                            nl_classifier = n_cluster,
                            nl_model = n_classes,
                            n_features = feature_sizes[_layer],
                            parser = cv_parsers[_layer],
                            device = device
                            )}
    
    concept_path = drillers[_layer]._clas_path/f'concept={_concept}'

    sim = torch.load(concept_path / f"distribution_similarity_{_concept}.pt")

    t_otsu = threshold_otsu(sim.cpu().numpy())
    t_gmm = find_thr_gmm(sim.cpu().numpy())
    t_kde = find_thr_kde_valley(sim.cpu().numpy())

    print(f"Otsu: {t_otsu:.4f} | GMM: {t_gmm:.4f} | KDE valley: {t_kde:.4f}")
    
    x = sim.detach().cpu().numpy().ravel()
    fig, ax = plt.subplots(figsize=(8,4))
    counts, bins, _ = ax.hist(x, bins=50, color='royalblue', alpha=0.8, edgecolor='none')
    ax.set_title("Distribuzione delle similarità con soglie")
    ax.set_xlabel("Similarità")
    ax.set_ylabel("Frequenza")

    grid = np.linspace(x.min(), x.max(), 2048)
    kde = KernelDensity(bandwidth=1.06*np.std(x, ddof=1)*len(x)**(-1/5)).fit(x.reshape(-1,1))
    dens = np.exp(kde.score_samples(grid.reshape(-1,1)))
    ax.plot(grid, dens * counts.max()/dens.max(), lw=2)

    for t, name, style, col in [
        (t_otsu, "Otsu", "--", "crimson"),
        (t_gmm,  "GMM",  ":",  "seagreen"),
        (t_kde,  "KDE valley", "-.", "purple"),
    ]:
        ax.axvline(t, ls=style, color=col, lw=2, label=f"{name} = {t:.4f}")

    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(concept_path/f'Similarity_distribution.png', dpi=300)
    

        