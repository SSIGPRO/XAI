import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial
from matplotlib import pyplot as plt
import math 
import numpy as np
import json
from tqdm import tqdm
from skimage.filters import threshold_otsu

# Our stuff
from torch.utils.data import DataLoader

from peepholelib.coreVectors.coreVectors import CoreVectors

from peepholelib.peepholes.classifiers.tkmeans import KMeans as tKMeans 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 

from peepholelib.utils.samplers import random_subsampling 

# torch stuff
import torch
import clip
from nltk.corpus import wordnet as wn

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=["vgg", "vit"], help="Model type to use")
parser.add_argument("--layer", required=True, help="Layer name to use")
parser.add_argument("--cv_dim", type=int, required=True, help="corevector dimensionality")
parser.add_argument("--n_cluster", type=int, required=True, help="Clusters number")
parser.add_argument("--concept", required=True, help="concept to find")
args = parser.parse_args()

model_type = args.model
_layer = args.layer
cv_dim = args.cv_dim
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

if __name__ == "__main__":

    #--------------------------------
    # Directories definitions
    #--------------------------------
    ds_path = '/srv/newpenny/dataset/ImageNet_torchvision'

    # model parameters
    bs = 512 
    n_threads = 1
    
    cvs_path = Path.cwd()/f'../../data/{model_name}/corevectors'
    cvs_name = 'corevectors'

    drill_path = Path.cwd()/f'../../data/{model_name}/drillers'
    drill_name = 'classifier'
    
    verbose = True 

    #--------------------------------
    # Models
    #--------------------------------

    model, _ = clip.load("ViT-L/14", device=device)

    #--------------------------------
    # Tokens
    #--------------------------------
    
    text_inputs = clip.tokenize([f"a photo of a {_concept}"]).to(device)
    
    with torch.no_grad():
        text_embeds = model.encode_text(text_inputs)   
        text_embeds /= text_embeds.norm(dim=-1, keepdim=True)
        
    #--------------------------------
    # CoreVectors 
    #--------------------------------
    
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )
    
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
    concept_path.mkdir(parents=True, exist_ok=True)

    # fitting classifiers
    with corevecs as cv:
        cv.load_only(
                loaders = ['train', 'val'],
                verbose = True
                ) 

        for drill_key, driller in drillers.items():
            print(f'Loading Classifier for {drill_key}') 
            driller.load()

        n_samples = len(cv._corevds['train'][_layer])                   

        probs = torch.empty(n_samples, n_cluster, dtype=torch.float32)

        cv_dl = DataLoader(cv._corevds['train'][_layer][...,:cv_dim], batch_size=bs, num_workers = n_threads)
        
        start = 0
        
        for data in cv_dl:
            bs = data.shape[0]
            probs[start:start+bs] = drillers[_layer]._classifier.predict_proba(data)
            start += bs

        conf, clusters = torch.max(probs, dim=1)

        check_path = concept_path/f"distribution_similarity_{_concept}.pt"

        if check_path.exists():

            sim = torch.load(concept_path / f"distribution_similarity_{_concept}.pt")

        else:        

            sim = torch.zeros(n_cluster)
            
            for cluster in tqdm(range(n_cluster)): 

                idx = torch.argwhere((clusters==cluster)).squeeze()
                images = cv._dss['train']['image'][idx]

                with torch.no_grad():
                    
                    image_features = model.encode_image(images.to(device))

                    mean_image = image_features.mean(dim=0, keepdim=True)           
                    mean_image = mean_image / mean_image.norm(dim=-1, keepdim=True)   

                sim[cluster] = mean_image @ text_embeds.t()

            torch.save(sim.cpu(), concept_path / f"distribution_similarity_{_concept}.pt") 

        sim_values = sim.flatten().cpu()

        # Plot histogram
        plt.hist(sim_values.numpy(), bins=50, alpha=0.7, color="blue")
        plt.xlabel("Similarity")
        plt.ylabel("Frequency")
        plt.title("Distribution of similarities")
        plt.savefig('prova.png')

        threshold = threshold_otsu(sim_values.numpy())
        print("Otsu threshold:", threshold)

        # Take the 95th percentile
        percentile_90 = torch.quantile(sim_values, 0.90)
        print("90th percentile:", percentile_90.item())

        # If you want the *values* above the 95th percentile
        top_values = sim_values[sim_values >= percentile_90]
        print("Number of values above 90th percentile:", top_values.numel())

        thr = torch.quantile(sim.float().flatten(), 0.90)

        top_mask = sim >= thr                         # [num_clusters]
        cluster_ids = torch.nonzero(top_mask).squeeze(1)  # indices of clusters
        top_vals = sim[top_mask]
        print("Clusters at/above 90th pct:", cluster_ids.tolist())
        print("Their sim values:", top_vals.tolist())

        pairs = sorted(
            zip(cluster_ids.tolist(), [float(v) for v in top_vals]),
            key=lambda x: x[1],
            reverse=True
        )

        txt = concept_path / "clusters_95pct_sorted.txt"

        with txt.open("w") as f:
            f.write(f"90th percentile threshold: {thr.item():.6f}\n")
            f.write("cluster_id\tsim\n")
            for cid, val in pairs:
                f.write(f"{cid}\t{val:.6f}\n")

        plt.hist(sim_values.numpy(), bins=50, alpha=0.7, color="blue")
        plt.xlabel("Similarity")
        plt.ylabel("Frequency")
        thr_val = float(thr.item())
        plt.axvline(thr_val, linestyle="--", linewidth=2, color="red",
                    label=f"90th pct = {thr_val:.4f}")
        plt.axvline(threshold, linestyle="--", linewidth=2, color="green",
                    label=f"Otsu = {threshold:.4f}")

        # (optional) shade the top 5% region
        plt.axvspan(thr_val, sim_values.max().item(), alpha=0.15, color="red", label="top 10%")

        plt.legend()
        plt.tight_layout()
        plt.title("Distribution of similarities")
        plt.savefig(concept_path/'Similarity_distribution.png')
        quit()

        for cluster, sim in tqdm(zip(cluster_ids.tolist(),top_vals.tolist())):
            idx = torch.argwhere((clusters==cluster)).squeeze()
            images = cv._dss['train']['image'][idx]

            if len(images) <= 50:
                num_images = len(images)-5
            else:
                num_images = 40

            # choose number of columns
            cols = 5
            rows = math.ceil(num_images / cols)

            # make a big enough figure
            fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

            # flatten the axes array for easy indexing

            axs = axs.flatten()

            for i, ax in enumerate(axs):

                # show the image
                img = images[i].detach().cpu().numpy().transpose(1,2,0)
                img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
                img = np.clip(img, 0, 1)
                ax.imshow(img)
                # ax.imshow(img.detach().cpu().numpy().transpose(1,2,0))
                # turn off ticks & frame
                ax.axis('off')

            # turn off any remaining empty subplots
            for ax in axs[num_images:]:
                ax.axis('off')
            
            fig.suptitle(f"{sim}: cluster population {len(images)}", fontsize=20, y=1.02)

            plt.tight_layout()
            fig.savefig(concept_path/f'samples_cluster.{cluster}_.png', dpi=200, bbox_inches='tight')