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
from sklearn.neighbors import KernelDensity

from scipy.signal import find_peaks

# Our stuff
from torch.utils.data import DataLoader

from peepholelib.coreVectors.coreVectors import CoreVectors

from peepholelib.peepholes.classifiers.tkmeans import KMeans as tKMeans 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 

from peepholelib.wordNet_utils.wordnet import *

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
parser.add_argument("--prefix", type=str, default=None, help="prefilter prefix to choose the synset")
args = parser.parse_args()

model_type = args.model
_layer = args.layer
cv_dim = args.cv_dim
n_cluster = args.n_cluster
_concept = args.concept
prefilter_prefix = args.prefix

# Load config depending on model_type
if model_type == "vgg":
    from config.config_vgg import *
elif model_type == "vit":
    from config.config_vit import *
else:
    raise RuntimeError(
        "Select a configuration: vgg or vit"
    )

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
    ds_path = '/srv/newpenny/dataset/ImageNet_torchvision'

    # model parameters
    bs = 2**12 
    n_threads = 1
    
    cvs_path = Path.cwd()/f'../../data/{model_name}/corevectors'
    cvs_name = 'corevectors'

    embeds_path = Path.cwd()/f'../../data/{model_name}/corevectors'
    embeds_name = 'CLIP_embeddings_ViT-L14'

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

    embeds = CoreVectors(
            path = embeds_path,
            name = embeds_name,
            model = model,
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
    with corevecs as cv, embeds as em:
        cv.load_only(
                loaders = ['train', 'val'],
                verbose = True
                ) 
        
        em.load_only(
                loaders = ['train', 'val'],
                verbose = True
                )

        for drill_key, driller in drillers.items():
            print(f'Loading Classifier for {drill_key}') 
            driller.load()
        
        check_path = concept_path/f"distribution_similarity_{_concept}.pt"

        n_samples = len(cv._corevds['train'][_layer])                   

        probs = torch.empty(n_samples, n_cluster, dtype=torch.float32)

        cv_dl = DataLoader(cv._corevds['train'][_layer][...,:cv_dim], batch_size=bs, num_workers = n_threads)
        
        start = 0
        
        for data in tqdm(cv_dl):
            bs = data.shape[0]
            probs[start:start+bs] = drillers[_layer]._classifier.predict_proba(data)
            start += bs

        conf, clusters = torch.max(probs, dim=1)

        if check_path.exists():

            sim = torch.load(concept_path / f"distribution_similarity_{_concept}.pt")

        else:        

            sim = torch.zeros(n_cluster)
            
            for cluster in tqdm(range(n_cluster)): 

                idx = torch.argwhere((clusters==cluster)).squeeze()

                mean_em = em._corevds['train']['embedding'][idx].mean(dim=0, keepdim=True)           
                mean_em = mean_em / mean_em.norm(dim=-1, keepdim=True)   

                sim[cluster] = mean_em.float().to(device) @ text_embeds.float().t()

            torch.save(sim.cpu(), concept_path / f"distribution_similarity_{_concept}.pt") 

        sim_values = sim.flatten().cpu()

        #--------------------------------
        # Threshold choice 
        #--------------------------------

        t_kde = find_thr_kde_valley(sim.cpu().numpy())

        print(f"KDE valley: {t_kde:.4f}")
        
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
            (t_kde,  "KDE valley", "-.", "orange"),
        ]:
            ax.axvline(t, ls=style, color=col, lw=2, label=f"{name} = {t:.4f}")
        plt.axvspan(t_kde, sim_values.max().item(), alpha=0.15, color="orange", label="selected clusters")
        ax.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig(concept_path/f'Similarity_distribution.png', dpi=300)
        
        #--------------------------------
        # Groundtruth extraction 
        #--------------------------------

        seed_synsets = get_synsets_from_concept(_concept, lang='eng')
        print(f"Seed synsets for '{_concept}': {[s.name() for s in seed_synsets]}")

        if len(seed_synsets) > 1:
            seed_synsets = choose_seed_synsets_cli(_concept, lang='eng', prefilter_prefix=prefilter_prefix)

        print(f"Seed synsets for '{_concept}': {[s.name() for s in seed_synsets]}")

        if not seed_synsets:
            raise RuntimeError(f"No synset found '{_concept}'")

        subtree = hyponym_closure_synsets(seed_synsets)
        subtree_wnids = {synset_to_wnid(s) for s in subtree} | {synset_to_wnid(s) for s in seed_synsets}
        
        with open(Path.cwd()/f'../../data/{model_name}/imagenet_class_synset.json', "r", encoding="utf-8") as f:
            wnid_to_idx_label = json.load(f) 

        target = {}
        for wnid in subtree_wnids:
            if wnid in wnid_to_idx_label:
                idx, label = wnid_to_idx_label[wnid]
                target[idx] = (wnid, label)

        list_labels = torch.tensor(sorted(target.keys()))

        mask_labels = torch.isin(cv._dss['train']['label'], list_labels)           
        tot_dataset = int(mask_labels.sum())

        print(f'Total number of samples associated to {_concept}: ', tot_dataset)
        
        #-----------------------------------------
        # Evaluating quality of the clustering
        #-----------------------------------------

        top_mask = sim >= t_kde
                           
        cluster_ids = torch.nonzero(top_mask).squeeze(1)  # indices of clusters
        top_vals = sim[top_mask]
        # print("Clusters at/above KDE thr:", cluster_ids.tolist())
        # print("Their sim values:", top_vals.tolist())

        mask_clusters = torch.isin(clusters,  cluster_ids)    

        # AND element-wise
        mask_notlabels = ~mask_labels & mask_clusters
        mask_both = mask_labels & mask_clusters 
        mask_excl = mask_labels & (~mask_clusters)

        tot_cluster = int(mask_clusters.sum())
        print(f'Total number of samples within the selected clusters: ', tot_cluster)

        tot_busted = int(mask_both.sum())
        print(f'Total number of samples within the selected clusters associated to {_concept}: ', tot_busted)

        tot_excl = int(mask_excl.sum())
        print(f'Total number of samples outside the selected clusters associated to {_concept}: ', tot_excl)

        print(f'Samples that do not relate to the concept {_concept} within the selected clusters: {mask_notlabels.sum()}')

        print(f'Correctly classified samples within the selected clusters: {cv._dss['train']['result'][mask_both].sum()/tot_dataset*100 :.2f} %')
        print(f'Correctly classified samples excluded from the selected clusters: {cv._dss['train']['result'][mask_excl].sum()/tot_dataset*100 :.2f} %')

        print(f'Wrongly classified samples within the selected clusters: {(1-cv._dss['train']['result'][mask_both]).sum()/tot_dataset*100 :.2f} %')
        print(f'Wrongly classified samples excluded from the selected clusters: {(1-cv._dss['train']['result'][mask_excl]).sum()/tot_dataset*100 :.2f} %')

        print(f'Correctly classified samples by the NN: {cv._dss['train']['result'][mask_labels].sum()/tot_dataset*100:.2f}%')
        print(f'Wrongly classified samples by the NN: {(1-cv._dss['train']['result'][mask_labels].sum()/tot_dataset)*100:.2f}%')

        res = cv._dss['train']['result']

        res_mask = (res == 0) if not res.dtype.is_floating_point else (res < 0.5)

        res_mask = res_mask.view_as(mask_excl)   # o mask_excl = mask_excl.view_as(res_mask)

        idx = torch.nonzero(mask_excl & res_mask).squeeze(1) 

        sel = clusters[idx].to(torch.long)

        counts = torch.bincount(sel, minlength=n_cluster)       # [n_cluster]
        ids    = torch.arange(n_cluster, device=counts.device)  # [n_cluster]

        # keep only non-zero frequency
        nz = counts > 0
        ids_nz    = ids[nz].cpu().numpy()
        counts_nz = counts[nz].cpu().numpy()
        freq_nz   = counts_nz / counts_nz.sum()

        order = np.argsort(counts_nz)[::-1]
        ids_plot, counts_plot, freq_plot = ids_nz[order], counts_nz[order], freq_nz[order]

        plt.figure(figsize=(9,4))
        plt.bar(ids_plot, counts_plot)
        plt.xlabel("Cluster ID"); plt.ylabel("Count"); plt.title("mask_excl: non-zero clusters")
        for x,c in zip(ids_plot, counts_plot):
            plt.text(x, c, f"{int(c)}", ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.show()
        plt.savefig(concept_path/'Cluster_population_mask_excl.png', dpi=300)
        plt.close()

        ids_t, counts_t = torch.unique(sel, sorted=True, return_counts=True)  # only non-zero
        ids    = ids_nz.tolist()
        counts = counts_nz.tolist()
        total  = sum(counts)

        # sort by count (desc)
        order = sorted(range(len(ids)), key=lambda i: counts[i], reverse=True)

        with open(concept_path/"cluster_counts_mask_excl.txt", "w", encoding="utf-8") as f:
            f.write("# mask_excl cluster counts (non-zero only)\n")
            f.write(f"# total_selected_samples: {total}\n")
            f.write("cluster_id\tcount\tpct\n")
            for i in order:
                pct = 100.0 * counts[i] / total
                f.write(f"{ids[i]}\t{counts[i]}\t{pct:.2f}\n")
        
        pairs = sorted(
            zip(cluster_ids.tolist(), [float(v) for v in top_vals]),
            key=lambda x: x[1],
            reverse=True
        )

        txt = concept_path / "clusters_KDE_sorted.txt"

        with txt.open("w") as f:
            f.write(f"KDE threshold: {t_kde.item():.6f}\n")
            f.write("cluster_id\tsim\n")
            for cid, val in pairs:
                f.write(f"{cid}\t{val:.6f}\n")

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
            fig.savefig(concept_path/f'samples_cluster.{cluster}.png', dpi=200, bbox_inches='tight')
            plt.close(fig)