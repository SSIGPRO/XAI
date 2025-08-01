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

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Our stuff
from peepholelib.datasets.imagenet import ImageNet
from peepholelib.datasets.transforms import vgg16_imagenet as ds_transform 
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.models.svd_fns import linear_svd, conv2d_toeplitz_svd, conv2d_kernel_svd
from torch.utils.data import DataLoader

from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import linear_svd_projection, conv2d_toeplitz_svd_projection, conv2d_kernel_svd_projection

from peepholelib.peepholes.parsers import trim_corevectors, trim_channelwise_corevectors, trim_kernel_corevectors
from peepholelib.peepholes.classifiers.tkmeans import KMeans as tKMeans 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 

from peepholelib.utils.samplers import random_subsampling 

# torch stuff
import torch
from torchvision.models import vgg16
from cuda_selector import auto_cuda

# Xp speicific
import clip
from nltk.corpus import wordnet as wn

if __name__ == "__main__":
    # use_cuda = torch.cuda.is_available()
    # device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    # print(f"Using {device} device")

    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 4
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    ds_path = '/srv/newpenny/dataset/ImageNet_torchvision'

    # model parameters
    dataset = 'ImageNet' 
    seed = 29
    bs = 512 
    n_threads = 1
    
    cvs_path = Path.cwd()/f'../data/{dataset}_/corevectors'
    cvs_name = 'corevectors'

    drill_path = Path.cwd()/f'../data/{dataset}_/drillers'
    drill_name = 'classifier'
    
    verbose = True 
    
    # Peepholelib
    target_layers = [
            'classifier.0',
            # 'classifier.3',
            # 'classifier.6',
            ]
    
    #--------------------------------
    # Dataset 
    #--------------------------------

    ds = ImageNet(
            data_path = ds_path,
            dataset = dataset
            )

    ds.load_data(
            transform = ds_transform,
            seed = seed,
            )
    
    n_classes = len(ds.get_classes()) 

    with open(Path.cwd()/f"../data/{dataset}_/imagenet_class_index.json") as f:
        class_idx = json.load(f)

    idx2label = {int(k): v[1] for k, v in class_idx.items()}
    synset_ids = [class_idx[str(i)][0] for i in range(1000)]
    short_labels = [class_idx[str(i)][1] for i in range(1000)]

    #--------------------------------
    # Models
    #--------------------------------

    model, preprocess = clip.load("ViT-B/32", device=device)

    #--------------------------------
    # Tokens
    #--------------------------------
    # classes = list(ds._classes.values())
    # classes.append('reptile')

    # class_names = [tup[0] for tup in classes]
    #class_names = ['person','small furry mammal','analogical clock', 'gown','dress', 'curly dog', 'food', 'close up', 'mirror' ,'basket', 'ladybug', 'ladybird', 'dragonfly', 'reptile', 'mountain', 'human', 'amphibian', 'mammal', 'fish', 'brown object', 'wood', 'seaside', 'bed', 'countryside', 'bird', 'horse', 'skyscraper', 'white background', 'line', 'geometric form', 'dog', 'white and brown dog' ]
    #text_inputs = [f"a photo of a {label}" for label in class_names]

    text_inputs = [f"a photo of a {lbl}" for lbl in short_labels]
    text_inputs = clip.tokenize([f"a photo of a {lbl}" for lbl in short_labels]).to(device)
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
    
    classifier_cv_dim = 100
    n_cluster = 500

    cv_parsers = {
            'classifier.0': partial(
                trim_corevectors,
                module = 'classifier.0',
                cv_dim = classifier_cv_dim
                ),
            # 'classifier.3': partial(
            #     trim_corevectors,
            #     module = 'classifier.3',
            #     cv_dim = classifier_cv_dim
            #     ),
            # 'classifier.6': partial(
            #     trim_corevectors,
            #     module = 'classifier.6',
            #     cv_dim = classifier_cv_dim
            #     ),
            }

    feature_sizes = {
            'classifier.0': classifier_cv_dim,
            # 'classifier.3': classifier_cv_dim,
            # 'classifier.6': classifier_cv_dim,
            }

    drillers = {}
    for peep_layer in target_layers:
        drillers[peep_layer] = tGMM(
                path = drill_path,
                name = drill_name+'.'+peep_layer,
                nl_classifier = n_cluster,
                nl_model = n_classes,
                n_features = feature_sizes[peep_layer],
                parser = cv_parsers[peep_layer],
                device = device
                )

    # fitting classifiers
    with corevecs as cv:
        cv.load_only(
                loaders = ['train', 'val'],
                verbose = True
                ) 

        for drill_key, driller in drillers.items():
            print(f'Loading Classifier for {drill_key}') 
            driller.load()
        layer = 'classifier.0'

        n_samples = len(cv._corevds['train'][layer])                   

        probs = torch.empty(n_samples, n_cluster, dtype=torch.float32)

        cv_dss = cv._corevds['train'][layer][:,:classifier_cv_dim]
        clust = AgglomerativeClustering(
                                    n_clusters=100,        # or None if you use distance_threshold
                                    metric='euclidean', # distance metric: ‘euclidean’ is default
                                    linkage='ward'       # ‘ward’, ‘complete’, ‘average’ or ‘single’
                                )

        labels = clust.fit_predict(cv_dss)

        Z = linkage(cv_dss, method='ward', metric='euclidean')

        plt.figure(figsize=(12, 6))
        dendrogram(
            Z,
            truncate_mode='level',     # uncomment to show only the top p levels
            #p=3,                        # how many levels to keep if truncating
            )
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Sample index")
        plt.ylabel("Distance")
        plt.legend()
        plt.show()
        plt.savefig('prova.png')
        cluster_labels = fcluster(Z, t=5, criterion='distance')
        n_clusters = len(np.unique(cluster_labels))
        for cluster_id in np.unique(cluster_labels):
            samples_in_cluster = np.where(cluster_labels == cluster_id)[0]
            print(f"Cluster {cluster_id}: {len(samples_in_cluster)} samples")
            print(f"  Samples: {list(samples_in_cluster)}")
        
        quit()
        start = 0
        
        for data in cv_dl:
            bs = data.shape[0]
            probs[start:start+bs] = drillers[layer]._classifier.predict_proba(data)
            start += bs

        conf, clusters = torch.max(probs, dim=1)
        labels_ = cv._dss['train']['label']

        for cluster in range(50):

            idx = torch.argwhere((clusters==cluster)).squeeze()
            images = cv._dss['train']['image'][idx]

            model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
            with torch.no_grad():
                
                image_features = model.encode_image(images.to(device))

                mean_image = image_features.mean(dim=0, keepdim=True)           
                mean_image = mean_image / mean_image.norm(dim=-1, keepdim=True)   

            similarity = mean_image @ text_embeds.t()  

            topk = 10
            values, indices = similarity[0].topk(topk)

            for score, idx in zip(values, indices):
                print(f"{short_labels[idx]}: {score.item():.3f}")
            
            if len(images) < 50:
                num_images = len(images) -2
            else:
                num_images = 50

            # choose number of columns
            cols = 7
            rows = math.ceil(num_images / cols)

            # make a big enough figure
            fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

            # flatten the axes array for easy indexing

            axs = axs.flatten()

            # inv_norm = transforms.Normalize(
            #     mean=[-m/s for m, s in zip([0.229, 0.224, 0.225], [0.485, 0.456, 0.406])],
            #     std =[ 1/s   for s   in [0.485, 0.456, 0.406]]
            # )
            # denorm = inv_norm(tensor_img)

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
            
            fig.suptitle(f"{short_labels[similarity[0].topk(1)[1]]}: cluster population {len(images)}", fontsize=20, y=1.02)

            plt.tight_layout()
            fig.savefig(drillers[layer]._clas_path/f'samples_cluster.{cluster}.png', dpi=200, bbox_inches='tight')
        

        

            
    
