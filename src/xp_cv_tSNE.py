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
from cuda_selector import auto_cuda
import clip
from sklearn.manifold import TSNE

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
            #'classifier.0',
             'classifier.3',
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

    with open(Path.cwd()/"../data/ImageNet_/imagenet_class_index.json") as f:
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

    # text_inputs = [f"a photo of a {lbl}" for lbl in short_labels]
    text_inputs = clip.tokenize([f"a photo of a {lbl}" for lbl in short_labels]).to(device)
    text_back = clip.tokenize([" "]).to(device)
    
    with torch.no_grad():
        text_embeds = model.encode_text(text_inputs)
        text_back_embed = model.encode_text(text_back)
        text_embeds /= text_embeds.norm(dim=-1, keepdim=True)
        text_back_embed /= text_back_embed.norm(dim=-1, keepdim=True)
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
            # 'classifier.0': partial(
            #     trim_corevectors,
            #     module = 'classifier.0',
            #     cv_dim = classifier_cv_dim
            #     ),
            'classifier.3': partial(
                trim_corevectors,
                module = 'classifier.3',
                cv_dim = classifier_cv_dim
                ),
            # 'classifier.6': partial(
            #     trim_corevectors,
            #     module = 'classifier.6',
            #     cv_dim = classifier_cv_dim
            #     ),
            }

    feature_sizes = {
            #'classifier.0': classifier_cv_dim,
             'classifier.3': classifier_cv_dim,
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
        layer = 'classifier.3'

        n_samples = len(cv._corevds['train'][layer])                   

        probs = torch.empty(n_samples, n_cluster, dtype=torch.float32)

        cv_dl = DataLoader(cv._corevds['train'][layer][...,:classifier_cv_dim], batch_size=bs, num_workers = n_threads)
        
        start = 0
        
        for data in cv_dl:
            bs = data.shape[0]
            probs[start:start+bs] = drillers[layer]._classifier.predict_proba(data)
            start += bs

        conf, clusters = torch.max(probs, dim=1)
        labels_ = cv._dss['train']['label']

        allowed = torch.tensor([281, 13, 6, 4, 7], dtype=clusters.dtype, device=clusters.device)

        # build a mask: True where t is in `allowed`
        # (requires PyTorch ≥1.10; for older, see note below)
        mask = torch.isin(clusters, allowed)

        # zero out everything not in allowed
        clusters_filtered = torch.where(mask, clusters, torch.zeros_like(clusters))

        X = cv._corevds['train'][layer][...,:classifier_cv_dim]
        y = clusters

        # 2) Move to CPU + numpy if you’re in PyTorch/TensorFlow
        if hasattr(X, 'detach'):
            X = X.detach()
        if hasattr(X, 'cpu'):
            X = X.cpu()
        X = np.array(X)

        if hasattr(y, 'detach'):
            y = y.detach()
        if hasattr(y, 'cpu'):
            y = y.cpu()
        y = np.array(y)

        # 3) Run t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        X_2d = tsne.fit_transform(X)

        classes = np.unique(y)  # e.g. array([  4,   6,   7,  13, 281])
        print(classes)

        # 2) choose a qualitative colormap with >= len(classes) colors
        cmap = plt.get_cmap('tab10', len(classes))

        # 3) build a mapping label -> color
        label_to_color = {lab: cmap(i) for i, lab in enumerate(classes)}

        # 4) plot each class separately
        plt.figure(figsize=(8,6))
        for lab in classes:
            idx = (y == lab)
            plt.scatter(
                X_2d[idx, 0],
                X_2d[idx, 1],
                color=label_to_color[lab],
                label=str(lab),
                s=30,             # marker size
                alpha=0.8         # optional transparency
            )
        #plt.legend()

        plt.show()
        plt.savefig('prova.png')