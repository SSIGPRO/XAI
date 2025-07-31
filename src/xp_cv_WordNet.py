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
import clip
from nltk.corpus import wordnet as wn

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
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
    
    cvs_path = Path.cwd()/f'../data/{dataset}/corevectors'
    cvs_name = 'corevectors'

    drill_path = Path.cwd()/f'../data/{dataset}/drillers'
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

    with open(Path.cwd()/"../data/ImageNet/imagenet_class_index.json") as f:
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

    lbl = 'canine'
    text_inputs = clip.tokenize(f"a photo of a {lbl}").to(device)
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

        cv_dl = DataLoader(cv._corevds['train'][layer][...,:classifier_cv_dim], batch_size=bs, num_workers = n_threads)
        
        start = 0
        
        for data in tqdm(cv_dl):
            bs = data.shape[0]
            probs[start:start+bs] = drillers[layer]._classifier.predict_proba(data)
            start += bs

        conf, clusters = torch.max(probs, dim=1)
        labels_ = cv._dss['train']['label']

        dog_similarity = []

        for cluster in tqdm(range(n_cluster)):

            idx = torch.argwhere((clusters==cluster)).squeeze()
            images = cv._dss['train']['image'][idx]

            model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
            with torch.no_grad():
                
                image_features = model.encode_image(images.to(device))

                mean_image = image_features.mean(dim=0, keepdim=True)           
                mean_image = mean_image / mean_image.norm(dim=-1, keepdim=True)   

            similarity = mean_image @ text_embeds.t() 
            dog_similarity.append(similarity.squeeze().detach().cpu().numpy())

        indexed = list(enumerate(dog_similarity))
        
        plt.figure(figsize=(15, 5))
        plt.bar(np.arange(len(dog_similarity)), dog_similarity,linewidth=0)
        plt.xlabel('Sample Index')
        plt.ylabel('similarity wrt dog')
        plt.title('Similarity Plot wrt Dog')
        plt.savefig('prova_dog.png')

        indexed.sort(key=lambda pair: pair[1], reverse=True)
        with open('indexed_dog.txt', 'w') as f:
            for idx, sim in indexed:
                # write each pair on its own line, tab-separated:
                f.write(f"{idx}\t{sim}\n")