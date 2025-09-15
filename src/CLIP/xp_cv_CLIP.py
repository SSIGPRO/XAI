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
parser.add_argument("--cv_dim", type=int, default=None, help="CoreVector dimension to use")
args = parser.parse_args()

model_type = args.model
_layer = args.layer
cv_dim = args.cv_dim

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
    # Text Encoding
    #-------------------------------- 

    with open(Path.cwd()/"../../data/vgg/imagenet_class_index.json") as f:
        class_idx = json.load(f)

    idx2label = {int(k): v[1] for k, v in class_idx.items()}
    synset_ids = [class_idx[str(i)][0] for i in range(1000)]
    short_labels = [class_idx[str(i)][1] for i in range(1000)]

    #--------------------------------
    # Models
    #--------------------------------

    model, preprocess = clip.load("ViT-L/14", device=device)

    #--------------------------------
    # Tokens
    #--------------------------------
    
    text_inputs = clip.tokenize([f"a photo of a {lbl}" for lbl in short_labels]).to(device)

    #text_inputs = clip.tokenize([f"a photo of a bird"]).to(device)
    
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
    
    n_cluster = 1000
    
    drillers = {_layer: tGMM(
                            path = drill_path,
                            name = drill_name+'.'+_layer,
                            nl_classifier = n_cluster,
                            nl_model = n_classes,
                            n_features = feature_sizes[_layer],
                            parser = cv_parsers[_layer],
                            device = device
                            )}

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

        if not cv_dim == None:

            cv_dl = DataLoader(cv._corevds['train'][_layer][...,:cv_dim], batch_size=bs, num_workers = n_threads)

        else:

            cv_dl = DataLoader(cv._corevds['train'][_layer], batch_size=bs, num_workers = n_threads)
        
        start = 0
        
        for data in cv_dl:
            bs = data.shape[0]
            probs[start:start+bs] = drillers[_layer]._classifier.predict_proba(data)
            start += bs

        conf, clusters = torch.max(probs, dim=1)
        
        for cluster in range(20, 50): # 

            idx = torch.argwhere((clusters==cluster)).squeeze()
            images = cv._dss['train']['image'][idx]

            with torch.no_grad():
                
                image_features = model.encode_image(images.to(device))

                mean_image = image_features.mean(dim=0, keepdim=True)           
                mean_image = mean_image / mean_image.norm(dim=-1, keepdim=True)   

            similarity = mean_image @ text_embeds.t()
            topk = 1
            values, indices = similarity[0].topk(topk)

            for score, idx in zip(values, indices):
                print(f"{short_labels[idx]}: {score.item():.3f}")
            
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
            
            fig.suptitle(f"{short_labels[similarity[0].topk(1)[1]]}: cluster population {len(images)}", fontsize=20, y=1.02)

            plt.tight_layout()
            fig.savefig(drillers[_layer]._clas_path/f'samples_cluster.{cluster}_.png', dpi=200, bbox_inches='tight')
            
        

        

            
    
