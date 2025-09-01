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
from torchvision.models import vgg16
from cuda_selector import auto_cuda
import clip
from nltk.corpus import wordnet as wn
from diffusers import StableDiffusionPipeline
from PIL import Image

def generate_with_image_steering(pipe, image_cross_attn_embeds, num_steps=50, guidance_scale=7.5, height=512, width=512):   
    """
    Generate images using CLIP image embeddings instead of text embeddings
    for cross-attention steering in UNet
    """
    device = pipe.device

    # Create unconditional (negative) embeddings for classifier-free guidance
    uncond_cross_attn_embeds = torch.zeros_like(image_cross_attn_embeds)
    
    # Concatenate for CFG: [unconditional, conditional]
    encoder_hidden_states = torch.cat([uncond_cross_attn_embeds, image_cross_attn_embeds])
    
    # Initialize random latents
    latents = torch.randn(
        (1, 4, height // 8, width // 8),
        device=device,
        dtype=pipe.unet.dtype
    )
    
    # Setup scheduler
    pipe.scheduler.set_timesteps(num_steps)
    latents = latents * pipe.scheduler.init_noise_sigma
    
    # Denoising loop with image-guided cross-attention
    for i, t in enumerate(pipe.scheduler.timesteps):
        # Expand latents for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        # UNet forward pass - image embeddings steer via cross-attention
        with torch.no_grad():
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,  # Image embeds → cross-attention
                return_dict=False
            )[0]
        
        # Classifier-free guidance
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        # Scheduler step
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        if i % 10 == 0:
            print(f"Step {i}/{num_steps}")
    
    # Decode latents to image
    with torch.no_grad():
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).round().astype("uint8")
    
    return image


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
    model_name = 'vgg'
    seed = 29
    bs = 512 
    n_threads = 1
    
    cvs_path = Path.cwd()/f'../data/{model_name}/corevectors'
    cvs_name = 'corevectors'

    drill_path = Path.cwd()/f'../data/{model_name}/drillers'
    drill_name = 'classifier'
    
    verbose = True 
    
    # Peepholelib
    target_layers = [
            'classifier.0',
            #'classifier.3',
            # 'classifier.6',
            #'features.28',
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

    #--------------------------------
    # Visual Encoder
    #--------------------------------

    model, preprocess = clip.load("ViT-L/14", device=device)

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float32
    ).to(device)

    #--------------------------------
    # CoreVectors 
    #--------------------------------
    
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )
    
    classifier_cv_dim = 100
    features28_cv_dim = 100
    n_cluster = 1000

    cv_parsers = {
            # 'features.24': partial(
            #     trim_kernel_corevectors,
            #     module = 'features.24',
            #     cv_dim = features24_cv_dim
            #     ),
            # 'features.26': partial(
            #     trim_channelwise_corevectors,
            #     module = 'features.26',
            #     cv_dim = features26_cv_dim
            #     ),
            # 'features.28': partial(
            #     trim_corevectors,
            #     module = 'features.28',
            #     cv_dim = features28_cv_dim
            #     ),
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
            #'features.28': features28_cv_dim,
            'classifier.0': classifier_cv_dim,
            #'classifier.3': classifier_cv_dim,
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
            print(driller._clas_path)
            driller.load()
        layer = 'classifier.0'

        n_samples = len(cv._corevds['train'][layer])                   

        probs = torch.empty(n_samples, n_cluster, dtype=torch.float32)

        cv_dl = DataLoader(cv._corevds['train'][layer][...,:classifier_cv_dim], batch_size=bs, num_workers = n_threads)
        
        start = 0
        
        for data in cv_dl:
            bs = data.shape[0]
            probs[start:start+bs] = drillers[layer]._classifier.predict_proba(data)
            start += bs

        conf, clusters = torch.max(probs, dim=1)
        
        for cluster in range(6,7): # 

            idx = torch.argwhere((clusters==cluster)).squeeze()
            images = cv._dss['train']['image'][idx]

            with torch.no_grad():
                
                image_features = model.encode_image(images.to(device))

                mean_image = image_features.mean(dim=0, keepdim=True)           
                image_embeddings = mean_image / mean_image.norm(dim=-1, keepdim=True) 

            batch_size = 1
            seq_length = 77  # Standard for SD
            
            # Option A: Repeat single image embedding across sequence
            cross_attention_embeddings = image_embeddings.unsqueeze(1).repeat(1, seq_length, 1)
            
            # Option B: Use image embedding as first token, pad rest with zeros
            # cross_attention_embeddings = torch.zeros(batch_size, seq_length, image_embeddings.shape[-1], device=device)
            # cross_attention_embeddings[:, 0] = image_embeddings
            
            cross_attention_embeddings = cross_attention_embeddings.to(device=pipe.device, dtype=pipe.unet.dtype)

            image = generate_with_image_steering(pipe=pipe, image_cross_attn_embeds=cross_attention_embeddings, guidance_scale=1)
            plt.imshow(image)
            plt.savefig(drillers[layer]._clas_path/f'diffusion_image.{cluster}.png', dpi=200, bbox_inches='tight')
            quit()
        

        

            
    
