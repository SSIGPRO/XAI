import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial
from matplotlib import pyplot as plt

# Our stuff
from peepholelib.models.model_wrap import ModelWrap 

# torch stuff
import torch
from cuda_selector import auto_cuda

# huggingface stuff
from datasets import load_dataset
from diffusers import StableDiffusionPipeline

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------

    # model parameters
    
    seed = 29
    bs = 512 
    n_threads = 1    
    
    verbose = True 

    # load in streaming mode (no local download needed)
    ds = load_dataset("laion/relaion-coco", split="train", streaming=True)

    for ex in ds:
        caption = ex["caption"]   # text only
        print(caption)
        # you can tokenize or store it as you like
    
    #--------------------------------
    # Model 
    #--------------------------------

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")

    unet = pipe.unet
     
    model = ModelWrap(
            model = unet,
            device = device
            )
    quit()
