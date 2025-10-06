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
from diffusers import StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler

import requests, io
from PIL import Image
import textwrap

from PIL import Image
from torchvision import transforms
from datasets import load_dataset



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-2.0"
DATASET = "laion/relaion-coco"
SPLIT = "train"
N = 10
IMAGE_SIZE = 512  # typical for diffusion models; change if you need 256/768/etc.

def save_image(url, path, timeout=8):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    with Image.open(io.BytesIO(r.content)) as im:
        im.convert("RGB").save(path)(path)

def fetch_image_to_tensor(url: str):
    """Download an image URL to a preprocessed tensor or return None on failure."""
    # A UA helps avoid some 403s
    headers = {"User-Agent": "Mozilla/5.0 (image-loader)"}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    # Basic content check (optional, some servers omit content-type)
    ctype = resp.headers.get("Content-Type", "")
    if ("image" not in ctype) and not url.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
        return None
    with Image.open(io.BytesIO(resp.content)) as im:
        im = im.convert("RGB")
        tensor = preprocess(im)  # shape [3, H, W], float32, [-1,1]
    return tensor

def first_n_tensors_from_laion(n=N, image_size=IMAGE_SIZE, device=DEVICE, want_captions=True):
    ds = load_dataset(DATASET, split=SPLIT, streaming=True)
    tensors = []
    captions = []

    def get_url_and_caption(row):
        url = row.get("url") or row.get("URL") or row.get("image_url")
        cap = row.get("top_caption") or row.get("caption") or row.get("text")
        return url, cap

    for row in ds:
        if len(tensors) >= n:
            break
        url, cap = get_url_and_caption(row)
        if not url:
            continue
        try:
            t = fetch_image_to_tensor(url)
            if t is None:
                continue
            tensors.append(t.to(device))
            if want_captions:
                captions.append(cap)
        except Exception:
            # Skip bad/slow/forbidden URLs and keep going
            continue

    if len(tensors) == 0:
        raise RuntimeError("No images could be loaded from the stream.")

    batch = torch.stack(tensors, dim=0)  # [B, 3, H, W]
    return (batch, captions) if want_captions else batch

def to_numpy_from_minus1_1(t):
    return t.detach().cpu().clamp(-1,1).add(1).div(2).numpy().transpose(1,2,0)

# output from diffusers (already [0,1])
def to_numpy_from_0_1(t):
    return t.detach().cpu().clamp(0,1).numpy().transpose(1,2,0)
    
if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    seed = 29
    bs = 512 
    n_threads = 1    
    
    verbose = True

    #--------------------------------
    # Dataset 
    #--------------------------------

    # Load (streaming)
    preprocess = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=Image.BICUBIC),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),                              # [0,1]
        transforms.Normalize([0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5])               # -> [-1,1]
    ])

    batch, caps = first_n_tensors_from_laion()
    print("Batch shape:", batch.shape)  # -> [10, 3, 512, 512] (if 10 succeeded)
    print("Dtype/range:", batch.dtype, f"[{batch.min().item():.2f}, {batch.max().item():.2f}]")
    print("Example caption:", caps[0] if caps and caps[0] else "(no caption)")

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        variant="fp16" if DEVICE == "cuda" else None,
        use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing()

    img = batch[2:3].clone()      # [-1,1] -> [0,1]
    img_01 = (img + 1) / 2
    img_01 = img_01.to(device=DEVICE, dtype=pipe.unet.dtype)

    prompt = "cinematic photo of a vintage motorcycle parked by the seaside road at golden hour"
    negative_prompt = "blurry, low-res, artifacts"

    out = pipe(
        prompt=prompt,
        image=img_01,
        negative_prompt=negative_prompt,
        strength=0.8,
        guidance_scale=6.5,
        num_inference_steps=30,
        output_type="pt"
    ).images[0]

    fig, axs = plt.subplots(1, 2, figsize=(10,8))
    idx = 2
    axs[0].imshow(to_numpy_from_minus1_1(batch[idx]))
    axs[0].axis("off")
    wrapped_cap = "\n".join(textwrap.wrap(caps[idx], width=40))  # adjust width for your figure
    axs[0].set_title(wrapped_cap, fontsize=12, weight='bold', wrap=True)

    axs[1].imshow(to_numpy_from_0_1(out.squeeze()))
    axs[1].axis("off")
    wrapped_propmt = "\n".join(textwrap.wrap(prompt, width=40))  # adjust width for your figure
    axs[1].set_title(wrapped_propmt, fontsize=12, weight='bold', wrap=True)
    fig.savefig('prova.png')


    