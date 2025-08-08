import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial
from matplotlib import pyplot as plt

# torch stuff
import torch
from cuda_selector import auto_cuda
from nltk.corpus import wordnet as wn
import json

def is_descendant(syn, ancestor):
    return any(ancestor in path for path in syn.hypernym_paths())


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    with open(Path.cwd()/"../data/ImageNet/imagenet_class_index.json") as f:
        imagenet_index = json.load(f)

    wnid_to_syn = {}
    for _, (wnid, _) in imagenet_index.items():
        pos = wnid[0]             # 'n' for noun
        offset = int(wnid[1:])    # e.g. 'n02481823' → 2481823
        try:
            syn = wn.synset_from_pos_and_offset(pos, offset)
            wnid_to_syn[wnid] = syn
        except Exception:
            # skip any WNIDs not in your local WordNet install
            continue

    root = wn.synset('reptile.n.01')

    filtered = {syn for syn in wnid_to_syn.values() if is_descendant(syn, root)}

    print(filtered)