import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

import nltk
from nltk.corpus import wordnet as wn
import json

import argparse

from peepholelib.wordNet_utils.wordnet import *

parser = argparse.ArgumentParser()
parser.add_argument("--concept", required=True, help="concept to find")
parser.add_argument("--prefilter_prefix", required=True, help="prefilter prefix to choos e the synset")
args = parser.parse_args()

_concept = args.concept
prefilter_prefix = args.prefilter_prefix

if __name__ == "__main__":

    model_name = 'vgg'

    seed_synsets = get_synsets_from_concept(_concept, lang='eng')
    print(f"Seed synsets for '{_concept}': {[s.name() for s in seed_synsets]}")

    seed_synsets = choose_seed_synsets_cli(_concept, lang='eng', prefilter_prefix=prefilter_prefix)

    print(f"Seed synsets for '{_concept}': {[s.name() for s in seed_synsets]}")

    if not seed_synsets:
        raise RuntimeError(f"No synset found '{_concept}'")

    subtree = hyponym_closure_synsets(seed_synsets)
    subtree_wnids = {synset_to_wnid(s) for s in subtree} | {synset_to_wnid(s) for s in seed_synsets}

    for wnid in sorted(subtree_wnids): 
        syn = wnid_to_synset(wnid)
        print(f"{wnid} -> {syn.name()} : {syn.definition()}")
    
    with open(Path.cwd()/f'../../data/{model_name}/imagenet_class_synset.json', "r", encoding="utf-8") as f:
        wnid_to_idx_label = json.load(f) 

    target = {}
    for wnid in subtree_wnids:
        if wnid in wnid_to_idx_label:
            idx, label = wnid_to_idx_label[wnid]
            target[idx] = (wnid, label)

    print(target.keys())