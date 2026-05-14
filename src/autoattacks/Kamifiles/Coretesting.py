import sys
from pathlib import Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
from peepholelib.utils.cdf_scores import build_typicality_matrix
from tensordict import PersistentTensorDict
import torch

ds_base = "/srv/newpenny/XAI/generated_data/Kami_attacks/datasets/CIFAR100"
train_td = PersistentTensorDict.from_h5(f"{ds_base}/dss.CIFAR100-train.WRN28-standard")
print(len(train_td['pred']))