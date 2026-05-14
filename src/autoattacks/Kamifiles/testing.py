import sys
from pathlib import Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
from peepholelib.utils.cdf_scores import build_typicality_matrix
from tensordict import PersistentTensorDict
import torch

ds_base = "/srv/newpenny/XAI/generated_data/Kami_attacks/datasets/CIFAR100"

train_td = PersistentTensorDict.from_h5(f"{ds_base}/dss.CIFAR100-train.WRN28-standard")
train_labels_td = PersistentTensorDict.from_h5(f"{ds_base}/dss.CIFAR100-train")

for i in range(5):
    output     = train_td['output'][i]
    logit_conf = output.max().item()
    soft_conf  = torch.softmax(output, dim=0).max().item()
    result     = train_td['result'][i].item()
    
    print(f"Sample {i} | logit max: {logit_conf:.3f} | softmax conf: {soft_conf:.3f} | result: {result}")