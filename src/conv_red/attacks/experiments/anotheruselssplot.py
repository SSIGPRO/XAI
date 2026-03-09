from pathlib import Path
import matplotlib.pyplot as plt
import torch
from tensordict import PersistentTensorDict

base = Path.cwd()/'../../../data'/'vgg'/'datasets'

cw  = PersistentTensorDict.from_h5(base/'dss.CW-CIFAR100-val', mode='r')
APGD = PersistentTensorDict.from_h5(base/'dss.APGD-CIFAR100-val', mode='r')

def success_rate(td):
    s = td['attack_success'][:]
    return s.float().mean().item()

vals = {"CW": success_rate(cw), "APGD": success_rate(APGD)}

plt.figure()
plt.bar(list(vals.keys()), list(vals.values()))
plt.ylim(0, 1)
plt.ylabel("Attack success rate")
plt.title("Targeted attack success rate")

out = Path("plots")
out.mkdir(exist_ok=True)
plt.savefig(out/"attack_success_val.png", dpi=200, bbox_inches="tight")
plt.close()

print("saved", out/"attack_success_val.png")