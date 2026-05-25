"""
Re-saves wrn70_16_cifar100.pt with the ModelWrap wrapper keys stripped,
so it can be loaded directly into a bare WideResNet.

Strips:  normalizer.*
Renames: model.<rest>  ->  <rest>
"""
import torch
from pathlib import Path

model_dir = Path('/srv/newpenny/XAI/models')
src = model_dir / 'wrn70_16_cifar100.pt'
dst = model_dir / 'wrn70_16_cifar100.pt'           # overwrite in-place
bak = model_dir / 'wrn70_16_cifar100.pt.bak'

ckpt = torch.load(src, map_location='cpu')

# Handle both plain state-dict and {'state_dict': ...} formats
if isinstance(ckpt, dict) and 'state_dict' in ckpt:
    sd = ckpt['state_dict']
    wrap = True
else:
    sd = ckpt
    wrap = False

cleaned = {}
for k, v in sd.items():
    if k.startswith('normalizer.'):
        continue
    if k.startswith('model.'):
        cleaned[k[len('model.'):]] = v
    else:
        cleaned[k] = v

# Back up original, then save
import shutil
shutil.copy(src, bak)
print(f"Backup saved to {bak}")

if wrap:
    ckpt['state_dict'] = cleaned
    torch.save(ckpt, dst)
else:
    torch.save(cleaned, dst)

print(f"Cleaned checkpoint saved to {dst}")
print(f"Keys before: {len(sd)}  |  Keys after: {len(cleaned)}")
