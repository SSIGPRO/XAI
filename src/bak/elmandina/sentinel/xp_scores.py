import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/FIORIRE/src').as_posix())

# XPs
from xp_corruption import corruptions
from xp_sentinel import sentinel_model

input_key = 'data'
emb_size = 'large'
num_sensors = 16
seq_len = 16
kernel = [3, 3]
lay3 = False   

seed = 42
bs = 2**17
verbose = True


tests = {}
for ci, corr in corruptions.items():
    #print(f'ci{ci}\ncorr{corr}')
    #quit()
    tests['all-'+ci] = {
        'loaders': ['train', 'val', 'test', f'val-c-all-{ci}'],# f'train-c-all-{ci}', f'test-c-all-{ci}'],
        'driller_fit_key': f'val-c-all-{ci}', 
        'empp_fit_key': f'val-c-all-{ci}', 
        'label_key': 'corruption',
        'n_classes': 5,
        'class_names': list(corruptions.keys())#corr.keys()but just corr since only val is corrupted
        }
