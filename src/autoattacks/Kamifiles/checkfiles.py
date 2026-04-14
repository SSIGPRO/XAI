# import h5py
# from pathlib import Path
# import numpy as np

# file_path = Path('/srv/newpenny/XAI/generated_data/Kami_attacks/datasets/CIFAR100/dss.CIFAR100-test.WRN28-standard')

# with h5py.File(file_path, 'r') as f:
#     result = f['result'][:]

# accuracy = np.mean(result)  # mean of True and False values gives accuracy
# print(f"Accuracy: {accuracy*100:.2f}%")

# import h5py
# from pathlib import Path

# file_path = Path('/srv/newpenny/XAI/generated_data/Kami_attacks/datasets/CIFAR100/dss.CIFAR100-test.APGD-t-WRN28-robust')

# with h5py.File(file_path, 'r') as f:
#     def print_structure(name, obj):
#         if isinstance(obj, h5py.Dataset):
#             print(f"{name} -> Dataset | shape: {obj.shape} | dtype: {obj.dtype}")
#         else:
#             print(f"{name} -> Group")

#     f.visititems(print_structure)

# import h5py
# from pathlib import Path

# file_path = Path('/srv/newpenny/XAI/generated_data/Kami_attacks/datasets/CIFAR100/dss.CIFAR100-test.APGD-t-WRN28-robust')

# with h5py.File(file_path, 'r') as f:
#     for i in range(5):
#         print(f"\n--- Sample {i} ---")
#         print("pred:", f['pred'][i])
#         print("result (correct?):", f['result'][i])
#         print("attack_success:", f['attack_success'][i])
#         print("output (first 5 logits):", f['output'][i][:5])

import h5py
from pathlib import Path
import numpy as np

clean_path = Path('/srv/newpenny/XAI/generated_data/Kami_attacks/datasets/CIFAR100/dss.CIFAR100-test.WRN28-robust')
adv_path   = Path('/srv/newpenny/XAI/generated_data/Kami_attacks/datasets/CIFAR100/dss.CIFAR100-test.APGD-ce-WRN28-robust')

with h5py.File(clean_path, 'r') as f_clean, h5py.File(adv_path, 'r') as f_adv:
    
    clean_correct = f_clean['result'][:]   # True = correct prediction (clean)
    adv_correct   = f_adv['result'][:]     # True = correct prediction (after attack)

    # Only consider samples that were correct BEFORE attack
    mask = clean_correct == True

    total_valid = mask.sum()
    still_correct = adv_correct[mask].sum()

    robust_accuracy = still_correct / total_valid

    print(f"Samples originally correct: {total_valid}")
    print(f"Still correct after attack: {still_correct}")
    print(f"Robust Accuracy: {robust_accuracy * 100:.2f}%")