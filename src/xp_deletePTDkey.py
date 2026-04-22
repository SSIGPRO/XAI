import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home() / 'repos/peepholelib').as_posix())

import torch
from tensordict import PersistentTensorDict
from cuda_selector import auto_cuda


class PTDReluCleaner:
    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)

    def preview(self) -> None:
        td = PersistentTensorDict.from_h5(self.file_path, mode='r')
        try:
            keys_to_delete = [key for key in td.keys() if 'relu' in str(key).lower()]
            print(f'Found {len(keys_to_delete)} keys containing "relu":')
            for key in keys_to_delete:
                print(f'  {key}')
        finally:
            if hasattr(td, 'close'):
                td.close()

    def delete(self) -> None:
        td = PersistentTensorDict.from_h5(self.file_path, mode='r+')
        try:
            keys_to_delete = [key for key in td.keys() if 'relu' in str(key).lower()]
            print(f'Found {len(keys_to_delete)} keys containing "relu":')

            for key in keys_to_delete:
                print(f'Deleting {key}')
                td.del_(key)

            if hasattr(td, 'flush'):
                td.flush()

            print('Deletion complete.')
        finally:
            if hasattr(td, 'close'):
                td.close()


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    ptd_path = Path(
        '/srv/newpenny/XAI/generated_data/Kami_attacks/CIFAR100_WRN28-standard/peepholes_/peepholes_cdc.svd.CIFAR100-test-APGD-ce-WRN28-standard'
    )

    cleaner = PTDReluCleaner(ptd_path)

    # cleaner.preview()
    cleaner.delete()