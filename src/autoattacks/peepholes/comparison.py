import h5py
import numpy as np
from pathlib import Path

BASE = Path.home() / 'newpenny/XAI/generated_data/attacks/CIFAR100_WRN28-standard/peepholes'

F1 = BASE / 'peepholes_cdc.svd.CIFAR100-val-WRN28-standard_'
F2 = BASE / 'peepholes_cdc.svd.CIFAR100-val-WRN28-standard'

ATOL = 1e-4  # tolerance for np.allclose


def compare(path1: Path, path2: Path, atol: float = ATOL):
    with h5py.File(path1, 'r') as f1, h5py.File(path2, 'r') as f2:
        keys1 = set(f1.keys())
        keys2 = set(f2.keys())

        assert keys1 == keys2, (
            f"Key mismatch:\n  only in f1: {keys1 - keys2}\n  only in f2: {keys2 - keys1}"
        )

        print(f"{'Layer':<40} {'Max abs diff':>14} {'Mean abs diff':>14} {'Close':>7}")
        print('-' * 80)

        max_diffs, mean_diffs = [], []
        all_close = True

        for key in sorted(keys1):
            a = f1[key][:]
            b = f2[key][:]

            assert a.shape == b.shape, f"{key}: shape mismatch {a.shape} vs {b.shape}"
            assert a.dtype == b.dtype, f"{key}: dtype mismatch {a.dtype} vs {b.dtype}"

            diff = np.abs(a - b)
            max_d = float(diff.max())
            mean_d = float(diff.mean())
            close = bool(np.allclose(a, b, atol=atol))

            max_diffs.append(max_d)
            mean_diffs.append(mean_d)
            all_close = all_close and close

            print(f"{key:<40} {max_d:>14.6f} {mean_d:>14.6f} {str(close):>7}")

        print('-' * 80)
        print(f"{'OVERALL':<40} {max(max_diffs):>14.6f} {np.mean(mean_diffs):>14.6f} {str(all_close):>7}")
        print()
        if all_close:
            print(f"PASS: files are numerically equivalent (atol={atol})")
        else:
            print(f"FAIL: files differ beyond tolerance (atol={atol})")

    return all_close


if __name__ == '__main__':
    compare(F1, F2)
