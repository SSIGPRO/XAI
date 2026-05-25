import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/autoattacks').as_posix())

from configs.eval.model_eval import *

import torch
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

def lp_norm(delta: torch.Tensor, p) -> torch.Tensor:
    """Per-sample Lp norm of a perturbation tensor (N, C, H, W)."""
    flat = delta.flatten(start_dim=1)
    if p == float('inf'):
        return flat.abs().max(dim=1).values
    return flat.norm(p=p, dim=1)


if __name__ == "__main__":

    atk_names   = ['APGD-ce', 'APGD-t']
    atk_loaders = [f'{dataset_name}-test-{atk}-{args.model}-{args.version}' for atk in atk_names]
    std_loader  = f'{dataset_name}-test-{args.model}-{args.version}'

    inference_names = {
        f'{dataset_name}-test': [
            f'{args.model}-{args.version}',
            f'APGD-ce-{args.model}-{args.version}',
            f'APGD-t-{args.model}-{args.version}',
        ]
    }

    c_atks = {'APGD-ce': 'xkcd:light orange', 'APGD-t': 'xkcd:dark hot pink'}

    with dataset as ds:
        ds.load_only(
            loaders          = [f'{dataset_name}-test'],
            inference_names  = inference_names,
            verbose          = verbose,
        )

        orig_images = ds._dss[std_loader]['image']

        rows = []
        for atk_name, atk_loader in zip(atk_names, atk_loaders):
            mask    = ds._dss[atk_loader]['attack_success'].bool()
            delta   = ds._dss[atk_loader]['image'][mask] - orig_images[mask]
            norms   = lp_norm(delta, 2)
            for v in norms.tolist():
                rows.append({'attack': atk_name, 'value': v})

        df = pd.DataFrame(rows)

        fig, ax = plt.subplots(figsize=(7, 5))

        for atk_name in atk_names:
            sb.kdeplot(
                data=df[df['attack'] == atk_name], ax=ax, x='value',
                color=c_atks[atk_name], common_norm=False,
                label=atk_name,
            )

        ax.set_xlabel('L2 perturbation magnitude')
        ax.set_ylabel('Density')
        ax.set_title(f'L2 norm distribution — {args.model} {args.version}  |  successful attacks')
        ax.legend(title='Attack', frameon=True)

        out_path = f'l2_norm_dist_{args.model}_{args.version}.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved plot to {out_path}')
