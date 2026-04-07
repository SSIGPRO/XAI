import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/autoattacks').as_posix())

from configs.eval.model_eval import *

import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

if __name__ == "__main__":

    inference_names = {
        f'{dataset_name}-test':
            [
                f'{args.model}-{args.version}',
                f'APGD-ce-{args.model}-{args.version}',
                f'APGD-t-{args.model}-{args.version}'
            ]
        }

    loader = [f'{dataset_name}-test']

    sm = torch.nn.Softmax(dim=1)

    atk_names = ['APGD-ce', 'APGD-t']
    atk_loaders = [f'{dataset_name}-test-{atk}-{args.model}-{args.version}' for atk in atk_names]
    std_loader = f'{dataset_name}-test-{args.model}-{args.version}'

    with dataset as ds:
        ds.load_only(
            loaders = loader,
            inference_names = inference_names,
            verbose = verbose
        )

        for atk_loader in atk_loaders:
            print(f'{atk_loader} attack success rate: {(ds._dss[atk_loader]['attack_success'].sum()/ds._dss[std_loader]["result"].sum())*100:.2f}%')

        colors = ['xkcd:cobalt', 'xkcd:bluish green', 'xkcd:light orange']

        std_conf = sm(ds._dss[std_loader]['output']).max(dim=1).values

        df = pd.DataFrame({
            'confidence': std_conf,
            'source': 'original',
        })

        for atk_name, atk_loader in zip(atk_names, atk_loaders):
            mask = ds._dss[atk_loader]['attack_success']
            atk_conf = sm(ds._dss[atk_loader]['output'][mask]).max(dim=1).values
            df = df._append(
                pd.DataFrame({
                    'confidence': atk_conf,
                    'source': atk_name,
                }),
                ignore_index=True,
            )

        sources = ['original'] + atk_names
        cs = {src: colors[i] for i, src in enumerate(sources)}

        fig, axs = plt.subplots(1, len(sources), figsize=(7 * len(sources), 5), sharey=True)

        for ax, src in zip(axs, sources):
            sb.kdeplot(
                data=df[df['source'] == src],
                ax=ax,
                x='confidence',
                color=cs[src],
                common_norm=False,
                clip=[0., 1.],
                alpha=0.75,
            )
            ax.set_xlabel('Max softmax confidence')
            ax.set_ylabel('Density')
            ax.set_title(src)
            ax.grid(True)

        out_path = f'confidence_dist_{args.model}_{args.version}.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved plot to {out_path}')
