import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/autoattacks').as_posix())

from configs.eval.model_eval import *

import torch
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

        from sklearn.metrics import roc_auc_score

        std_output = sm(ds._dss[std_loader]['output'])
        std_conf = std_output.max(dim=1).values  # (N,)

        c_orig = 'xkcd:cobalt'
        c_atks = ['xkcd:light orange', 'xkcd:dark hot pink']

        fig, axs = plt.subplots(1, len(atk_names), figsize=(7 * len(atk_names), 5), sharey=False)
        if len(atk_names) == 1:
            axs = [axs]

        for ax, atk_name, atk_loader, c_atk in zip(axs, atk_names, atk_loaders, c_atks):
            mask = ds._dss[atk_loader]['attack_success'].bool()

            orig_conf = std_conf[mask]                                              # originals that were successfully attacked
            atk_conf  = sm(ds._dss[atk_loader]['output'][mask]).max(dim=1).values  # their attacked versions

            scores = torch.cat([orig_conf, atk_conf])
            labels = torch.cat([torch.ones(len(orig_conf)), torch.zeros(len(atk_conf))])
            auc = roc_auc_score(labels.numpy(), scores.numpy())

            df = pd.DataFrame({
                'confidence': torch.cat([orig_conf, atk_conf]),
                'source':     ['original'] * len(orig_conf) + [atk_name] * len(atk_conf),
            })

            sb.kdeplot(data=df[df['source'] == 'original'], ax=ax, x='confidence',
                       color=c_orig, common_norm=False, clip=[0., 1.], alpha=0.75, label='original')
            sb.kdeplot(data=df[df['source'] == atk_name],  ax=ax, x='confidence',
                       color=c_atk,  common_norm=False, clip=[0., 1.], alpha=0.75, label=atk_name)

            ax.set_xlabel('Max softmax confidence')
            ax.set_ylabel('Density')
            ax.set_title(f'{atk_name}  (AUROC = {auc:.3f})')
            ax.legend(title='Source', loc='upper left', frameon=True)
            ax.grid(True)

        fig.suptitle(f'Confidence distribution — {args.model} {args.version}  |  only successful attacks', fontsize=13)

        out_path = f'confidence_dist_{args.model}_{args.version}.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved plot to {out_path}')

        # ----------------------------------------------------------------
        # Joint plot: disjoint halves of each attack vs their originals,
        # single AUROC pooling both attacks together.
        # ----------------------------------------------------------------

        # Indices of successful attacks for each method
        idx_ce = ds._dss[atk_loaders[0]]['attack_success'].bool().nonzero(as_tuple=True)[0]
        idx_t  = ds._dss[atk_loaders[1]]['attack_success'].bool().nonzero(as_tuple=True)[0]

        n_ce = len(idx_ce) // 2
        idx_ce = idx_ce[:n_ce]

        idx_ce_set = set(idx_ce.tolist())
        idx_t_unique = idx_t[torch.tensor([i.item() not in idx_ce_set for i in idx_t])]

        # Take half of each (random, without replacement)
        n_ce = len(idx_ce) // 2
        n_t  = len(idx_t_unique) // 2
        perm_ce = torch.randperm(len(idx_ce))
        perm_t  = torch.randperm(len(idx_t_unique))[:n_t]
        sampled_ce = idx_ce[perm_ce]
        sampled_t  = idx_t_unique[perm_t]

        # Confidence of the attacked versions
        atk_conf_ce = sm(ds._dss[atk_loaders[0]]['output'][sampled_ce]).max(dim=1).values
        atk_conf_t  = sm(ds._dss[atk_loaders[1]]['output'][sampled_t]).max(dim=1).values
        atk_conf_all = torch.cat([atk_conf_ce, atk_conf_t])

        # Confidence of the corresponding originals
        orig_conf_ce = std_conf[sampled_ce]
        orig_conf_t  = std_conf[sampled_t]
        orig_conf_all = torch.cat([orig_conf_ce, orig_conf_t])

        # Single AUROC: originals=1, attacks=0
        scores_joint = torch.cat([orig_conf_all, atk_conf_all])
        labels_joint = torch.cat([torch.ones(len(orig_conf_all)), torch.zeros(len(atk_conf_all))])
        auc_joint = roc_auc_score(labels_joint.numpy(), scores_joint.numpy())

        df_joint = pd.DataFrame({
            'confidence': scores_joint,
            'source': ['original'] * len(orig_conf_all) + ['attack'] * len(atk_conf_all),
        })

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        sb.kdeplot(data=df_joint[df_joint['source'] == 'original'], ax=ax2, x='confidence',
                   color='xkcd:cobalt', common_norm=False, clip=[0., 1.], alpha=0.75, label='original')
        sb.kdeplot(data=df_joint[df_joint['source'] == 'attack'], ax=ax2, x='confidence',
                   color='xkcd:light orange', common_norm=False, clip=[0., 1.], alpha=0.75, label='attack')

        ax2.set_xlabel('Max softmax confidence')
        ax2.set_ylabel('Density')
        ax2.set_title(
            f'Confidence distribution — {args.model} {args.version}\n'
            f'disjoint halves of APGD-ce & APGD-t  |  AUROC = {auc_joint:.3f}'
        )
        ax2.legend(title='Source', loc='upper left', frameon=True)
        ax2.grid(True)

        out_path2 = f'confidence_dist_joint_{args.model}_{args.version}.png'
        plt.savefig(out_path2, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved joint plot to {out_path2}')
