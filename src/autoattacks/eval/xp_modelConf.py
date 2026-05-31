import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/autoattacks').as_posix())

from configs.eval.model_eval import *

import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib.lines import Line2D
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

# from peepholelib.plots.confidence import risk_coverage_curve

def risk_coverage_curve_from_score(probs, labels, score):
    """Returns (coverage, risk) arrays sorted by descending custom score."""
    correct = (probs.argmax(dim=1) == labels).float()

    order = score.argsort(descending=True)
    correct = correct[order]

    n = len(correct)
    coverage = torch.arange(1, n + 1).float() / n
    risk = 1 - correct.cumsum(0) / torch.arange(1, n + 1).float()
    return coverage, risk

if __name__ == "__main__":
    aurc_df = pd.DataFrame()
    margin_aurc_df = pd.DataFrame()
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    aurc_ax, margin_aurc_ax = axs
    cs = {
        'Standard': 'xkcd:cobalt', 
        }#,'Robust': 'xkcd:bluish green' 'xkcd:light orange', 'xkcd:dark hot pink', 'xkcd:purplish', 'xkcd:slate gray', 'xkcd:cinnamon']

    sm = torch.nn.Softmax(dim=1)

    _inference_names = {
            k: [f'{args.model}-{version}' for version in ['standard']] for k in loaders #, 'robust'
            }

    with dataset as ds:

        ds.load_only(
            loaders = [f'{dataset_name}-test'],
            transforms = transforms,
            inference_names = _inference_names,
            verbose = verbose
        )
    
        lst = ds._dss[f'{dataset_name}-test-{args.model}-standard'][:]['label']
        ost = sm(ds._dss[f'{dataset_name}-test-{args.model}-standard'][:]['output'])

        # lrb = ds._dss[f'{dataset_name}-test-{args.model}-robust']['label']
        # orb = sm(ds._dss[f'{dataset_name}-test-{args.model}-robust']['output'])
        
        # cov_st, risk_st = risk_coverage_curve(ost, lst)
        # cov_rb, risk_rb = risk_coverage_curve(orb, lrb)
        top2_st = torch.topk(ost, k=2, dim=1).values
        # top2_rb = torch.topk(orb, k=2, dim=1).values
        # margin_st = top2_st[:, 0] - top2_st[:, 1]
        # margin_rb = top2_rb[:, 0] - top2_rb[:, 1]
        # cov_margin_st, risk_margin_st = risk_coverage_curve_from_score(ost, lst, margin_st)
        # cov_margin_rb, risk_margin_rb = risk_coverage_curve_from_score(orb, lrb, margin_rb)

        # aurc_df = aurc_df._append(
        #             pd.DataFrame({
        #                 'x': cov_st,
        #                 'y': risk_st,
        #                 'score name': ['Standard' for i in range(len(risk_st))] 
        #                         }),
        #             ignore_index = True,
        #             )
        
        # aurc_df = aurc_df._append(
        #             pd.DataFrame({
        #                 'x': cov_rb,
        #                 'y': risk_rb,
        #                 'score name': ['Robust' for i in range(len(risk_rb))] 
        #                         }),
        #             ignore_index = True,
        #             )

        # margin_aurc_df = margin_aurc_df._append(
        #             pd.DataFrame({
        #                 'x': cov_margin_st,
        #                 'y': risk_margin_st,
        #                 'score name': ['Standard' for i in range(len(risk_margin_st))]
        #                         }),
        #             ignore_index = True,
        #             )

        # margin_aurc_df = margin_aurc_df._append(
        #             pd.DataFrame({
        #                 'x': cov_margin_rb,
        #                 'y': risk_margin_rb,
        #                 'score name': ['Robust' for i in range(len(risk_margin_rb))]
        #                         }),
        #             ignore_index = True,
        #             )
        
        # #--------------------
        # # Plotting
        # #--------------------

        # sb.lineplot(
        #     data=aurc_df,
        #     ax=aurc_ax,
        #     x='x',
        #     y='y',
        #     hue='score name',
        #     hue_order=list(cs.keys()),
        #     palette=cs,
        #     errorbar=None,
        #     alpha=0.75,
        #     linewidth=0.5,
        #     legend=False,
        # )

        # sb.lineplot(
        #     data=margin_aurc_df,
        #     ax=margin_aurc_ax,
        #     x='x',
        #     y='y',
        #     hue='score name',
        #     hue_order=list(cs.keys()),
        #     palette=cs,
        #     errorbar=None,
        #     alpha=0.75,
        #     linewidth=0.5,
        #     legend=False,
        # )

        # aurc_ax.set_xlabel('Coverage')
        # aurc_ax.set_ylabel('Risk')
        # aurc_ax.grid(True)

        # margin_aurc_ax.set_xlabel('Coverage')
        # margin_aurc_ax.set_ylabel('Risk')
        # margin_aurc_ax.grid(True)

        # score_names = list(cs.keys())
        # lw = 2.0
        # color_handles = [
        #     Line2D([0], [0], color=cs[score_name], lw=lw)
        #     for score_name in score_names
        # ]
        # color_legend = aurc_ax.legend(
        #     color_handles,
        #     score_names,
        #     title='Score type',
        #     loc='upper left',
        #     frameon=True,
        # )
        # aurc_ax.add_artist(color_legend)

        # aurc_ax.set_title('AURC - Model Confidence')
        # margin_aurc_ax.set_title('AURC - Top-1 minus Top-2 Score')

        # plt.savefig('AURC.png', dpi=300, bbox_inches='tight')
        # plt.close()

        # ----------------------------------------------------------------
        # KDE: confidence distribution — correct vs. incorrect
        # ----------------------------------------------------------------
        conf_st = ost.max(dim=1).values
        # conf_rb = orb.max(dim=1).values
        correct_st = (ost.argmax(dim=1) == lst)
        # correct_rb = (orb.argmax(dim=1) == lrb)

        c_correct   = 'xkcd:cobalt'
        c_incorrect = 'xkcd:dark hot pink'

        fig2, (kde_ax_st, kde_ax_rb) = plt.subplots(1, 2, figsize=(10, 5))

        for ax, conf, correct, title in [
            (kde_ax_st, conf_st, correct_st, 'Standard'),
            # (kde_ax_rb, conf_rb, correct_rb, 'Robust'),
        ]:
            y_true = correct.numpy().astype(int)
            y_score = conf.numpy()

            auc = roc_auc_score(y_true, y_score)
            fprs, tprs, _ = roc_curve(y_true, y_score)
            # FPR at the first threshold where TPR >= 95%
            idx = next((i for i, t in enumerate(tprs) if t >= 0.95), -1)
            fpr95 = fprs[idx]

            df_kde = pd.DataFrame({
                'confidence': y_score,
                'classification': ['correct' if c else 'incorrect' for c in correct.tolist()],
            })
            sb.kdeplot(data=df_kde[df_kde['classification'] == 'correct'],   ax=ax,
                       x='confidence', color=c_correct,   common_norm=False,
                       clip=[0., 1.], label='correct')
            sb.kdeplot(data=df_kde[df_kde['classification'] == 'incorrect'], ax=ax,
                       x='confidence', color=c_incorrect, common_norm=False,
                       clip=[0., 1.], label='incorrect')
            ax.set_xlabel('Max softmax confidence')
            ax.set_ylabel('Density')
            ax.set_title(f'{title} — AUC = {auc:.2f} FPR@95TPR = {fpr95:.2f}')
            ax.legend(title='Classification', frameon=True)
            ax.grid(True)

        fig2.suptitle(f'Confidence distribution — {args.model}', fontsize=13)
        plt.savefig('confidence_dist_correct_incorrect.png', dpi=300, bbox_inches='tight')
        plt.close()
