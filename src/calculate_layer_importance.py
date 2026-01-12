from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch


def _to_1d_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _roc_auc_binary(scores: np.ndarray, labels01: np.ndarray) -> float:
    """
    AUROC where labels01 is {0,1} and 1 is the "positive" class.
    Uses rank statistic; no sklearn dependency.
    """
    y = labels01.astype(np.int64)
    s = scores.astype(np.float64)

    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)

    # average ranks for exact ties
    sorted_s = s[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    sum_ranks_pos = ranks[y == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _fpr95_from_okko(scores: np.ndarray, results_bool: np.ndarray) -> float:
    """
    Matches your plot_confidence logic:
      - positives = OK (results == True)
      - negatives = KO (results == False)
      - threshold = value such that 95% of positives are above it (descending sort)
      - fpr95 = fraction of negatives >= threshold
    """
    s_oks = scores[results_bool]
    s_kos = scores[~results_bool]
    if s_oks.size == 0 or s_kos.size == 0:
        return float("nan")

    sorted_pos = np.sort(s_oks)[::-1]  # descending
    tpr95_index = int(np.ceil(0.95 * sorted_pos.size)) - 1
    tpr95_index = max(0, min(tpr95_index, sorted_pos.size - 1))
    thr = sorted_pos[tpr95_index]
    return float(np.mean(s_kos >= thr))


def compute_metrics_per_loader(*, datasets, scores: Dict[str, Dict[str, Any]],
    loaders: List[str], score_name: str, ) -> Dict[str, Dict[str, float]]:
    """
    Computes (AUC, FPR95) per loader using datasets._dss[loader]['result'].
    Returns:
      metrics[loader] = {"auc": ..., "fpr95": ...}
    """
    out = {}
    for loader in loaders:
        if loader not in scores or score_name not in scores[loader]:
            raise KeyError(f"Missing scores['{loader}']['{score_name}'].")

        results = datasets._dss[loader]["result"]
        # results might be torch/bool; ensure numpy bool
        if isinstance(results, torch.Tensor):
            results_bool = results.detach().cpu().numpy().astype(bool).reshape(-1)
        else:
            results_bool = np.asarray(results, dtype=bool).reshape(-1)

        s = _to_1d_numpy(scores[loader][score_name])

        # AUC uses labels where 1=OK, 0=KO 
        labels01 = results_bool.astype(np.int64)
        auc = _roc_auc_binary(s, labels01)

        fpr95 = _fpr95_from_okko(s, results_bool)

        out[loader] = {"auc": auc, "fpr95": fpr95}
    return out


def layer_importance_lolo_deltas_per_loader_okko( *, score_fn,  datasets,
    peepholes, target_modules: List[str],
    loaders: List[str], score_name: str,
    append_scores: Optional[dict] = None, verbose: bool = True,
    # extra kwargs for score_fn (passed-through)
    **score_fn_kwargs, ) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    LOLO deltas per layer and per loader, using OK/KO labels from datasets.

    Returns:
      deltas[layer][loader] = {"delta_auc": ..., "delta_fpr95": ...}
    """
    # baseline
    base_out = score_fn(
        datasets=datasets,
        peepholes=peepholes,
        target_modules=target_modules,
        score_name=score_name,
        append_scores=append_scores,
        verbose=verbose,
        **score_fn_kwargs,
    )
    # proto_score returns (ret, proto); DMD_score returns ret
    base_scores = base_out[0] if isinstance(base_out, tuple) else base_out

    base_metrics = compute_metrics_per_loader(
        datasets=datasets, scores=base_scores, loaders=loaders, score_name=score_name
    )

    deltas = {}

    for layer in target_modules:
        reduced = [m for m in target_modules if m != layer]
        if len(reduced) == 0:
            deltas[layer] = {ld: {"delta_auc": float("nan"), "delta_fpr95": float("nan")} for ld in loaders}
            continue

        loo_out = score_fn(
            datasets=datasets,
            peepholes=peepholes,
            target_modules=reduced,
            score_name=score_name,
            append_scores=append_scores,
            verbose=verbose,
            **score_fn_kwargs,
        )
        loo_scores = loo_out[0] if isinstance(loo_out, tuple) else loo_out

        loo_metrics = compute_metrics_per_loader(
            datasets=datasets, scores=loo_scores, loaders=loaders, score_name=score_name
        )

        layer_deltas = {}
        for ld in loaders:
            layer_deltas[ld] = {
                "delta_auc": base_metrics[ld]["auc"] - loo_metrics[ld]["auc"],
                "delta_fpr95": loo_metrics[ld]["fpr95"] - base_metrics[ld]["fpr95"],
            }

        deltas[layer] = layer_deltas

        if verbose:
            mean_dauc = float(np.mean([v["delta_auc"] for v in layer_deltas.values()]))
            mean_dfpr = float(np.mean([v["delta_fpr95"] for v in layer_deltas.values()]))
            print(f"[LOLO] remove={layer:>20s} | mean ΔAUC={mean_dauc:+.4f} | mean ΔFPR95={mean_dfpr:+.4f}")

    return deltas

def topk_layers_per_loader(deltas: Dict[str, Dict[str, Dict[str, float]]], k: int,
    *, mode: str = "auc", require_positive: bool = False,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Print and return the top-k layers per loader, sorted by the chosen delta.

    - mode="auc"   -> rank by ΔAUC (descending)
    - mode="fpr95" -> rank by ΔFPR95 (descending)
    - mode="joint" -> rank by (ΔAUC + ΔFPR95) (descending)

    If require_positive=True, keeps only layers with score > 0 under the chosen mode.
    """
    if k <= 0:
        raise ValueError("k must be positive.")
    if mode not in {"auc", "fpr95", "joint"}:
        raise ValueError("mode must be 'auc', 'fpr95', or 'joint'.")

    loaders = set()
    for _, per_loader in deltas.items():
        loaders.update(per_loader.keys())
    loaders = sorted(loaders)

    print("\n" + "=" * 80)
    print(f"Top-{k} layers per loader (mode = {mode}, require_positive = {require_positive})")
    print("=" * 80)

    out: Dict[str, List[Tuple[str, float]]] = {}

    for loader in loaders:
        layers = []
        dauc = []
        dfpr = []

        for layer, per_loader in deltas.items():
            if loader not in per_loader:
                continue
            layers.append(layer)
            dauc.append(float(per_loader[loader].get("delta_auc", np.nan)))
            dfpr.append(float(per_loader[loader].get("delta_fpr95", np.nan)))

        if not layers:
            print(f"\n[Loader: {loader}] — no data")
            out[loader] = []
            continue

        dauc = np.asarray(dauc, dtype=np.float64)
        dfpr = np.asarray(dfpr, dtype=np.float64)

        if mode == "auc":
            scores = dauc
            label = "ΔAUC"
        elif mode == "fpr95":
            scores = dfpr
            label = "ΔFPR95"
        else:
            scores = dauc + dfpr
            label = "ΔAUC + ΔFPR95"

        mask = np.isfinite(scores)
        if require_positive:
            mask &= (scores <= 0)

        idx = np.where(mask)[0]
        if idx.size == 0:
            print(f"\n[Loader: {loader}] — no layers after filtering")
            out[loader] = []
            continue

        idx_sorted = idx[np.argsort(scores[idx])[::-1]]
        idx_sorted = idx_sorted[: min(k, idx_sorted.size)]

        print(f"\n[Loader: {loader}]")
        print("-" * 80)

        ranked: List[Tuple[str, float]] = []
        for rank, i in enumerate(idx_sorted, start=1):
            print(
                f"{rank:>2d}. {layers[i]:<25s} | "
                f"{label} = {scores[i]:+.5f} | "
                f"ΔAUC = {dauc[i]:+.5f} | ΔFPR95 = {dfpr[i]:+.5f}"
            )
            ranked.append((layers[i], float(scores[i])))

        out[loader] = ranked

    print("=" * 80 + "\n")
    return out