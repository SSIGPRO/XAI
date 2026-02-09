import numpy as np
import torch
from peepholelib.utils.localization import *
from peepholelib.utils.gini_index import *



def delta_auc_lolo(**kwargs):
    """
    LOLO (leave-one-layer-out) importance using ΔAUC from sparse function.

    ΔAUC(layer) = AUC(all_layers) - AUC(all_layers \ {layer})

    Returns:
        dict:
            {
              "loader": loader,
              "base_auc": float,
              "delta_auc": {layer_name: float, ...}
            }
    """
    phs = kwargs["phs"]
    ds = kwargs["ds"]
    loader = kwargs["loader"]
    target_modules = kwargs["target_modules"] # all layers!
    sparse_method = kwargs.get("sparse_method", "gini") # either "gini" (gini index) or "localization"

    eps = kwargs.get("eps", 1e-12)
    verbose = kwargs.get("verbose", True)

    # baseline (all layers)
    if sparse_method == "gini":
        base_out = gini_from_peepholes(phs=phs, ds=ds, ds_key=loader, target_modules=target_modules)

    else:
        base_out = localization_from_peepholes(phs=phs, ds=ds, ds_key=loader, target_modules=target_modules)
    base_auc = float(base_out["auc"])

    deltas = {}

    for layer in target_modules:
        reduced = [m for m in target_modules if m != layer]
        if len(reduced) == 0:
            deltas[layer] = float("nan")
            continue
        if sparse_method == "gini":
            loo_out = gini_from_peepholes(phs=phs, ds=ds, ds_key=loader, target_modules=reduced)

        else:
            loo_out = localization_from_peepholes(phs=phs, ds=ds, ds_key=loader, target_modules=reduced)

        loo_auc = float(loo_out["auc"])

        delta_auc = base_auc - loo_auc
        deltas[layer] = delta_auc

        if verbose:
            print(f"loader={loader} remove={layer:>20s} | ΔAUC={delta_auc:+.6f}")

    return {"loader": loader, "base_auc": base_auc, "delta_auc": deltas}


def topk_layers_by_delta_auc(**kwargs):
    """
    Top-k layers by ΔAUC (descending).

    Accepts either:
      - deltas: dict[layer_name -> float]
      - deltas: output dict from delta_auc_lolo (with key "delta_auc")
    """
    deltas = kwargs["deltas"]
    k = kwargs.get("k", 10)
    negatives = kwargs.get("negatives", False)

    # If caller passed the full delta_auc_lolo output, extract the mapping
    if isinstance(deltas, dict) and "delta_auc" in deltas and isinstance(deltas["delta_auc"], dict):
        deltas = deltas["delta_auc"]

    layers = list(deltas.keys())
    vals = np.asarray([deltas[l] for l in layers], dtype=np.float64)

    mask = np.isfinite(vals)
    if negatives:
        mask &= (vals < 0)

    idx = np.where(mask)[0]
    if idx.size == 0:
        return []

    idx_sorted = idx[np.argsort(vals[idx])[::-1]]
    idx_sorted = idx_sorted[: min(k, idx_sorted.size)]

    ranked = [(layers[i], float(vals[i])) for i in idx_sorted]

    print("\n" + "=" * 80)
    print(f"Top-{k} layers by ΔAUC (negatives={negatives})")
    print("=" * 80)
    for r, (layer, score) in enumerate(ranked, 1):
        print(f"{r:>2d}. {layer:<25s} | ΔAUC = {score:+.6f}")
    print("=" * 80 + "\n")

    return ranked

