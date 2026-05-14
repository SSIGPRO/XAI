import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tensordict import PersistentTensorDict
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score


# =============================================================================
# Configuration
# =============================================================================

DATASET = "CIFAR100"
SPLIT = "test"

# Choose one:
MODEL = "WRN28-standard"
# MODEL = "WRN28-robust"

ATTACKS = ["APGD-ce", "APGD-t"]

# Manual per-layer FPR budgets.
# These are individual clean FPRs for each detector.
# The final combined FPR is measured after OR-combining them.
DETECTORS = [
    {
        "name": "early_block",
        "layer": "block1.layer.2.conv2",
        "tail": 100,
        "direction": "high",
        "fpr_budget": 0.02,
    },
    {
        "name": "fc",
        "layer": "fc",
        "tail": 100,
        "direction": "high",
        "fpr_budget": 0.0001,
    },
]

DS_BASE = Path(f"/srv/newpenny/XAI/generated_data/Kami_attacks/datasets/{DATASET}")
CV_BASE = Path(f"/srv/newpenny/XAI/generated_data/Kami_attacks/{DATASET}_{MODEL}/corevectors")

PLOT_DIR = Path(
    f"/srv/newpenny/XAI/generated_data/Kami_attacks/{DATASET}_{MODEL}/manual_two_layer_detector"
)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Path helpers
# =============================================================================

def dataset_path(attack=None):
    """
    Dataset filenames:

    Clean:
        dss.CIFAR100-test.WRN28-standard

    Attack:
        dss.CIFAR100-test.APGD-ce-WRN28-standard
        dss.CIFAR100-test.APGD-t-WRN28-standard
    """
    if attack is None:
        name = f"dss.{DATASET}-{SPLIT}.{MODEL}"
    else:
        name = f"dss.{DATASET}-{SPLIT}.{attack}-{MODEL}"

    return DS_BASE / name


def corevector_path(attack=None):
    """
    Corevector filenames:

    Clean:
        corevectors.svd.CIFAR100-test-WRN28-standard

    Attack:
        corevectors.svd.CIFAR100-test-APGD-ce-WRN28-standard
        corevectors.svd.CIFAR100-test-APGD-t-WRN28-standard
    """
    if attack is None:
        name = f"corevectors.svd.{DATASET}-{SPLIT}-{MODEL}"
    else:
        name = f"corevectors.svd.{DATASET}-{SPLIT}-{attack}-{MODEL}"

    return CV_BASE / name


def require_exists(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file:\n{path}")
    return path


def load_td(path):
    require_exists(path)
    return PersistentTensorDict.from_h5(str(path))


# =============================================================================
# Detector helpers
# =============================================================================

def compute_tail_energy(cv, layer, mask, tail):
    """
    Tail energy = sum of squared values in the final `tail` SVD/corevector
    coordinates.
    """
    if layer not in cv.keys():
        raise KeyError(
            f"Layer '{layer}' not found.\n"
            f"Available layers:\n{list(cv.keys())}"
        )

    z = cv[layer][:].float()[mask]
    energy = (z[:, -tail:] ** 2).sum(dim=1)
    return energy.cpu().numpy()


def calibrate_threshold(clean_scores, direction, fpr):
    """
    Calibrate one detector threshold using clean scores only.

    direction = "high":
        reject scores above upper threshold.

    direction = "low":
        reject scores below lower threshold.

    direction = "two-sided":
        reject scores below lower or above upper threshold.
    """
    if direction == "high":
        return {
            "lower": None,
            "upper": np.quantile(clean_scores, 1.0 - fpr),
        }

    if direction == "low":
        return {
            "lower": np.quantile(clean_scores, fpr),
            "upper": None,
        }

    if direction == "two-sided":
        return {
            "lower": np.quantile(clean_scores, fpr / 2.0),
            "upper": np.quantile(clean_scores, 1.0 - fpr / 2.0),
        }

    raise ValueError(f"Unknown direction: {direction}")


def apply_threshold(scores, threshold, direction):
    """
    Apply one detector threshold.
    """
    if direction == "high":
        return scores > threshold["upper"]

    if direction == "low":
        return scores < threshold["lower"]

    if direction == "two-sided":
        return (scores < threshold["lower"]) | (scores > threshold["upper"])

    raise ValueError(f"Unknown direction: {direction}")


def score_auc(clean_scores, adv_scores, direction):
    """
    AUC for one detector score.

    For high-tail detectors:
        larger score = more adversarial.

    For low-tail detectors:
        smaller score = more adversarial, so we flip sign.

    For two-sided detectors:
        farther from clean median = more adversarial.
    """
    labels = np.concatenate([
        np.zeros(len(clean_scores)),
        np.ones(len(adv_scores)),
    ])

    if direction == "high":
        scores = np.concatenate([clean_scores, adv_scores])

    elif direction == "low":
        scores = -np.concatenate([clean_scores, adv_scores])

    elif direction == "two-sided":
        center = np.median(clean_scores)
        scores = np.abs(np.concatenate([clean_scores, adv_scores]) - center)

    else:
        raise ValueError(f"Unknown direction: {direction}")

    return roc_auc_score(labels, scores)


def plot_detector_layer(
    clean_scores,
    attack_scores,
    threshold,
    detector,
    combined_fpr,
):
    """
    Save one KDE plot for one detector/layer.
    """
    name = detector["name"]
    layer = detector["layer"]
    direction = detector["direction"]
    fpr_budget = detector["fpr_budget"]

    all_scores = [clean_scores] + [attack_scores[a] for a in ATTACKS]

    x_min = min(s.min() for s in all_scores)
    x_max = max(s.max() for s in all_scores)
    x_grid = np.linspace(x_min, x_max, 600)

    plt.figure(figsize=(7, 5))

    curves = [
        (clean_scores, "clean", "steelblue"),
        (attack_scores["APGD-ce"], "APGD-CE", "tomato"),
        (attack_scores["APGD-t"], "APGD-T", "seagreen"),
    ]

    for scores, label, color in curves:
        if np.std(scores) > 1e-8:
            kde = gaussian_kde(scores)
            y = kde(x_grid)
            plt.plot(x_grid, y, label=label, color=color)
            plt.fill_between(x_grid, y, alpha=0.15, color=color)

    if direction == "high":
        upper = threshold["upper"]

        plt.axvline(
            upper,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"threshold = {upper:.2f}",
        )

        plt.axvspan(
            upper,
            x_max,
            color="gray",
            alpha=0.15,
            label="rejected region",
        )

    elif direction == "low":
        lower = threshold["lower"]

        plt.axvline(
            lower,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"threshold = {lower:.2f}",
        )

        plt.axvspan(
            x_min,
            lower,
            color="gray",
            alpha=0.15,
            label="rejected region",
        )

    elif direction == "two-sided":
        lower = threshold["lower"]
        upper = threshold["upper"]

        plt.axvline(
            lower,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"lower = {lower:.2f}",
        )

        plt.axvline(
            upper,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"upper = {upper:.2f}",
        )

        plt.axvspan(
            x_min,
            lower,
            color="gray",
            alpha=0.15,
            label="rejected region",
        )

        plt.axvspan(
            upper,
            x_max,
            color="gray",
            alpha=0.15,
        )

    plt.title(
        f"{name}: tail-energy detector at {layer}\n"
        f"individual FPR = {fpr_budget:.2%}, combined FPR = {combined_fpr:.2%}"
    )

    plt.xlabel("tail energy")
    plt.ylabel("density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    safe_layer = layer.replace(".", "_")
    safe_name = name.replace(".", "_")
    safe_individual_fpr = f"{fpr_budget:.4f}".replace(".", "p")
    safe_combined_fpr = f"{combined_fpr:.4f}".replace(".", "p")

    save_path = (
        PLOT_DIR
        / f"{safe_name}_{safe_layer}_individual_fpr_{safe_individual_fpr}_combined_fpr_{safe_combined_fpr}.png"
    )

    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close()

    return save_path


# =============================================================================
# Load data
# =============================================================================

print("=" * 80)
print("Loading datasets")
print("=" * 80)

clean_ds = load_td(dataset_path())

attack_ds = {}
for attack in ATTACKS:
    attack_ds[attack] = load_td(dataset_path(attack))

print("Dataset files loaded successfully.")

print("\n" + "=" * 80)
print("Loading corevectors")
print("=" * 80)

clean_cv = load_td(corevector_path())

attack_cv = {}
for attack in ATTACKS:
    attack_cv[attack] = load_td(corevector_path(attack))

print("Corevector files loaded successfully.")


# =============================================================================
# Clean baseline
# =============================================================================

clean_correct = clean_ds["result"][:].bool()
good_mask = clean_correct

n_total = len(good_mask)
n_good = good_mask.sum().item()
base_clean_accuracy = n_good / n_total

print("\n" + "=" * 80)
print("Clean baseline")
print("=" * 80)
print(f"Dataset:              {DATASET}")
print(f"Split:                {SPLIT}")
print(f"Model:                {MODEL}")
print(f"Clean correct:         {n_good} / {n_total}")
print(f"Base clean accuracy:   {base_clean_accuracy:.4f}")


# =============================================================================
# Compute scores for every detector
# =============================================================================

print("\n" + "=" * 80)
print("Computing detector scores")
print("=" * 80)

clean_scores_by_detector = {}
attack_scores_by_detector = {}

for det in DETECTORS:
    name = det["name"]
    layer = det["layer"]
    tail = det["tail"]

    clean_scores_by_detector[name] = compute_tail_energy(
        clean_cv,
        layer=layer,
        mask=good_mask,
        tail=tail,
    )

    attack_scores_by_detector[name] = {}

    for attack in ATTACKS:
        attack_scores_by_detector[name][attack] = compute_tail_energy(
            attack_cv[attack],
            layer=layer,
            mask=good_mask,
            tail=tail,
        )

    print(
        f"{name}: "
        f"layer={layer}, "
        f"tail={tail}, "
        f"direction={det['direction']}, "
        f"chosen FPR={det['fpr_budget']:.4f}, "
        f"samples={len(clean_scores_by_detector[name])}"
    )


# =============================================================================
# Calibrate detector using manual per-layer FPR budgets
# =============================================================================

print("\n" + "=" * 80)
print("Calibrating manual two-layer OR detector")
print("=" * 80)

thresholds = {}
individual_clean_flags = {}
combined_clean_flags = np.zeros(n_good, dtype=bool)

for det in DETECTORS:
    name = det["name"]
    direction = det["direction"]
    fpr_budget = det["fpr_budget"]

    clean_scores = clean_scores_by_detector[name]

    threshold = calibrate_threshold(
        clean_scores=clean_scores,
        direction=direction,
        fpr=fpr_budget,
    )

    flags = apply_threshold(
        scores=clean_scores,
        threshold=threshold,
        direction=direction,
    )

    thresholds[name] = threshold
    individual_clean_flags[name] = flags
    combined_clean_flags |= flags

combined_clean_fpr = combined_clean_flags.mean()
clean_accuracy_after_detector = base_clean_accuracy * (1.0 - combined_clean_fpr)

print(f"Clean accuracy before detector:  {base_clean_accuracy:.4f}")
print(f"Empirical combined clean FPR:    {combined_clean_fpr:.4f}")
print(f"Clean accuracy after detector:   {clean_accuracy_after_detector:.4f}")

print("\nIndividual detector thresholds:")

for det in DETECTORS:
    name = det["name"]
    direction = det["direction"]
    fpr_budget = det["fpr_budget"]
    threshold = thresholds[name]
    flags = individual_clean_flags[name]

    print()
    print(f"Detector:        {name}")
    print(f"Layer:           {det['layer']}")
    print(f"Direction:       {direction}")
    print(f"Chosen FPR:      {fpr_budget:.4f}")
    print(f"Empirical FPR:   {flags.mean():.4f}")
    print(f"Threshold:       {threshold}")


# =============================================================================
# Save detector plots
# =============================================================================

print("\n" + "=" * 80)
print("Saving detector plots")
print("=" * 80)

for det in DETECTORS:
    name = det["name"]

    plot_path = plot_detector_layer(
        clean_scores=clean_scores_by_detector[name],
        attack_scores=attack_scores_by_detector[name],
        threshold=thresholds[name],
        detector=det,
        combined_fpr=combined_clean_fpr,
    )

    print(f"Saved plot for {name}:\n{plot_path}")


# =============================================================================
# Evaluate attacks
# =============================================================================

print("\n" + "=" * 80)
print("Evaluating attacks")
print("=" * 80)

print(
    f"{'Attack':<10} "
    f"{'Raw ASR':>10} "
    f"{'Detect block':>14} "
    f"{'Detect fc':>10} "
    f"{'Detect either':>14} "
    f"{'ASR after det.':>16} "
    f"{'Def. Acc.':>12}"
)
print("-" * 100)

final_results = []

for attack in ATTACKS:
    adv_correct = (
        attack_ds[attack]["result"][:]
        .bool()[good_mask]
        .cpu()
        .numpy()
    )

    attack_success = ~adv_correct

    combined_adv_flags = np.zeros(n_good, dtype=bool)
    individual_adv_flags = {}
    aucs = {}

    for det in DETECTORS:
        name = det["name"]
        direction = det["direction"]

        adv_scores = attack_scores_by_detector[name][attack]

        flags = apply_threshold(
            scores=adv_scores,
            threshold=thresholds[name],
            direction=direction,
        )

        individual_adv_flags[name] = flags
        combined_adv_flags |= flags

        aucs[name] = score_auc(
            clean_scores=clean_scores_by_detector[name],
            adv_scores=adv_scores,
            direction=direction,
        )

    raw_asr = attack_success.mean()
    detect_either = combined_adv_flags.mean()
    post_detector_asr = (attack_success & ~combined_adv_flags).mean()
    defended_accuracy = (adv_correct | combined_adv_flags).mean()

    detect_block = individual_adv_flags["early_block"].mean()
    detect_fc = individual_adv_flags["fc"].mean()

    final_results.append({
        "attack": attack,
        "raw_asr": raw_asr,
        "detect_block": detect_block,
        "detect_fc": detect_fc,
        "detect_either": detect_either,
        "post_detector_asr": post_detector_asr,
        "defended_accuracy": defended_accuracy,
        "auc_block": aucs["early_block"],
        "auc_fc": aucs["fc"],
    })

    print(
        f"{attack:<10} "
        f"{raw_asr:>10.4f} "
        f"{detect_block:>14.4f} "
        f"{detect_fc:>10.4f} "
        f"{detect_either:>14.4f} "
        f"{post_detector_asr:>16.4f} "
        f"{defended_accuracy:>12.4f}"
    )

print("-" * 100)


# =============================================================================
# Print AUCs
# =============================================================================

print("\n" + "=" * 80)
print("Individual detector AUCs")
print("=" * 80)

print(
    f"{'Attack':<10} "
    f"{'AUC block1.layer.1.conv2':>28} "
    f"{'AUC fc':>12}"
)
print("-" * 60)

for r in final_results:
    print(
        f"{r['attack']:<10} "
        f"{r['auc_block']:>28.4f} "
        f"{r['auc_fc']:>12.4f}"
    )

print("-" * 60)


# =============================================================================
# Human-readable interpretation
# =============================================================================

print("\nInterpretation:")
print("This is a manual two-layer OR detector.")
print("You choose the clean FPR budget for each detector separately.")
print()
print("Detector rule:")
print("    detected = detected_at_block1.layer.1.conv2 OR detected_at_fc")
print()
print(f"Empirical combined clean FPR:   {combined_clean_fpr:.2%}")
print(f"Clean accuracy before detector: {base_clean_accuracy:.2%}")
print(f"Clean accuracy after detector:  {clean_accuracy_after_detector:.2%}")

print("\nChosen individual FPR budgets:")
for det in DETECTORS:
    name = det["name"]
    print(f"    {name}: {det['fpr_budget']:.2%}")

for r in final_results:
    print()
    print(f"{r['attack']}:")
    print(f"    Raw ASR:                 {r['raw_asr']:.2%}")
    print(f"    Detected by block layer: {r['detect_block']:.2%}")
    print(f"    Detected by fc:          {r['detect_fc']:.2%}")
    print(f"    Detected by either:      {r['detect_either']:.2%}")
    print(f"    ASR after detector:      {r['post_detector_asr']:.2%}")
    print(f"    Defended accuracy:       {r['defended_accuracy']:.2%}")

print()
print(f"Plots saved in:\n{PLOT_DIR}")