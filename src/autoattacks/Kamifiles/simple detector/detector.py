# =============================================================================
# Libraries We needed
# =============================================================================

import numpy as np
import torch
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

MODEL = "WRN28-standard"

ATTACKS = ["APGD-ce", "APGD-t"]

LAYER = "block1.layer.2.conv2" # target layer
TAIL = 100

TARGET_FPR = 0.01

# =============================================================================
# Paths
# =============================================================================

DS_BASE = Path(f"/srv/newpenny/XAI/generated_data/Kami_attacks/datasets/{DATASET}")
CV_BASE = Path(f"/srv/newpenny/XAI/generated_data/Kami_attacks/{DATASET}_{MODEL}/corevectors")
PLOT_DIR = Path(f"/srv/newpenny/XAI/generated_data/Kami_attacks/{DATASET}_{MODEL}/tail_energy")
PLOT_DIR.mkdir(parents=True, exist_ok=True) # create the plot directory if it does not already exist
 

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
        name = f"dss.{DATASET}-{SPLIT}.{MODEL}" # returns the path for the clean dataset
    else:
        name = f"dss.{DATASET}-{SPLIT}.{attack}-{MODEL}" # returns the path for the adversarial dataset

    return DS_BASE / name

# =============================================================================
# Corevector path helper
# =============================================================================

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
# Metric helpers
# =============================================================================

def compute_tail_energy(cv, layer, mask, tail):
    """
    Tail energy = sum of squared values in the final `tail` SVD/corevector
    coordinates.

    Detector score:
        score(x) = || tail(corevector(x)) ||_2^2
    """
    if layer not in cv.keys():
        available = list(cv.keys())
        raise KeyError(
            f"Layer '{layer}' not found.\n"
            f"Available layers include:\n{available[:10]} ..."
        )

    z = cv[layer][:].float()[mask]
    energy = (z[:, -tail:] ** 2).sum(dim=1)
    return energy.cpu().numpy()


def evaluate_attack(attack_name, clean_energy, adv_energy, adv_correct, threshold):
    """
    Evaluate one attack under the detector.

    Raw ASR:
        adversarial example fools classifier

    ASR after detector:
        adversarial example fools classifier AND is not detected

    Defended accuracy:
        adversarial example is correctly classified OR detected
    """
    adv_flagged = adv_energy > threshold

    attack_success = ~adv_correct

    raw_asr = attack_success.mean() # attack success rate before applying the detector
    detection_rate = adv_flagged.mean() # how many adversarial samples are rejected by the detector
    post_detector_asr = (attack_success & ~adv_flagged).mean()
    defended_accuracy = (adv_correct | adv_flagged).mean()

    labels = np.concatenate([
        np.zeros(len(clean_energy)),
        np.ones(len(adv_energy)),
    ])

    scores = np.concatenate([clean_energy, adv_energy])
    auc = roc_auc_score(labels, scores)

    return {
        "attack": attack_name,
        "auc": auc,
        "raw_asr": raw_asr,
        "detection_rate": detection_rate,
        "post_detector_asr": post_detector_asr,
        "defended_accuracy": defended_accuracy,
    }


def plot_threshold(clean_energy, attack_energy, threshold, target_fpr):
    """
    Plot KDE curves and the detector threshold.
    Everything to the right of the dashed line is rejected.
    """
    ce_energy = attack_energy["APGD-ce"]
    t_energy = attack_energy["APGD-t"]

    x_min = min(clean_energy.min(), ce_energy.min(), t_energy.min())
    x_max = max(clean_energy.max(), ce_energy.max(), t_energy.max())

    x_grid = np.linspace(x_min, x_max, 500)

    plt.figure(figsize=(7, 5))

    curves = [
        (clean_energy, "clean", "steelblue"),
        (ce_energy, "APGD-CE", "tomato"),
        (t_energy, "APGD-T", "seagreen"),
    ]

    for energy, label, color in curves:
        if np.std(energy) > 1e-8:
            kde = gaussian_kde(energy)
            y = kde(x_grid)
            plt.plot(x_grid, y, label=label, color=color)
            plt.fill_between(x_grid, y, alpha=0.15, color=color)

    plt.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"threshold = {threshold:.2f}",
    )

    plt.axvspan(
        threshold,
        x_max,
        color="gray",
        alpha=0.15,
        label="rejected region",
    )

    plt.title(
        f"Tail-energy detector at {LAYER}\n"
        f"{MODEL}, target clean FPR = {target_fpr:.0%}"
    )
    plt.xlabel("tail energy")
    plt.ylabel("density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    safe_layer = LAYER.replace(".", "_")
    safe_fpr = str(target_fpr).replace(".", "p")

    save_path = PLOT_DIR / f"threshold_{MODEL}_{safe_layer}_fpr_{safe_fpr}.png"

    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close()

    return save_path


# =============================================================================
# Load datasets
# =============================================================================

print("=" * 80)
print("Loading datasets")
print("=" * 80)

clean_ds = load_td(dataset_path())

attack_ds = {}
for attack in ATTACKS:
    attack_ds[attack] = load_td(dataset_path(attack))

print("Dataset files loaded successfully.")


# =============================================================================
# Load corevectors
# =============================================================================

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
good_mask = clean_correct # what was correct by classifier, only these used for adversarial evaluation

n_total = len(good_mask)
n_good = good_mask.sum().item()
base_clean_accuracy = n_good / n_total # original clean acc

print("\n" + "=" * 80)
print("Clean baseline")
print("=" * 80)
print(f"Dataset:              {DATASET}")
print(f"Split:                {SPLIT}")
print(f"Model:                {MODEL}")
print(f"Clean correct:         {n_good} / {n_total}")
print(f"Base clean accuracy:   {base_clean_accuracy:.4f}")


# =============================================================================
# Compute clean tail energy
# =============================================================================

print("\n" + "=" * 80)
print("Computing clean tail energy")
print("=" * 80)

clean_energy = compute_tail_energy(
    clean_cv,
    layer=LAYER,
    mask=good_mask,
    tail=TAIL,
)

print(f"Clean energy samples: {len(clean_energy)}")


# =============================================================================
# Compute adversarial tail energies and correctness
# =============================================================================

print("\n" + "=" * 80)
print("Computing adversarial tail energies")
print("=" * 80)

attack_energy = {}
attack_correct = {}

for attack in ATTACKS:
    attack_energy[attack] = compute_tail_energy(
        attack_cv[attack],
        layer=LAYER,
        mask=good_mask, # so restricted to correct clean samples
        tail=TAIL,
    )

    attack_correct[attack] = (
        attack_ds[attack]["result"][:]
        .bool()[good_mask]
        .cpu()
        .numpy()
    ) # whether each adversarial sample is correctly classified by the model, True, model ressited, false model did not

    print(f"{attack}: {len(attack_energy[attack])} samples")


# =============================================================================
# Calibrate detector threshold on clean data
# =============================================================================

print("\n" + "=" * 80)
print("Calibrating detector on clean data")
print("=" * 80)

threshold = np.quantile(clean_energy, 1.0 - TARGET_FPR)

clean_flagged = clean_energy > threshold # actual fpr
empirical_fpr = clean_flagged.mean()

clean_accuracy_after_detector = base_clean_accuracy * (1.0 - empirical_fpr) # clean accuracy after adding the detector, clean acc times 1 - fpr (probably should recompute)

print(f"Layer:                         {LAYER}")
print(f"Tail size:                     {TAIL}")
print(f"Target FPR:                    {TARGET_FPR:.4f}")
print(f"Threshold:                     {threshold:.6f}")
print(f"Empirical clean FPR:           {empirical_fpr:.4f}")
print(f"Clean accuracy after detector: {clean_accuracy_after_detector:.4f}")

print("\nDetector rule:")
print(f"Reject input if tail_energy({LAYER}) > {threshold:.6f}")


# =============================================================================
# Plot threshold figure
# =============================================================================

print("\n" + "=" * 80)
print("Saving threshold plot")
print("=" * 80)

plot_path = plot_threshold(
    clean_energy=clean_energy,
    attack_energy=attack_energy,
    threshold=threshold,
    target_fpr=TARGET_FPR,
)

print(f"Saved plot to:\n{plot_path}")


# =============================================================================
# Evaluate attacks
# =============================================================================

print("\n" + "=" * 80)
print("Evaluating attacks")
print("=" * 80)

results = []

for attack in ATTACKS:
    result = evaluate_attack(
        attack_name=attack,
        clean_energy=clean_energy,
        adv_energy=attack_energy[attack],
        adv_correct=attack_correct[attack],
        threshold=threshold,
    )

    results.append(result)


# =============================================================================
# Print final results
# =============================================================================

print("\n" + "=" * 80)
print("Final results")
print("=" * 80)

print(
    f"{'Attack':<10} "
    f"{'AUC':>8} "
    f"{'Raw ASR':>10} "
    f"{'Detect':>10} "
    f"{'ASR after det.':>16} "
    f"{'Def. Acc.':>12}"
)
print("-" * 80)

for r in results:
    print(
        f"{r['attack']:<10} "
        f"{r['auc']:>8.4f} "
        f"{r['raw_asr']:>10.4f} "
        f"{r['detection_rate']:>10.4f} "
        f"{r['post_detector_asr']:>16.4f} "
        f"{r['defended_accuracy']:>12.4f}"
    )

print("-" * 80)


# =============================================================================
# interpretation
# =============================================================================

print("\nInterpretation:")
print(f"Threshold is placed at x = {threshold:.6f} on the tail-energy plot.")
print("Everything to the right of the dashed line is detected thus rejected.")
print(f"The threshold is calibrated using clean samples only.")
print(f"Target clean FPR: {TARGET_FPR:.2%}")
print(f"Empirical clean FPR: {empirical_fpr:.2%}")
print(f"Clean accuracy before detector: {base_clean_accuracy:.2%}")
print(f"Clean accuracy after detector:  {clean_accuracy_after_detector:.2%}")

for r in results:
    print(
        f"{r['attack']}: raw ASR = {r['raw_asr']:.2%}, "
        f"ASR after detector = {r['post_detector_asr']:.2%}, "
        f"detected = {r['detection_rate']:.2%}"
    )